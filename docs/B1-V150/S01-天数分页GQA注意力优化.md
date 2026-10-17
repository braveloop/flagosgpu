# S01：天数分页 GQA 注意力优化

本篇解释当前 B1 怎样在输入处理阶段复用分页 K/V 历史，并减少在线 softmax 的状态更新。MiniCPM5-2B 的 16 个查询头共用 2 个 KV 头，每个 KV 头对应 8 个查询头；头分组与一次处理多少查询位置是两个维度。当前 kernel 处理 Q2/Q8/Q16 查询 tile，Q8/Q16 的完整共同历史按 128→64→32 个 key 连续划分，因果尾部仍逐 32 个 key 处理。下面沿分页寻址、在线累积和调用入口解释完整实现；Decode 源码见 [S02](S02-天数纯Decode直接序列索引与分页归约优化.md)，Mixed/GEMM/配置加载及唯一精度与正式结果表见 [S03 第 5 节](S03-天数混合注意力与矩阵访存及配置加载优化.md#5-精度与官方正式结果)。

## 1. 模型、运行环境与设计目标

模型使用 BF16，42 层，hidden size 2048、intermediate size 6144，查询头/KV 头/每头维度为 16/2/128，最大上下文 131072。KV 缓存每 token 为 `42 × 2 × 2 × 128 × 2 = 43008 bytes`，约 42 KiB；64 个请求各含 16K 输入和 1K 输出约需 44.6 GiB（二进制换算），尚未计入权重、激活和工作区。分页读取成本与服务中的缓存压力因此都可能影响端到端时间，估算不能替代实测峰值。

本版于 2026-10-16 在官方 OnlineLab BI-V150 32 GiB 上验证，设备为 16 SM、warp size 64、每 SM 最大 shared memory 128 KiB。Python 3.12.11、Torch 2.10.0 CoreX、CoreX 4.5.0、FlagTree 0.6.2a3+iluvatar3.6、Triton 3.6 及 site-packages 的 vLLM 0.24 vendor 保持不变，安装包 commit 未确认。FlagGems 基底为 `900531d91192bc5095a9838c9f3b03fc4b5098a0`，plugin 基底为 `c75ae7e02507a2e4cee3b3369ff142318dca2bd0`；当前 KV128 修改尚未提交，不虚构新 commit。目标进程通过原 TRITON_ATTN 枚举的受限实现子类进入，vLLM 保持只读。

本次实际包源码加载位置分别为 `/workspace/bi150-20261016-125500Z-kv128-e2e/src/FlagGems/src/flag_gems/__init__.py` 与 `/workspace/bi150-20261016-125500Z-kv128-e2e/src/vllm-plugin-FL/vllm_fl/__init__.py`；vLLM 来自 `/usr/local/lib/python3.12/site-packages/vllm/__init__.py`。两份工程包属于同一冻结源码副本，vendor vLLM保持只读。该运行身份对应下文完整实现与S03当前结果。

当前组合保留 Q2/Q8/Q16 Prefill、设备端 32/64-key Decode、Mixed q1 分派及 runtime 数量参数。这里扩大的是完整共同历史的一次 key 分块，不改变 16/2 的查询头/KV 头配置，也不改变选择器和 fallback。相比两个连续 64-key 更新，一个 128-key 更新减少循环、行最大值/指数和更新及累积器重标定次数；总 K/V 字节、QK/PV 矩阵乘工作量和逐分数指数数量并没有因此减少。10 月 12 日旧组合的实际模型 Profile 保留历史热点身份；本版正式评分后另完成单4K初始混合短窗，实际Q16与部分128-key分支的联合证据及范围见S03第1节。

## 2. 当前 Prefill、共享分页读取与在线 softmax

源文件为 [paged_gqa_prefill.py](../../FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa_prefill.py)。以下按原文件顺序连续、仅一次收录完整生产代码，每块不超过 40 行。输入输出为 BF16，最大值、指数和 `l`（归一化分母）及累积值为 FP32；兼容 guard 与未知路径 fallback 仍由 S03 的 plugin bridge 负责。

### 2.1 分页地址与在线更新

`paged_gqa_prefill.py` 第 1–35 行。

```python
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Paged BF16 GQA: reuse each KV tile across eight queries and eight heads."""

import triton
import triton.language as tl


@triton.jit
def _paged_gqa_tile_update(
    q, K, V, Table, seq, kv_head, positions, valid_rows, context,
    begin, seq_len, scale, m, l, acc,
    BLOCK_SIZE: tl.constexpr, BLOCKS: tl.constexpr, TABLE_COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr, KS0: tl.constexpr, KS1: tl.constexpr,
    KS2: tl.constexpr, VS0: tl.constexpr, VS1: tl.constexpr,
    VS2: tl.constexpr, CAUSAL_BOUNDARY: tl.constexpr,
    KV_TILE: tl.constexpr = 32, DEFER_V: tl.constexpr = False,
):
    lanes = tl.arange(0, KV_TILE)
    j = begin + lanes
    logical = j // BLOCK_SIZE
    page_valid = logical < TABLE_COLS
    if CAUSAL_BOUNDARY:
        page_valid = page_valid & (j < seq_len)
    # Read page IDs at page granularity for aligned 32/64/128-key tiles.
    # Load page IDs once at page granularity, not once for every token lane.
    if KV_TILE == 128 and (BLOCK_SIZE == 16 or BLOCK_SIZE == 32 or BLOCK_SIZE == 64):
        first_page = begin // BLOCK_SIZE
        physical = tl.full((KV_TILE,), 0, tl.int64)
        for page_offset in tl.static_range(0, KV_TILE // BLOCK_SIZE):
            page = first_page + page_offset
            page_id = tl.load(Table + seq * TABLE_STRIDE + page,
                              (page < TABLE_COLS) & (begin + page_offset * BLOCK_SIZE < seq_len), 0)
            physical = tl.where(lanes // BLOCK_SIZE == page_offset,
                                page_id.to(tl.int64), physical)
```

helper 接收已加载的 Q、分页 K/V 指针和上一轮 `m/l/acc`，先把 key 位置转换为逻辑页号。128-key tile 配合 page16/32/64 时，静态循环分别取得 8/4/2 个标量页号，每个页号只广播给属于该页的 lane，而不是把旧 page16 的四页映射直接扩大。页表容量与页起点的序列边界都参与加载 mask，随后仍统一检查物理页上下界。`physical` 使用 int64，页内偏移与实际 stride 在后续组合成地址，Q/K/V 的底层布局不被改写。

`paged_gqa_prefill.py` 第 36–64 行。

```python
    elif BLOCK_SIZE == 16:
        first_page = begin // 16
        p0 = tl.load(Table + seq * TABLE_STRIDE + first_page,
                     (first_page < TABLE_COLS) & (begin < seq_len), 0)
        p1 = tl.load(Table + seq * TABLE_STRIDE + first_page + 1,
                     (first_page + 1 < TABLE_COLS) & (begin + 16 < seq_len), 0)
        if KV_TILE == 64:
            p2 = tl.load(Table + seq * TABLE_STRIDE + first_page + 2,
                         (first_page + 2 < TABLE_COLS) & (begin + 32 < seq_len), 0)
            p3 = tl.load(Table + seq * TABLE_STRIDE + first_page + 3,
                         (first_page + 3 < TABLE_COLS) & (begin + 48 < seq_len), 0)
            physical = tl.where(lanes < 32, tl.where(lanes < 16, p0, p1),
                                tl.where(lanes < 48, p2, p3)).to(tl.int64)
        else:
            physical = tl.where(lanes < 16, p0, p1).to(tl.int64)
    elif BLOCK_SIZE == 32 and KV_TILE == 64:
        first_page = begin // 32
        p0 = tl.load(Table + seq * TABLE_STRIDE + first_page,
                     (first_page < TABLE_COLS) & (begin < seq_len), 0)
        p1 = tl.load(Table + seq * TABLE_STRIDE + first_page + 1,
                     (first_page + 1 < TABLE_COLS) & (begin + 32 < seq_len), 0)
        physical = tl.where(lanes < 32, p0, p1).to(tl.int64)
    elif BLOCK_SIZE % KV_TILE == 0:
        first_page = begin // BLOCK_SIZE
        physical = tl.load(Table + seq * TABLE_STRIDE + first_page,
                           (first_page < TABLE_COLS) & (begin < seq_len), 0).to(tl.int64)
        physical = physical + tl.zeros((KV_TILE,), tl.int64)
    else:
        physical = tl.load(Table + seq * TABLE_STRIDE + logical, page_valid, 0).to(tl.int64)
```

原 32/64-key 分支继续保留：page16 读取两页或四页，page32 的 64-key tile 读取两页。大页能完整覆盖当前 tile 时只取一个页号并广播，其余合法布局逐 lane 查页表，例如 page48 不会被错误套入对齐快路。128-key 对齐起点由外层完整历史循环保证，旧分支的映射和 fallback 不受新增分支影响。所有分支最终给出每个 lane 的物理页号，失效页由下一段统一屏蔽，而不是访问任意缓存位置。

`paged_gqa_prefill.py` 第 65–86 行。

```python
    page_valid = page_valid & (physical >= 0) & (physical < BLOCKS)
    d = tl.arange(0, 128)
    k_address = physical[None, :] * KS0 + (j % BLOCK_SIZE)[None, :] * KS1 + kv_head * KS2 + d[:, None]
    k = tl.load(K + k_address, page_valid[None, :], 0)
    if not DEFER_V:
        v_address = physical[:, None] * VS0 + (j % BLOCK_SIZE)[:, None] * VS1 + kv_head * VS2 + d[None, :]
        v = tl.load(V + v_address, page_valid[:, None], 0)
    scores = tl.dot(q, k).to(tl.float32) * scale
    allowed = page_valid[None, :]
    if CAUSAL_BOUNDARY:
        allowed = allowed & valid_rows[:, None] & (j[None, :] <= context + positions[:, None])
    scores = tl.where(allowed, scores, -float("inf"))
    m_new = tl.maximum(m, tl.max(scores, 1))
    safe_max = tl.where(m_new == -float("inf"), 0.0, m_new)
    p = tl.exp(scores - safe_max[:, None])
    alpha = tl.exp(m - safe_max)
    l = l * alpha + tl.sum(p, 1)
    if DEFER_V:
        v_address = physical[:, None] * VS0 + (j % BLOCK_SIZE)[:, None] * VS1 + kv_head * VS2 + d[None, :]
        v = tl.load(V + v_address, page_valid[:, None], 0)
    acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
    return m_new, l, acc
```

物理页必须落在 `[0, BLOCKS)`，地址由物理页、页内 token、KV 头和 head dimension 的真实 stride 组成。QK 点积转为 FP32 分数，`m_new` 和 `alpha` 把旧指数和与累积值重标定后再加入当前 tile，三个状态持续保持 FP32。PV 点积前概率仍转为 Q 的 BF16 类型，改变分块会改变浮点累积路径，因此本版采用独立 FP32 容差门，而不声称相对旧实现逐位相同。全空行用有限 `safe_max` 避免无效指数，`DEFER_V` 只把 V 加载放到分数与 softmax 后面，不直接证明硬件重叠或占用率提升。

### 2.2 任务映射与完整共同历史

`paged_gqa_prefill.py` 第 87–115 行。

```python


@triton.jit(do_not_specialize=["TOKENS", "SEQS"])
def iluvatar_paged_gqa_prefill_kernel(
    Q, K, V, O, QStart, SeqLen, Table, Scale,
    TOKENS, SEQS, BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, TABLE_COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr, QS0: tl.constexpr, QS1: tl.constexpr,
    OS0: tl.constexpr, OS1: tl.constexpr,
    KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
    VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
    QUERY_TILE: tl.constexpr, SKIP_Q1: tl.constexpr = False,
):
    tile, kv_head = tl.program_id(0), tl.program_id(1)
    # Strictly increasing prefixes also allocate harmless slots for empty queries.
    lo, hi = 0, SEQS
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        prefix = tl.load(QStart + mid) // QUERY_TILE + mid
        if tile >= prefix:
            lo = mid
        else:
            hi = mid
    seq = lo
    q_start = tl.load(QStart + seq).to(tl.int64)
    q_end = tl.load(QStart + seq + 1).to(tl.int64)
    local_tile = tile - (q_start // QUERY_TILE + seq)
    first_query = local_tile * QUERY_TILE
    q_len = q_end - q_start
```

第一维 program ID 表示查询 tile，第二维选择两个 KV 头之一，kernel 在 QStart 累计边界上二分寻找所属序列。前缀额外加入序列号，让空查询也占一个安全槽，避免相同前缀混淆归属；找到序列后再得到本序列的真实查询范围和局部 tile 位置。`TOKENS/SEQS` 是不按值特化的运行时数量，减少数量变化产生的编译体，而分页大小、stride 和查询 tile 仍是编译常量。任务顺序与原网格保持不变，本次没有反转查询 tile、压缩任务或改变空槽语义。

`paged_gqa_prefill.py` 第 116–144 行。

```python
    if (first_query < q_len) & ((q_len != 1) | (not SKIP_Q1)):
        M: tl.constexpr = QUERY_TILE * 8
        r = tl.arange(0, M)
        positions = first_query + r // 8
        heads = kv_head * 8 + r % 8
        d = tl.arange(0, 128)
        valid_rows = (positions < q_len) & (q_start + positions < TOKENS)
        q = tl.load(Q + (q_start + positions[:, None]) * QS0 + heads[:, None] * QS1 + d[None, :], valid_rows[:, None], 0)
        seq_len = tl.load(SeqLen + seq).to(tl.int64)
        context = seq_len - q_len
        m = tl.full((M,), -float("inf"), tl.float32)
        # Empty softmax has no mass; do not rely on a first update cancelling
        # an artificial nonzero normalizer initializer.
        l = tl.full((M,), 0.0, tl.float32)
        acc = tl.full((M, 128), 0.0, tl.float32)
        # The first query sees this entire prefix, so all later rows do too.
        full_end = tl.maximum(0, (context + first_query + 1) // 32 * 32)
        stop = tl.minimum(seq_len, context + tl.minimum(first_query + QUERY_TILE, q_len))
        # Q8/Q16 use wider tiles only in the prefix visible to every query row.
        wide_end = 0
        if QUERY_TILE >= 8:
            wide128_end = full_end // 128 * 128
            for begin in range(0, wide128_end, 128):
                m, l, acc = _paged_gqa_tile_update(
                    q, K, V, Table, seq, kv_head, positions, valid_rows, context,
                    begin, seq_len, Scale, m, l, acc,
                    BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
                    KS0, KS1, KS2, VS0, VS1, VS2, False, 128, True,
                )
```

有效行检查同时限制序列内位置和打包 token 容量，Mixed 已由 Decode 完成的 q1 序列只在原 SKIP_Q1 条件下跳过。每个 KV 头对应的八个查询头与 QUERY_TILE 相乘，形成 Q2/Q8/Q16 的 16/64/128 行，`m/l/acc` 从空 softmax 状态开始。`full_end` 是最早查询也完全可见的 32-key 对齐历史，因此该前缀可以关闭逐行 causal 判断。只有 Q8/Q16 将其中完整的 128-key 前缀交给新分块，末端不会跨入任一查询的未来位置。

### 2.3 历史余段、因果边界与输出

`paged_gqa_prefill.py` 第 145–168 行。

```python
            wide_end = full_end // 64 * 64
            for begin in range(wide128_end, wide_end, 64):
                m, l, acc = _paged_gqa_tile_update(
                    q, K, V, Table, seq, kv_head, positions, valid_rows, context,
                    begin, seq_len, Scale, m, l, acc,
                    BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
                    KS0, KS1, KS2, VS0, VS1, VS2, False, 64, True,
                )
        for begin in range(wide_end, full_end, 32):
            m, l, acc = _paged_gqa_tile_update(
                q, K, V, Table, seq, kv_head, positions, valid_rows, context,
                begin, seq_len, Scale, m, l, acc,
                BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
                KS0, KS1, KS2, VS0, VS1, VS2, False,
            )
        for begin in range(full_end, stop, 32):
            m, l, acc = _paged_gqa_tile_update(
                q, K, V, Table, seq, kv_head, positions, valid_rows, context,
                begin, seq_len, Scale, m, l, acc,
                BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
                KS0, KS1, KS2, VS0, VS1, VS2, True,
            )
        result = acc / tl.where(l > 0, l, 1.0)[:, None]
        tl.store(O + (q_start + positions[:, None]) * OS0 + heads[:, None] * OS1 + d[None, :], result, valid_rows[:, None])
```

128-key 前缀之后先补一个可能存在的 64-key 共同历史，再用 32-key 补齐到 full_end，三个范围连续且互不重叠。Q2 的 wide_end 保持零，全部共同历史仍走原 32-key 更新，未进入 128 路径的调用也保留原数学。最后从 full_end 到 stop 的 32-key 尾段继续按真实序列长度和每行因果边界屏蔽未来 key，同一组 FP32 状态贯穿全部范围。输出按调用方 O 的实际 stride 写回，指数和为零的行分母取一，padding 和空槽不会越过有效输出所有权。

### 2.4 调用方输出与原入口合同

`paged_gqa_prefill.py` 第 169–189 行。

```python


def paged_gqa_prefill(query, key_cache, value_cache, output, query_start_loc, seq_lens, block_table, softmax_scale, *, query_tile=8, skip_q1=False):
    """Fill caller-owned output; bridge owns feature/shape compatibility guards."""
    tokens, sequences = query.shape[0], seq_lens.numel()
    if type(query_tile) is not int or query_tile not in (2, 8, 16):
        raise ValueError("query_tile must be 2, 8 or 16")
    if type(skip_q1) is not bool:
        raise ValueError("skip_q1 must be a bool")
    if not sequences or not tokens:
        return output
    iluvatar_paged_gqa_prefill_kernel[(tokens // query_tile + sequences, 2)](
        query, key_cache, value_cache, output, query_start_loc, seq_lens,
        block_table, softmax_scale,
        tokens, sequences, key_cache.shape[0], key_cache.shape[1],
        block_table.shape[1], block_table.stride(0),
        query.stride(0), query.stride(1), output.stride(0), output.stride(1),
        *key_cache.stride()[:3], *value_cache.stride()[:3],
        query_tile, skip_q1, num_warps=8 if query_tile >= 8 else 4, num_stages=1,
    )
    return output
```

Python 入口只接受整数 Q2/Q8/Q16 和真正 bool 的 skip_q1，空 token 或空序列直接返回原 output 对象。数据存在时仍使用原网格与两个 KV 头维度，页表、缓存容量和全部 stride 均来自实际张量，不引入输出分配或逐层主机读取序列长度。Q8/Q16 仍为 8 warps、1 stage，Q2 为 4 warps、1 stage；本机 BI 的 warp 为 64，不能套用 NVIDIA 的占用率假设。bridge 的 Q2/Q8/Q16 密度选择、功能限制和回退不变；资源门实测 Q8/Q16 shared 为 48/64 KiB、reg 均 129、spill 均零，这些是编译资源而非实际 occupancy 结论。

## 3. 正确性与正式结果

独立 FP32 参照按块表取 K/V，扩展查询头后显式计算 causal softmax 与加权 V，以 `atol=0.02、rtol=0.02` 检查 BF16 输出，同时记录对原实现的差分。24 个局部用例覆盖 page16/32/48/64/128/256、64/128 边界、非连续合法 stride、失效页、共同历史与因果尾、mixed/empty/q1 跳过、零值/较大量级、调用方输出及输入不变性。Q2 和未进入 128 路径的用例与原实现逐位一致；已进入的新数学路径使用上述 FP32 容差，不能沿用任务重排的 bitwise 判据。

三组局部完整成本使用真实 B10/9q1、B64/63q1、B31/30q1 的同一层张量各重复 42 次，计入相同 native builder、bridge、partial、merge 和 Prefill 启动，并按原→新→恢复原配对。event/wall 的保守净省分别为 4.78%/4.35%、5.20%/5.98%、3.87%/3.84%，均超过对应实测噪声；这不是 42 个不同层真实数据的整模耗时，也不直接等于端到端吞吐收益。完整组合重新完成 105 题和原 wrapper 两场景各四轮后获准保留，精度、所有请求/token、统计口径与正式增益仅维护于 [S03 第 5 节](S03-天数混合注意力与矩阵访存及配置加载优化.md#5-精度与官方正式结果)。当前保留范围是 BI-V150，本版跨平台适用路径和最终双平台出口仍未完成。
