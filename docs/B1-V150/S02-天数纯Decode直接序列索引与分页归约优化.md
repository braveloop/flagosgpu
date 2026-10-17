# S02：天数纯 Decode 直接序列索引与分页归约优化

当前 B1 由 [S01 的 Q2/Q8/Q16 分页 Prefill与共享在线 softmax](S01-天数分页GQA注意力优化.md)、本文的直接序列索引 Decode，以及 [S03 的 Mixed 分派、宽输出 GEMM 和默认缓存反射](S03-天数混合注意力与矩阵访存及配置加载优化.md) 组成。纯 Decode 为每条有效序列直接找到一个查询，把长历史分成 16 段后合并；Mixed 则只在严格 eager 资格下复用这条 q1 路径，未知特征和布局返回 native 实现。最新完整精度、正式双场景四轮及唯一 A0/B1 表统一维护于 [S03 第 5 节](S03-天数混合注意力与矩阵访存及配置加载优化.md#5-精度与官方正式结果)。

## 1. 设计目标与运行身份

纯 Decode 每个有效序列只有一个查询，直接使用 sequence program ID 并从 QStart 读取其 token 位置，避免通用查询 tile 映射。每序列保留 16 个 softmax 分段，复用原 metadata 的三个 FP32 工作区；设备端长度小于两个宽 tile 时使用 32-key，否则使用 64-key。该选择只根据实际序列长度，不依赖评分 case。

实验运行在官方 OnlineLab BI-V150 32 GiB，IX-ML 4.4.0、driver/CoreX 4.5.0、Python 3.12.11、Torch 2.10.0+corex.4.5.0、FlagTree 0.6.2a3+iluvatar3.6 和 Triton 3.6 保持冻结。当前 FlagGems/plugin 基底与未提交生产修改的运行身份统一见[S01第1节](S01-天数分页GQA注意力优化.md#1-模型运行环境与设计目标)，本文Decode源码未随KV128修改。vLLM 执行镜像 site-packages 的 0.24.0 vendor，安装包 commit 未确认，只读参考 checkout 为 `ee0da84ab9e04ac7610e28580af62c365e898389`。同一评分服务完成 35 个 FULL graphs；模型为 `/workspace/MiniCPM5-2B`，端口 9031，FULL_DECODE_ONLY，显存比例 0.85，最大长度 131072，评分未启用 profiler。原 wrapper 的 RUNS=4、SKIP_FIRST=1、随机数据、ignore-eos、1024 输出 token、采样默认及 seed 参数均未改，client 的 plugin、GPU 与导入作用域保持原流程。数值源码、实际 kernel/config、模型及数值库冻结，精度与 CPU SQL 的复用边界见 [S03 第 5 节](S03-天数混合注意力与矩阵访存及配置加载优化.md#5-精度与官方正式结果)。

## 2. Decode 完整生产实现与注册

共享页读取及在线 softmax helper 已在 S01 连续收录。下面连续完整收录 [paged_gqa_decode.py](../../FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa_decode.py)，每段最多 40 行。plugin bridge 的完整源码只在 S03 收录，本文保留注册入口及其所有权关系。

### 2.1 分段范围与在线状态

```python
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Direct sequence-indexed paged GQA Decode using caller-owned split buffers."""

import triton
import triton.language as tl

from .paged_gqa_prefill import _paged_gqa_tile_update


@triton.jit
def _paged_gqa_decode_segment(
    q, K, V, Table, seq, kv_head, segment, positions, valid_rows, seq_len, Scale,
    BLOCK_SIZE: tl.constexpr, BLOCKS: tl.constexpr,
    TABLE_COLS: tl.constexpr, TABLE_STRIDE: tl.constexpr,
    KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
    VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
    KV_TILE: tl.constexpr,
):
    tiles_per_segment = tl.cdiv(seq_len, 16 * KV_TILE)
    begin = segment * tiles_per_segment * KV_TILE
    stop = tl.minimum(seq_len, begin + tiles_per_segment * KV_TILE)
    full_end = stop // KV_TILE * KV_TILE
    m = tl.full((16,), -float("inf"), tl.float32)
    l = tl.full((16,), 0.0, tl.float32)
    acc = tl.full((16, 128), 0.0, tl.float32)
```

Decode 分段 helper 复用 S01 的分页读取和在线 softmax 更新，输入包括当前 Q、序列长度、分段号及选定的 32/64-key tile 宽度。`tiles_per_segment=ceil(seq_len/(16*KV_TILE))` 让 16 个分段覆盖完整历史，`begin/stop` 划定本段范围，`full_end` 将末端向下对齐为完整 tile。比如 `seq_len=130、KV_TILE=64` 时每段跨度 64，前两段完整、第三段含 2 个 key，其余段为空。`m=-inf、l=0、acc=0` 表示尚未看到任何合法 key，三者均使用 FP32，空段也必须返回这组中性状态。分段将长历史交给多个 program 并行计算，具体并行收益由完整组合测量，页表和 stride 等编译常量继续保证地址布局一致。下面partial与merge两核的TOKENS改为runtime标量并使用do_not_specialize，只取消实际token数量的按值特化；共享分段helper没有这个参数，其编译策略保持原样。

```python
    for tile in range(begin, full_end, KV_TILE):
        m, l, acc = _paged_gqa_tile_update(
            q, K, V, Table, seq, kv_head, positions, valid_rows, seq_len - 1,
            tile, seq_len, Scale, m, l, acc,
            BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
            KS0, KS1, KS2, VS0, VS1, VS2, False, KV_TILE, KV_TILE == 64,
        )
    for tile in range(tl.maximum(begin, full_end), stop, KV_TILE):
        m, l, acc = _paged_gqa_tile_update(
            q, K, V, Table, seq, kv_head, positions, valid_rows, seq_len - 1,
            tile, seq_len, Scale, m, l, acc,
            BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
            KS0, KS1, KS2, VS0, VS1, VS2, True, KV_TILE, KV_TILE == 64,
        )
    return m, l, acc
```

第一个循环只遍历完全落在本段内的 tile，这些 key 对唯一的 Decode 查询都属于可见历史，所以 helper 可以关闭逐行因果判断。第二个循环从 `max(begin, full_end)` 开始补尾段，保留真实序列长度和查询位置 mask，避免最后不足一个 tile 时读取或计入 padding。两次调用都传入 `seq_len-1` 作为 Decode 的历史边界，并沿用同一组 `m/l/acc` 在线重标定状态。宽 tile 传入 `DEFER_V=True` 调整 V 的读取顺序，分页地址复用仍由共享 helper 执行。两个循环连续覆盖本段且不重复 key，返回的未归一化 FP32 状态稍后写入 partial 工作区，供跨段合并保持与整体 softmax 一致。

### 2.2 直接序列定位与设备长度选择

```python


@triton.jit(do_not_specialize=["TOKENS"])
def iluvatar_paged_gqa_decode_partial_kernel(
    Q, K, V, QStart, SeqLen, Table, Partial, Max, Sum, Scale,
    TOKENS, BLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    TABLE_COLS: tl.constexpr, TABLE_STRIDE: tl.constexpr,
    QS0: tl.constexpr, QS1: tl.constexpr,
    KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
    VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
):
    seq, kv_head, segment = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    q_start = tl.load(QStart + seq).to(tl.int64)
    q_end = tl.load(QStart + seq + 1).to(tl.int64)
    # FULL graphs may contain holes or tail padding; sequence is not token index.
    if (q_end - q_start == 1) & (q_start >= 0) & (q_start < TOKENS):
        r = tl.arange(0, 16)
        d = tl.arange(0, 128)
        heads = kv_head * 8 + r % 8
        valid_rows = r < 8
        positions = r // 8
        q = tl.load(Q + q_start * QS0 + heads[:, None] * QS1 + d[None, :],
                    valid_rows[:, None], 0)
        seq_len = tl.maximum(tl.load(SeqLen + seq).to(tl.int64), 0)
        # Below two wide tiles, use narrow tiles to avoid padded key work.
        if seq_len < 2 * 64:
            m, l, acc = _paged_gqa_decode_segment(
                q, K, V, Table, seq, kv_head, segment, positions, valid_rows, seq_len, Scale,
                BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
                KS0, KS1, KS2, VS0, VS1, VS2, 32,
            )
        else:
            m, l, acc = _paged_gqa_decode_segment(
                q, K, V, Table, seq, kv_head, segment, positions, valid_rows, seq_len, Scale,
                BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
                KS0, KS1, KS2, VS0, VS1, VS2, 64,
            )
```

这里的三维 program ID 直接给出序列、KV 头和 16 个分段号，省去通用 Prefill 的查询 tile 二分定位。真正的 Q 地址来自 `QStart[seq]`，而 `QStart[seq+1]-QStart[seq]==1` 和 token 容量检查共同判定该槽是否有合法 Decode 查询。FULL graph 可以含 holes 或尾 padding，例如第 3 个序列槽的 `q_start` 可能为 1，此时读取 token 1 才正确，使用序列号读 Q 会错位。`heads=kv_head*8+r%8` 把一个 KV 头映射到八个查询头，16 行矩阵仅前八行有效，其余行被 mask。序列长度从设备 `SeqLen` 读取并截断为非负值，小于两个 64-key tile 时采用 32-key 减少补齐 key 的计算，其余长度使用 64-key。分支依据是实际历史长度，查询地址、分页映射和分段状态均沿既有 contract 传递。

```python
        # Write every split for valid queries, including empty splits on replay.
        state_offset = seq * 16 * 16 + heads * 16 + segment
        tl.store(Max + state_offset, m, valid_rows)
        tl.store(Sum + state_offset, l, valid_rows)
        tl.store(Partial + state_offset[:, None] * 128 + d[None, :], acc,
                 valid_rows[:, None])


@triton.jit(do_not_specialize=["TOKENS"])
def iluvatar_paged_gqa_decode_merge_kernel(
    QStart, Partial, Max, Sum, O,
    TOKENS, OS0: tl.constexpr, OS1: tl.constexpr,
):
    seq, head = tl.program_id(0), tl.program_id(1)
    q_start = tl.load(QStart + seq).to(tl.int64)
    q_end = tl.load(QStart + seq + 1).to(tl.int64)
    if (q_end - q_start == 1) & (q_start >= 0) & (q_start < TOKENS):
        segments = tl.arange(0, 16)
        d = tl.arange(0, 128)
        state_offset = seq * 16 * 16 + head * 16 + segments
        m = tl.load(Max + state_offset)
        l = tl.load(Sum + state_offset)
        maximum = tl.max(m, 0)
        safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
        weights = tl.where(l > 0, tl.exp(m - safe_maximum), 0.0)
        mass = tl.sum(l * weights, 0)
        partial = tl.load(Partial + state_offset[:, None] * 128 + d[None, :])
        result = tl.sum(partial * weights[:, None], 0) / tl.where(mass > 0, mass, 1.0)
        tl.store(O + q_start * OS0 + head * OS1 + d, result)
```

`state_offset=seq*16*16+head*16+segment` 将每条序列的 16 个查询头和每头 16 段映射到原 metadata 工作区，Partial 再按每段 128 个通道展开。每个有效查询的全部分段都写入 `Max/Sum/Partial`，即使该段没有 key，也写入 `-inf/0/0`。这一步防止固定地址的 graph 从长历史回放到短历史时，未被覆盖的旧分段参与下一次归约。merge 使用同样的 QStart 有效性检查读取 16 段，先取公共最大值，再用 `exp(m-safe_maximum)` 把每段质量和累积值重标定到同一基准。零质量段权重为零，全部为空时分母为 1，因此输出为零且不会产生无效归一化。结果写到真实 `q_start` 和 output stride 指定的地址，工作区按序列编号组织与 Q 的 token 编号可以安全分离。

### 2.3 原 workspace 与调用方输出

```python


def paged_gqa_decode(query, key_cache, value_cache, output, query_start_loc,
                     seq_lens, block_table, softmax_scale, partial, maximum, expsum):
    """Fill output using the original metadata's contiguous FP32 workspaces."""
    tokens, sequences = query.shape[0], seq_lens.numel()
    if not sequences or not tokens:
        return output
    iluvatar_paged_gqa_decode_partial_kernel[(sequences, 2, 16)](
        query, key_cache, value_cache, query_start_loc, seq_lens, block_table,
        partial, maximum, expsum, softmax_scale,
        tokens, key_cache.shape[0], key_cache.shape[1],
        block_table.shape[1], block_table.stride(0),
        query.stride(0), query.stride(1),
        *key_cache.stride()[:3], *value_cache.stride()[:3],
        num_warps=4, num_stages=1,
    )
    iluvatar_paged_gqa_decode_merge_kernel[(sequences, 16)](
        query_start_loc, partial, maximum, expsum, output,
        tokens, output.stride(0), output.stride(1), num_warps=4,
    )
    return output
```

Python 入口接收调用方的 output 和 metadata 已持有的三个 FP32 分段缓冲，空序列或空 token 直接返回原 output。第一个 launch 按 `(sequences, 2, 16)` 计算分段，第二个按 `(sequences, 16)` 合并每个查询头，前后通过同一组 partial/max/expsum 传递状态。真实缓存容量、页大小和 stride 全部来自传入张量，因而地址计算支持 guard 已允许的布局。入口只从张量 shape 取得数量，序列长度选择仍在设备完成，可减少主机读取长度带来的同步。两个 launch 不新建工作区，graph 回放继续使用原 metadata 缓冲地址，返回值也保留调用方 output 的对象所有权。

### 2.4 注册与外部 override 所有权

```python


class IluvatarPagedGQAPrefillBackend(TritonAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return IluvatarPagedGQAPrefillImpl


def register_paged_gqa_prefill_backend():
    global _reported_external_override
    if getattr(current_platform, "vendor_name", None) != "iluvatar":
        return
    existing = _ATTN_OVERRIDES.get(AttentionBackendEnum.TRITON_ATTN)
    if existing == BACKEND_PATH:
        return
    if existing is not None:
        if not _reported_external_override:
            logger.info("Keeping existing TRITON_ATTN override %s; own Prefill not registered", existing)
            _reported_external_override = True
        return
    register_backend(AttentionBackendEnum.TRITON_ATTN, BACKEND_PATH)
    logger.info("Registered own iluvatar paged-GQA Prefill/Decode with native Triton fallbacks")
```

backend 子类只替换 `get_impl_cls` 返回的实现类，注册目标仍是原 `TRITON_ATTN` 枚举，上层选择器可以沿既有规则找到它。注册函数先检查 iluvatar 厂商，其他平台直接返回，避免同名枚举影响 C500 或其他后端。注册表已指向自身时直接返回，保证多次平台注册调用不会重复改变状态。若已有外部 override，函数保留原条目并最多提示一次，尊重已有实现对该枚举的所有权。仅在没有 override 时才写入 `BACKEND_PATH`，计算时仍由子类 guard 和父类 fallback 决定使用哪条路径，logger 只记录注册结果。

[register_ops.py](../../vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/register_ops.py) 的 `register_builtins` 中增加下列导入：

```python
    from .paged_gqa_prefill import register_paged_gqa_prefill_backend
```

这行导入位于 iluvatar 的 `register_builtins`，把受限 Attention 注册函数带入现有平台注册入口。导入目标是 plugin bridge，函数体随后会查询 `current_platform` 和 override 注册表，因此厂商判断仍由同一处负责。这里只取得函数引用，Attention kernel 的按需导入保留在实际 `forward` 分支中。复用原入口可以让初始化顺序保持可追踪，真正修改注册表的时机由下面的调用决定。

在创建 backend 与取得 availability 后调用：

```python
    register_paged_gqa_prefill_backend()
```

这次调用在已有 backend 创建和 availability 获取之后执行，将受限实现纳入原初始化顺序。availability 的值没有被用作这次调用的布尔条件，实际注册边界来自函数内的 iluvatar 厂商检查和已有 override 检查。多次调用由注册函数的幂等判断处理，有外部实现时保留该实现。这个入口负责接线，是否计算自研 Attention 仍由原 selector、注册结果及每次 `forward` 的兼容 guard 共同决定。

## 3. 精度与官方正式结果

完整精度、统计口径与唯一 A0/B1 正式四行表统一见 [S03 第 5 节](S03-天数混合注意力与矩阵访存及配置加载优化.md#5-精度与官方正式结果)。
