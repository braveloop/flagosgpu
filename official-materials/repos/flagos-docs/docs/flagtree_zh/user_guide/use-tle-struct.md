# 使用 TLE-Struct

本节介绍如何使用 TLE-Struct。TLE-Struct 在 trition_3.6.x 分支上可用。

## GPU 内存管理

您可以使用以下操作来管理 GPU 的内存。

### tle.gpu.memory_space

为给定张量指定 memory_space：

```{code-block} python
x = ...
x = tle.gpu.memory_space(x, "shared_memory")
```

### tle.gpu.alloc

以下示例演示如何在 GPU 的高速片上 SMEM（共享内存）中预留一块维度为 `XBLOCK * YBLOCK`、数据类型为 `float32` 的内存。

```{code-block} python
a_smem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=tl.float32,
                      layout=None, scope=tle.gpu.storage_kind.smem)
```

### tle.gpu.local_ptr

获取内存指针。

```{code} python
# 获取 a_smem[0,:] 的指针: [(0, 0), (0, 1)...(0, YBLOCK-1)]
a_smem_ptrs = tle.gpu.local_ptr(a_smem,
    indices=(tl.broadcast(0, [YBLOCK]), tl.arrange(0, YBLOCK)))
```

- 签名: `tle.gpu.local_ptr(buffer, indices=None) -> tl.tensor | tl.ptr`
- 用途: 在共享内存缓冲区上构建任意形状的指针视图，用于 `tl.load`/`tl.store`。
- 参数:
  - `buffer`: 由 `tle.gpu.alloc` 返回的 buffered_tensor（SMEM / TMEM）。
  - `indices`: 可选的整数张量元组，其长度必须等于 `rank(buffer)`，且每个张量必须具有相同的形状。如果省略或传入 `None`，后端将按照完整索引语义处理。
- 语义:
  - 当显式提供 `indices` 时，输出指针张量的形状等于索引的公共（广播后）形状。
  - 对于输出形状中的每个逻辑索引 `(i0, i1, ...)`，相应的指针指向 `buffer[indices0(i0, ...), indices1(i0, ...), ...]`。
  - 当 `indices=None` 时，返回覆盖整个 `buffer` 的完整视图指针：
    - 如果秩 > 0，返回形状等于 `shape(buffer)` 的指针张量。
    - 如果秩 = 0，返回标量指针。
  - 返回的指针驻留在共享内存地址空间（LLVM 地址空间 3）。索引必须是整数类型（例如 i32、i64 等），并在下层时标准化为 i32。
  - 内存布局按行主序线性化（最后一维变化最快）。共享内存布局和编码遵循缓冲区的 memdesc。

- 示例 1: 1D 切片

  ```{code-block} python
  smem = tle.alloc([BLOCK], dtype=tl.float32, scope=tle.smem)
  # 切片 [offset, offset + SLICE)
  idx = offset + tl.arange(0, SLICE)
  slice_ptr = tle.local_ptr(smem, (idx,))
  vals = tl.load(slice_ptr)
  ```

- 示例 2: K 维分块（矩阵切片）

  ```{code-block} python
  smem_a = tle.alloc([BM, BK], dtype=tl.float16, scope=tle.smem)
  # 切片 (BM, KW)，其中 KW 是 K 维切片
  rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, KW))
  cols = tl.broadcast_to(tl.arange(0, KW)[None, :] + k_start, (BM, KW))
  a_slice = tle.local_ptr(smem_a, (rows, cols))
  a_vals = tl.load(a_slice)
  ```
  
- 示例 3: 任意 gather 视图
  
  ```{code-block} python
      smem = tle.alloc([H, W], dtype=tl.float32, scope=tle.smem)
      # 每行取一个偏移列
      rows = tl.broadcast_to(tl.arange(0, H)[:, None], (H, SLICE))
      cols = tl.broadcast_to(1 + tl.arange(0, SLICE)[None, :], (H, SLICE))
      gather_ptr = tle.local_ptr(smem, (rows, cols))
      out = tl.load(gather_ptr)
  ```

支持的下游操作：

- `tl.load`
- `tl.store`
- `tl.atomic_add`、`atomic_and`、`atomic_cas`、`atomic_max`、`atomic_min`、`atomic_or`、`atomic_xchg`、`atomic_xor`

实践注意事项：

- 原子操作的可用性取决于元素数据类型（dtype）和后端硬件的能力。建议优先使用在目标硬件上明确验证支持的整数或浮点类型。

- 对于涉及 local_ptr 的 load-after-store 冒险，TLE 后端 pass `TleInsertLocalPointerBarriers` 会自动插入必要的内存屏障。仅在使用超出此 pass 范围的自定义同步模式时才需要手动插入屏障。

- 示例 4: 在同一 local_ptr 上执行 load、store 和原子操作。

```{code-block} python
smem_i32 = tle.gpu.alloc([BLOCK], dtype=tl.int32, scope=tle.gpu.smem)
ptr = tle.gpu.local_ptr(smem_i32, (tl.arange(0, BLOCK),))

tl.store(ptr, tl.zeros([BLOCK], dtype=tl.int32))
tl.atomic_add(ptr, 1)
vals = tl.load(ptr)
```

### tle.gpu.local_ptr（用于远程）

- 签名: `tle.gpu.local_ptr(remote_buffer, indices=None) -> tl.tensor | tl.ptr`
- 用途: 在由 `tle.remote(...)` 返回的远程共享/本地缓冲区上构造指针视图。
- 输入:
  - `remote_buffer`: 由 `tle.remote(buffer, shard_id, scope)` 返回，其中 `buffer` 通常通过 `tle.gpu.alloc` 分配。
  - `indices`: 与本地模式一致（`None` 表示完整视图，或可提供形状匹配的整数张量元组）。
- 语义:
  - 指针的形状、索引行为和线性化规则与本地 `tle.gpu.local_ptr` 相同。
  - 地址解析路由到由 `shard_id` 指定的远程分片。
  - 对于需要排序保证的跨分片读/写，请结合使用 `tle.distributed_barrier(...)`。
  
读取相邻分片上的远程 SMEM 切片。

```{code-block} python
smem = tle.gpu.alloc([BM, BK], dtype=tl.float16, scope=tle.gpu.storage_kind.smem)
remote_smem = tle.remote(smem, shard_id=(node_rank, next_device), scope=mesh)

rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
cols = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
remote_ptr = tle.gpu.local_ptr(remote_smem, (rows, cols))

vals = tl.load(remote_ptr)
```

### tle.gpu.copy

以下示例演示如何将数据切片从低速 GMEM（全局内存）加载到高速片上 SMEM 中。

- 从源复制:
  - `a_ptrs`: GMEM 中的基指针
  - `ystride_a * yoffs[None, :]`: 添加到基指针的偏移向量。
    - `yoffs[None, :]`: 表示 Y 轴偏移的范围，广播为行向量。
    - `ystride_a`: 源布局中行之间的步长。这计算了倾向于从 GMEM 加载的 2D 块的确切地址。
- 到目标:
  - `a_smem`: 之前分配的 SMEM 缓冲区。数据将写入此处，供此块中的线程快速访问。

```{code-block} python
tle.gpu.copy(a_ptrs + ystride_a * yoffs[None, :], a_smem, [XBLOCK, YBLOCK])
```

## 执行编排

### tle.gpu.warp_specialize

`tle.gpu.warp_specialize` 用于在同一 CTA 内显式创建 warp 专用区域，将不同的 JIT 函数放入不同的 warp 分区。典型用例是将 TMA/cp.async 生产者、WGMMA 消费者和 epilogue/reduction 等任务分离，并通过 `tle.pipe` 或其他显式同步原语在它们之间传递共享内存数据。

- **签名**: `tle.gpu.warp_specialize(functions_and_args, worker_num_warps, worker_num_regs)`
- **参数**:
  - `functions_and_args`: `[(fn0, args0), (fn1, args1), ...]`。第 0 项进入默认分区；后续项进入工作分区。
  - `worker_num_warps`: 工作分区的 warp 数量列表；长度必须等于 `len(functions_and_args) - 1`。
  - `worker_num_regs`: 工作分区请求的寄存器数量列表；长度必须等于 `len(functions_and_args) - 1`。
- **语义**:
  - 每个 `args` 必须是元组；普通的 Python `int`/`float`/`bool`/`tl.dtype` 作为 constexpr 传递。
  - 默认分区可以返回值；`tle.gpu.warp_specialize(...)` 的返回值来自默认分区；工作分区仅执行副作用并以 warp return 结束。
  - 工作分区的被调用者将携带相应的 `"ttg.num-warps"` 属性，区域将记录 `requestedRegisters`。
  - 捕获的工作参数在 IR 中会去重；多个工作线程可以共享同一个管道端点或缓冲区句柄。
  - `warp_specialize` 本身不提供数据可见性保证；生产者/消费者排序应通过 `tle.pipe` 的 commit/wait/release、屏障或其他同步原语来表达。

示例: 生产者分区加载共享内存，消费者工作线程进行计算。

```{code-block} python
@triton.jit
def producer(writer, x_ptr, n_tiles: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for i in tl.range(0, n_tiles):
        slot = writer.acquire(i)
        vals = tl.load(x_ptr + i * BLOCK + offs)
        tl.store(tle.gpu.local_ptr(slot.tile), vals)
        writer.commit(i)


@triton.jit
def consumer(reader, out_ptr, n_tiles: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.range(0, n_tiles):
        ready = reader.wait(i)
        tile = tl.load(tle.gpu.local_ptr(ready.slot.tile))
        acc += tile
        reader.release(i)
    tl.store(out_ptr + offs, acc)


@triton.jit
def kernel(x_ptr, out_ptr, n_tiles: tl.constexpr, BLOCK: tl.constexpr):
    smem = tle.gpu.alloc([2, BLOCK], dtype=tl.float32, scope=tle.gpu.smem)
    pipe = tle.pipe(capacity=2, scope="cta", name="x_pipe", tile=smem)

    tle.gpu.warp_specialize(
        [
            (producer, (pipe.writer(), x_ptr, n_tiles, BLOCK)),
            (consumer, (pipe.reader(), out_ptr, n_tiles, BLOCK)),
        ],
        [4],      # consumer worker 使用 4 个 warp
        [168],    # consumer worker 请求的寄存器数
    )
```

示例: 多个 worker 与 SPMC 管道配对。

```{code-block} python
tile = tle.gpu.alloc([2, BM, BK], dtype=tl.float16, scope=tle.gpu.smem)
pipe = tle.pipe(
    capacity=2,
    scope="cta",
    name="spmc_tile",
    readers=("qk", "value"),
    tile=tile,
)

tle.gpu.warp_specialize(
    [
        (load_tile_producer, (pipe.writer(), a_desc, b_desc)),
        (qk_consumer, (pipe.reader("qk"), acc_qk)),
        (value_consumer, (pipe.reader("value", fields=("tile",)), acc_v)),
    ],
    [4, 4],
    [240, 168],
)
```

## DSA 内存管理和数据移动

### tle.dsa.alloc

签名: `tle.dsa.alloc(shape, dtype, mem_addr_space)`
用途: 在指定的内存地址空间中分配 DSA 本地缓冲区。
华为 Ascend 暴露的地址空间:

- `tle.dsa.ascend.UB`
- `tle.dsa.ascend.L1`
- `tle.dsa.ascend.L0A`
- `tle.dsa.ascend.L0B`
- `tle.dsa.ascend.L0C`

```{code-block} python
a_ub = tle.dsa.alloc([XBLOCK, YBLOCK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
b_l1 = tle.dsa.alloc([XBLOCK, YBLOCK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.L1)
```

### tle.dsa.copy

签名: `tle.dsa.copy(src, dst, shape, inter_no_alias=False)`
用途: 在 GMEM 指针和 DSA 本地缓冲区之间执行显式数据移动（双向）。

```{code-block} python
tle.dsa.copy(x_ptrs, a_ub, [tail_m, tail_n])    # GMEM → 本地缓冲区  
tle.dsa.copy(a_ub, out_ptrs, [tail_m, tail_n])  # 本地缓冲区 → GMEM
```

### tle.dsa.local_ptr  

- 签名: `tle.dsa.local_ptr(buffer, indices=None) -> tl.tensor | tl.ptr`  
- 用途: 在 DSA 本地缓冲区（例如 UB 或 L1）上构造指针视图，以启用显式本地内存访问模式。  

- 参数:  
  - `buffer`: DSA 缓冲张量，通常通过 `tle.dsa.alloc` 分配。  
  - `indices`: 可选的整数张量元组；如果省略或设置为 `None`，则使用完整索引空间（完整视图语义）。  

语义:  
  指针视图模型与 `tle.gpu.local_ptr` 相同（相同的形状和索引规则）。  
  适用于需要显式指针物化的 DSA 本地访问模式。  

```{code-block} python
a_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
cols = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
a_ptr = tle.dsa.local_ptr(a_ub, (rows, cols))
a_val = tl.load(a_ptr)
```

### tle.dsa.local_ptr（用于远程）  

- 签名: `tle.dsa.local_ptr(remote_buffer, indices=None) -> tl.tensor | tl.ptr`  
- 用途: 在由 `tle.remote(...)` 返回的远程 DSA 本地缓冲区上构造指针视图。  

- 输入:  
  - `remote_buffer`: 由 `tle.remote(dsa_buffer, shard_id, scope)` 返回。  
  - `indices`: 与本地 DSA 情况相同的语义。  

- 语义:  
  - 保持与本地 DSA 变体相同的指针视图规则。  
  - 解引用指针将内存访问路由到由 `shard_id` 标识的远程分片。  
  - 当需要跨分片排序时，请结合使用 `tle.distributed_barrier(...)`。  

```{code-block} python
a_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
remote_a_ub = tle.remote(a_ub, shard_id=peer_rank, scope=mesh)

rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
cols = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
remote_ptr = tle.dsa.local_ptr(remote_a_ub, (rows, cols))
remote_val = tl.load(remote_ptr)
```

### tle.dsa.to_tensor 和 tle.dsa.to_buffer

- `tle.dsa.to_tensor(buffer, writable=True)`: 将 DSA 缓冲区转换为张量视图以参与张量表达式。
- `tle.dsa.to_buffer(tensor, space)`: 将张量值转换回指定地址空间中的 DSA 缓冲区。

```{code-block} python
c_val = tle.dsa.to_tensor(c_ub, writable=True)
result = c_val * 0.5
d_ub = tle.dsa.to_buffer(result, tle.dsa.ascend.UB)
tle.dsa.copy(d_ub, out_ptrs, [tail_m, tail_n])
```

## 向量运算符（缓冲区形式）

### tle.dsa.add、tle.dsa.sub、tle.dsa.mul、tle.dsa.div、tle.dsa.max 和 tle.dsa.min

内置运算符:
`tle.dsa.add`
`tle.dsa.sub`
`tle.dsa.mul`
`tle.dsa.div`
`tle.dsa.max`
`tle.dsa.min`

通用签名:  
`tle.dsa.(lhs, rhs, out)`

计算模型:  
对 DSA 本地缓冲区执行逐元素二元运算。

形状规则:

- `lhs`、`rhs` 和 `out` 的秩和形状必须相同。
- 此 API 层默认不执行隐式广播。

- 类型规则:
  - 实践中，建议所有三个操作数使用相同的数据类型（`dtype`）。
  - 整数类型通常用于索引/计数路径，而浮点类型通常用于激活/数值计算路径。

- 地址空间规则:
  - 缓冲区必须分配在后端支持的 DSA 本地地址空间中（例如 UB/L1 组合）。
  - 热数据应尽可能保留在本地内存中，以避免不必要的全局内存（GMEM）往返。

运算符语义:
`tle.dsa.add(lhs, rhs, out)`: `out = lhs + rhs`
`tle.dsa.sub(lhs, rhs, out)`: `out = lhs - rhs`
`tle.dsa.mul(lhs, rhs, out)`: `out = lhs * rhs`
`tle.dsa.div(lhs, rhs, out)`: `out = lhs / rhs`（精度和舍入行为取决于后端实现）
`tle.dsa.max(lhs, rhs, out)`: `out = max(lhs, rhs)`
`tle.dsa.min(lhs, rhs, out)`: `out = min(lhs, rhs)`

就地/重用建议:

- 输出缓冲区可以在多个计算步骤中重用，例如 `tle.dsa.mul(tmp, b, tmp)`。
- 除非后端明确保证别名安全，否则输入和输出缓冲区不应任意共享内存。

示例 1: 算术链 `((a - b) * b) / scale`

```{code-block} python
a_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
b_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
scale_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
tmp_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
out_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)

tle.dsa.copy(a_ptrs, a_ub, [BM, BK])
tle.dsa.copy(b_ptrs, b_ub, [BM, BK])
tle.dsa.copy(scale_ptrs, scale_ub, [BM, BK])

tle.dsa.sub(a_ub, b_ub, tmp_ub)        # tmp = a - b
tle.dsa.mul(tmp_ub, b_ub, tmp_ub)      # tmp = tmp * b
tle.dsa.div(tmp_ub, scale_ub, out_ub)  # out = tmp / scale

tle.dsa.copy(out_ub, out_ptrs, [BM, BK])
```

示例 2: 使用 `max` + `min` 进行 Clamp

```{code-block} python
x_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
floor_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
ceil_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
tmp_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
y_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)

tle.dsa.copy(x_ptrs, x_ub, [BM, BK])
tle.dsa.copy(floor_ptrs, floor_ub, [BM, BK])
tle.dsa.copy(ceil_ptrs, ceil_ub, [BM, BK])

tle.dsa.max(x_ub, floor_ub, tmp_ub)    # tmp = max(x, floor)
tle.dsa.min(tmp_ub, ceil_ub, y_ub)     # y = min(tmp, ceil)

tle.dsa.copy(y_ub, y_ptrs, [BM, BK])
```

## 循环和提示

### tle.dsa.pipeline、tle.dsa.parallel 和 tle.dsa.hint

循环和提示 API 包括:

- `tle.dsa.pipeline(...)`
- `tle.dsa.parallel(...)`
- `tle.dsa.hint(...)` — 以上下文管理器 `with tle.dsa.hint(...)` 的形式提供编译时提示。

```{code-block} python
with tle.dsa.hint(inter_no_alias=True):
    tle.dsa.copy(x_ptr + offs, a_ub, [tail_size], inter_no_alias=True)
```

## 切片和视图

### tle.dsa.extract_slice、tle.dsa.insert_slice、tle.dsa.extract_element 和 tle.dsa.subview

切片和视图 API 包括:

- `tle.dsa.extract_slice`
- `tle.dsa.insert_slice`
- `tle.dsa.extract_element`
- `tle.dsa.subview`

```{code-block} python
sub = tle.dsa.extract_slice(full, offsets=(0, k0), sizes=(BM, BK), strides=(1, 1))
full = tle.dsa.insert_slice(full, sub, offsets=(0, k0), sizes=(BM, BK), strides=(1, 1))
elem = tle.dsa.extract_element(sub, indice=(i, j))
```
