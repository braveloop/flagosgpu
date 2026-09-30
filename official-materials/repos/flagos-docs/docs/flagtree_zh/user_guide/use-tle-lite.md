# 使用 TLE-Lite

本节介绍如何使用 TLE-Lite。TLE-Lite 在 trition_3.6.x 分支上可用。

## 内存管理

您可以使用以下操作来管理内存。

### tle.load

`tle.load` 从 GMEM 异步加载张量。它支持异步提示。

```{code-block} python
x = tle.load(..., is_async=True)
```

## 张量切片

根据指定的子切片形状将输入张量分割为子切片网格，并提取给定坐标处的子切片。
GPU：支持提取到寄存器和共享内存中。

### tle.extract_tile

根据指定的子切片形状将输入张量分割为子切片网格，并提取给定坐标处的子切片。

支持提取到寄存器和共享内存中。

```{code-block} python
# x 是 [4, 4]
# z 是 [2, 2]
# 将 x 分割为 shape=[2, 2] 的子切片网格，并提取 [0, 0] 处的子切片
z = x.extract_tile(index=[0, 0], shape=[2, 2])
```

### tle.insert_tile

根据子切片形状将输入张量分割为子切片网格，并用新的切片更新指定坐标处的子切片。

支持从寄存器和共享内存进行更新。

```{code-block} python
# x 是 [4, 4]，y 是 [2, 2]，z 是 [4, 4]
# 将 x 分割为 shape=[2, 2] 的子切片，用 y 更新 [0, 0] 子切片，并返回完整的 [4, 4] 张量
z = x.insert_tile(y, index=[0, 0])
```

## 扫描和排序操作

扫描和排序操作提供了部分张量原语，如前缀、排名和选择，适用于基于直方图的 top-k、流压缩以及块级排序和分桶场景。

TLE-Lite 将这些操作保持为高级语义，而不是将其绑定到特定的硬件实现：用户描述扫描和排序意图，后端根据硬件选择寄存器或共享内存的下层策略。

### tle.cumsum

`tle.cumsum(input, axis=0, reverse=False, dtype=None)` 在一次操作中沿 `axis` 维度计算排他性累积和与总和。

- **签名**: `tle.cumsum(input, axis=0, reverse=False, dtype=None)`
- **用途**: 使用单个语义扫描操作同时计算块张量的排他性前缀/后缀和与总和。
- **返回值**: `(exclusive_sum, total_sum)`。
- **典型场景**: top-k、直方图前缀、流压缩以及需要部分排名/偏移的块级分区逻辑。
- `exclusive` 与 `input` 形状相同；`total` 是扫描块的标量和。
- `reverse=True` 表示反向排他性和，适用于降序基数/top-k 选择中的后缀计数。
- `dtype` 可以显式控制累加/结果类型。默认情况下，窄整数提升为 32 位整数，bfloat16 提升为 float32。
- 对于包含性累积和，使用 `exclusive_sum + input`。
- 对无效通道使用显式掩码加载，并将非活动通道设置为 0，确保 `total_sum` 仅计算有效元素。
- 支持的作用域是 `axis=0` 的静态秩-1 块张量；这涵盖了 TLE top-k 内核已使用的直方图和基数选择工作负载。

简单示例：

```{code-block} python
exclusive, total = tle.cumsum(x, axis=0)
inclusive = exclusive + x
```

## 流水线

### 管道与阶段

`tle.pipe` 描述了生产者与一个或多个消费者之间的显式数据流边。它同时记录持有逻辑块的共享内存阶段以及使该块对消费者可见所需的同步，使得 CTA 级别的加载/计算重叠和 warp 专用的生产者/消费者代码能够使用类型化描述符，而不是手动编写多个屏障。

- **签名**: `tle.pipe(*, capacity, scope="cta", name=None, readers=None, one_shot=False, **fields)`
- **用途**: 创建一个类型化管道，用于显式描述 CTA 级别的生产者/消费者数据流、环形缓冲区阶段重用和同步边。
- **参数**:
  - `capacity`: 编译时常量正整数，表示管道阶段的数量；每个负载字段的第一维必须等于 `capacity`。
  - `scope`: 支持的值为 `"cta"`。
  - `name`: 可选的管道名称，用于 IR/诊断；如果提供，必须是字符串。
  - `readers`: 可选的读取器名称列表；省略表示默认的 SPSC 读取器；对于 SPMC，传入 `("left", "right")`。
  - `one_shot`: 是否为单次就绪/完成边；适用于启动数据广播。`one_shot=True` 不支持 `close`。
  - `**fields`: 一个或多个负载缓冲区，必须是由 `tle.gpu.alloc(..., scope=tle.gpu.smem)` 返回的共享内存缓冲张量，秩 >= 2。
- **命名规则**:
  - 管道字段名称和读取器名称必须是有效的 Python 标识符。
  - 名称不能以 `_` 开头。
  - `fields` 和 `readers` 是保留名称。
- `tle.pipe(...)` 返回一个管道描述符。它拥有分阶段的负载字段，并通过 `writer()` 和 `reader(...)` 创建生产者/消费者端点。
- `capacity` 个阶段形成一个环形缓冲区。`iter` 映射到 `stage = iter % capacity`，使用相位位来区分重用轮次。

### 生产者

生产者持有 `pipe.writer()`。它获取一个可写阶段，为逻辑块填充所有必要的字段，然后提交该块，使数据对消费者可见。

- `pipe_value.writer()` → `pipe_writer`: 为当前管道创建单个写入器端点。
- 写入器始终可以看到所有负载字段。
- `writer.acquire(iter)` → `pipe_slot`: 获取一个生产者可写的阶段，返回一个移除了前导 `capacity` 维度的槽位。
- 用户应在 `writer.acquire(iter)` 和 `writer.commit(iter)` 之间生成字段数据。
- `writer.commit(iter)` → `None`: 将阶段标记为就绪，对订阅的消费者可见。同一逻辑块的所有字段写入必须在提交之前完成。
- `writer.close(iter)` → `None`: 发布一个关闭的阶段，供感知关闭的消费者循环退出或切换状态。`one_shot=True` 的管道不支持 `close`。
- 提交是生产者侧的可见性边界。

### 消费者

消费者持有 `pipe.reader(...)`。它等待已发布的块，读取返回的槽位，并在所有读取完成后释放阶段。

- `pipe_value.reader(name=None, fields=None)` → `pipe_reader`: 创建一个消费者端点。
- 对于 SPSC 管道（`readers=None`），必须省略 `name`。
- 对于 SPMC 管道（例如 `readers=("mma", "epilogue")`），必须传入 `name` 并匹配已声明的读取器。
- `fields` 可以是编译时常量的非空元组/列表，包含唯一的负载字段名称；省略表示订阅所有字段。
- 字段子集的消费者仅缩小端点视图和 `wait().slot`；它们不会创建新管道。
- `reader.wait(iter)` → `pipe_wait_result`: 等待阶段就绪或关闭，返回槽位和关闭标志。
- 标准消费路径读取 `wait_result.slot`；仅在处理关闭时检查 `wait_result.is_closed`。
- `reader.release(iter)` → `None`: 消费后释放阶段，允许生产者重用。应在所有 `wait(iter).slot` 读取完成后调用。
- 等待是消费者侧的可见性边界；释放是消费者侧的释放信号。

### 负载字段

- `**fields` 定义了每个阶段携带的数据。每个字段通过名称在 `pipe_slot` 上暴露，例如 `slot.q` 或 `slot.scale`。
- `pipe_slot` 还暴露 `fields: dict[str, tle.gpu.buffered_tensor]`。
- `pipe_wait_result` 包含 `slot: pipe_slot` 和 `is_closed: tl.tensor`。
- 一个管道可以携带一个或多个字段。拆分管道时，按逻辑生命周期和读取器协议拆分，而不是按底层传输拆分。
- 同一槽位中的不同字段可以通过不同的机制生成，例如 TMA 复制、cp.async 风格的复制或 `tle.gpu.local_ptr` + `tl.store`。用户仍然在为该逻辑块生成所有字段后调用一次 `writer.commit(iter)`。
- 每个字段的传输方式由编译器从生产者侧 IR 推断；它不是用户填写的管道属性，也不应编码到管道名称、字段名称或额外的用户属性中。
- 当读取器仅消费字段的子集时，使用 `pipe.reader(name, fields=(...))` 缩小读取器视图；这不会创建新的令牌。
- 保持管道字段来源可见。不透明的共享内存指针逃逸、未跟踪的共享存储或无法证明安全的重叠写入将直接报错，不会静默回退。
- NVIDIA 下层将 CTA 作用域的 SMEM 管道映射到 NVWS/mbarrier 同步。多字段负载需要在管道字段根粒度上证明负载窗口、字段所有权、参与者数量和源顺序安全性。

### 生命周期

- SPSC 管道表示一个生产者向一个默认消费者发布数据。
- SPMC 管道表示一个生产者向多个命名消费者发布相同的逻辑块，例如 `("mma", "epilogue")`。
- `iter` 是逻辑块 ID。在同一块内，生产者和所有参与的消费者应使用相同的 `iter`。
- 标准循环生命周期为 `writer.acquire(iter)` → 生成字段 → `writer.commit(iter)` → `reader.wait(iter)` → 消费字段 → `reader.release(iter)`。
- `one_shot=True` 表示单次就绪/完成边，通常与 `capacity=1` 一起使用；在此模式下不要依赖环形重用或 `close`。

### 简单示例

自动软件流水线仍然可以通过 `tl.range(..., num_stages=...)` 触发。显式管道适用于需要在程序中可见生产者/消费者拆分的场景。

```{code-block} python
stage_buf = tle.gpu.alloc([2, BLOCK], dtype=tl.float32, scope=tle.gpu.smem)
pipe = tle.pipe(capacity=2, scope="cta", name="x_pipe", x=stage_buf)
writer = pipe.writer()
reader = pipe.reader()
offs = tl.arange(0, BLOCK)

slot = writer.acquire(k)
tl.store(tle.gpu.local_ptr(slot.x), tl.load(x_ptr + k * BLOCK + offs))
writer.commit(k)

ready = reader.wait(k)
x = tl.load(tle.gpu.local_ptr(ready.slot.x))
reader.release(k)
```

## 分布式

Triton 分布式 API 由四个核心部分组成：设备网格定义、分片规范描述、同步和远程访问（点对点通信）。

### 设备网格

#### tle.device_mesh

`tle.device_mesh` 定义物理设备的拓扑结构。它是所有分布式操作的基础上下文。

```{code-block} python
class device_mesh:
    def __init__(self, topology: dict):
        """
        初始化 DeviceMesh。

        Args:
            topology (dict): 描述硬件层次结构的字典。
                             键是层级名称；值可以是整数（用于 1D）
                             或元组列表（用于多维层级）。
        """
        self._physical_ids = ...  # 内部存储：扁平化的物理 ID 列表 (0..N-1)
        self._shape = ...         # 当前逻辑视图的形状，例如 (2, 2, 4, 2, 2, 4)
        self._dim_names = ...     # 当前维度的名称
        # 初始化和解析逻辑...

    @property
    def shape(self):
        """返回当前网格的逻辑形状。"""
        return self._shape

    @property
    def ndim(self):
        """返回维度数量。"""
        return len(self._shape)

    def flatten(self):
        """
        将网格展平为 1D。通常用于基于环的通信模式。
        """
        return self.reshape(prod(self._shape))

    def __getitem__(self, key):
        """
        支持切片操作并返回子网格。
        支持标准切片（slice 对象）和整数索引。
        """
        # 计算切片后的新形状和选定的物理 ID
        # ...
        return sub_mesh

    def __repr__(self):
        return f"DeviceMesh(shape={self._shape}, names={self._dim_names})"


# 定义复杂的硬件层次结构
topology = {
    # 节点间层级（2x2 = 4 个节点）
    "node": [("node_x", 2), ("node_y", 2)],
    # 节点内 GPU（4 个设备）
    "device": 4,
    # GPU 内集群（2x2）
    "block_cluster": [("cluster_x", 2), ("cluster_y", 2)],
    # 每个集群内的块（4 个块）
    "block": 4
}

# mesh.shape -> (2, 2, 4, 2, 2, 4)
# 总大小 = 256
mesh = tle.device_mesh(topology=topology)
```

### 分片规范

`tle.sharding` 用于声明张量在设备网格上的当前分布状态。splits 列表描述了张量的每个维度如何在网格上分区，而 partials 列表指示张量是否处于部分和状态。任何未显式提及的网格轴被视为广播（复制）。

- tle.S(axis): 分割 — 表示张量维度沿指定的网格轴分区。
- tle.B: 广播/复制 — 表示张量维度沿任何未显式引用的网格轴完全复制（即不分割）。
- tle.P(axis): 部分 — 表示张量仅持有部分值（例如部分和），必须沿指定的网格轴归约以获得完整结果。

```{code-block} python
def sharding(tensor, splits, partials):
    """
    注解：仅用于标注张量的布局状态。
    它不生成任何运行时代码，但指导编译器进行后续优化或正确性检查。
    """
    return tensor


# 定义一个分片规范，其中：
# - 轴 0 沿 "cluster" 维度分割（具体为 ["cluster_x", "cluster_y"]），
# - 轴 1 沿 "device" 维度分割，
# - 张量沿 "block" 维度处于部分状态（需要归约来解析）。
x_shard = tle.sharding(
    mesh,
    split=[["cluster_x", "cluster_y"], "device"],
    partial=["block"]
)

# 使用上述分片规范创建分片张量
x = tle.make_sharded_tensor(x_ptr, sharding=x_shard, shape=[4, 4])
```

### 同步

在复杂的分布式算子中——例如 Ring-AllReduce 或具有独立行/列通信的流水线执行——我们通常只需要同步同一"行"或"列"内的线程块，而不是整个集群。全局同步会引入不必要的等待开销。
此 API 支持子网格同步，这意味着在大型物理集群中，我们可以定义多个逻辑"通信组"，并在每个组内独立执行同步。

```{code-block} python
def distributed_barrier(mesh):
    """
    如果传入子网格，则仅同步该子网格内的设备。
    子网格外的设备应将此指令视为空操作
    （或者编译器应确保其控制流永远不会到达此点）。
    """
    pass
```

#### tle.distributed_barrier

`tle.distributed_barrier` 仅同步与给定网格或子网格对应的设备集合。

从相邻分片读取（环式交换）。

```{code-block} python
node_rank = tle.shard_id(mesh, "node")
device_rank = tle.shard_id(mesh, "device")
next_device = (device_rank + 1) % mesh.shape[1]
remote_x = tle.remote(x, shard_id=(node_rank, next_device), scope=mesh)
tle.distributed_barrier(mesh)
neighbor_vals = tl.load(remote_x)
```

### 远程访问

`tle.remote` 用于获取位于另一设备上的张量的句柄。这对应于点对点通信或直接内存访问（例如 RDMA/NVLink Load）。它使内核能够显式地从特定分片访问数据。

```{code-block} python
def remote(tensor, shard_id, scope):
    """
    获取驻留在特定设备分片上的远程张量的句柄。

    :param tensor: 逻辑分布式张量（已用 tle.sharding 标注）。
    :param shard_id: tuple。目标设备在设备网格中的坐标。
                     例如，如果 mesh=(2,4) 且 shard_id=(0, 3)，则指节点 #0 上的 GPU #3。
    :return: RemoteTensor。支持 load、store 等操作。
    """
```

`tle.remote`: 显式读取或写入远程分片。

```{code-block} python
node_rank = tle.shard_id(mesh, "node")
device_rank = tle.shard_id(mesh, "device")
next_device = (device_rank + 1) % mesh.shape[1]
remote_x = tle.remote(x, shard_id=(node_rank, next_device), scope=mesh)
tle.distributed_barrier(mesh)
neighbor_vals = tl.load(remote_x)
```

## 与 local_ptr 交互的原语

以下 API 与 `tle.gpu.local_ptr` 一起使用。更多信息请参见 [使用 TLE-Struct](use-tle-struct.md)。

- `tl.load`（用于 local_ptr）
- `tl.store`（用于 local_ptr）
- `tl.atomic_add`/`and`/`cas`/`max`/`min`/`or`/`xchg`/`xor`（用于 local_ptr）
