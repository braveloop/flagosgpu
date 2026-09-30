# FlagOS 2026 S2：MiniCPM5-2B 吞吐优化静态分析与执行计划

> 分析日期：2026-09-29  
> 本机约束：仅下载、阅读和比对源码；未安装依赖，未编译，未加载模型，未执行测试或 benchmark。所有性能与准确率实验只在赛事官方算力上进行。

## 1. 本地源码基线

| 目录 | 版本 | HEAD | 用途 |
|---|---|---|---|
| `vllm-plugin-FL` | `flagos-2026-s2` | `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0` | 比赛主仓库 |
| `FlagGems` | `v5.3.5` | `a7620cc191a0b42e040194622c5758b22a7a25dc` | 比赛指定算子库 |
| `vllm` | `v0.24.0` | `ee0da84ab9e04ac7610e28580af62c365e898389` | 只读上游参考，用于还原框架调用与默认参数 |

三个仓库当前均为干净工作区。`vllm` 不应作为提交依赖直接修改；赛事提交应围绕指定的 plugin 与 FlagGems 版本形成最小补丁。

## 2. 题目本质与约束

这不是一个孤立算子的微基准，而是固定模型、固定服务命令、固定请求形状的端到端 serving 优化：

- 模型：MiniCPM5-2B，BF16，42 层，hidden size 2048，intermediate size 6144，16 个 Q head、2 个 KV head、head dim 128，最大长度 131072。
- 场景：`[4096, 1024, 64, 256]` 与 `[16384, 1024, 64, 128]`。
- 每个场景运行 4 次，丢弃第一次，取后三次均值。
- 主指标是 total token throughput；每个场景至少提升 1% 才有意义。
- Mean TTFT 不能相对基线恶化约 1%。
- MATH-500 Level 3 准确率基线为 0.962，门槛为 0.95。105 条样本下容错空间很小，应把准确率作为硬门禁，而不是最后再测。
- 不得修改 benchmark 脚本、服务参数或模型行为；不得靠直接切换已有算子、合并较新框架分支、开启投机解码或参数化量化来获益。

KV-cache 粗略占用为每 token 42 KiB（42 层 × 2 个 KV head × 128 × K/V × BF16）。一个 16K+1K 请求约需 714 MiB，64 个并发请求约需 44.6 GiB，因此长上下文场景明显受 KV-cache 带宽、paged attention 和调度节奏影响。

## 3. 官方基线与最小有效目标

以下“吞吐目标”按基线 × 1.01 计算；“TTFT 上限”按基线 × 1.01 计算，仅用于保守筛选，最终以赛事判定口径为准。

| 平台 | 场景 | 基线 total tok/s | 最小吞吐目标 | 基线 Mean TTFT | 保守 TTFT 上限 |
|---|---:|---:|---:|---:|---:|
| 天数智芯 BI-V150 | 4K/1K | 2028.01 | 2048.29 | 11573.53 ms | 11689.27 ms |
| 天数智芯 BI-V150 | 16K/1K | 915.15 | 924.30 | 599262.03 ms | 605254.65 ms |
| 沐曦 C500 64GB | 4K/1K | 5089.645 | 5140.54 | 3199.435 ms | 3231.43 ms |
| 沐曦 C500 64GB | 16K/1K | 7029.675 | 7099.97 | 27197.135 ms | 27469.11 ms |

网页中两个评分说明的聚合措辞存在一点歧义。优化时必须分别保存四个场景的结果，不能只看一个跨场景平均数。

## 4. 实际热路径

MiniCPM5-2B 每层的主要路径为：

1. fused QKV GEMM；
2. RoPE；
3. KV-cache 写入与 paged attention；
4. output projection GEMM；
5. fused residual + RMSNorm；
6. merged gate/up GEMM；
7. SiLU × gate；
8. down projection GEMM。

上述过程重复 42 次。当前 plugin 主要替换 attention、RMSNorm、SiLU×mul 和 RoPE；大矩阵乘本身不是 FlagGems dispatch 的主要覆盖点。

平台路径并不相同。本地固定 BI-V150 dispatch 配置会优先调用 FlagGems Attention selector，但 selector 在未显式设置 `VLLM_FL_USE_FLAGGEMS_ATTN` 时返回 `TRITON_ATTN` 枚举路径；官方镜像还可能通过环境变量、注册覆盖或补丁改变最终结果。C500 本地配置则选择 MetaX/MACA vendor Attention，其他小算子仍需按实际 dispatch 确认。

因此不存在一个“只调 FlagGems attention 就同时提升两张卡”的方案。先用原始服务命令、环境白名单和日志冻结实际 backend，再按平台分别 Profile；不能为了套用候选而自行切换已有后端。

## 5. 静态分析结论

### 5.1 调度默认值会强烈影响长 prompt

上游 vLLM 0.24 对 OpenAI API server 的默认值取决于设备总显存：小于 70 GiB 时通常为 `max_num_batched_tokens=2048`、`max_num_seqs=256`；达到 70 GiB 且不是 A100 时为 8192/1024。chunked prefill 默认启用，但默认只允许 1 个 partial prefill。

C500 64GB 大概率落在 2048/256 分支。BI-V150 必须在官方环境读取实际显存与最终 resolved config，不能凭型号猜测。16K prompt 在 2048-token budget 下需要多轮调度，这也解释了其 TTFT 对调度与 prefill kernel 极敏感。

比赛不允许改变 serve 参数，因此不能把“调大 batch token 参数”当作提交方案；但可以优化相同调度语义下的 metadata 构建、拷贝和 kernel 实现。

### 5.2 BI-V150 Attention 候选取决于实际 backend

只有官方日志确认 BI-V150 实际命中目标 FlagGems Attention 后，下面的源码分析才适用。该 varlen attention 对 `avg_rows_per_cta` 使用四档粗粒度启发式，选择 `mha_block_16/32/64/128`。MiniCPM 的 GQA 比例为 8，head dim 为 128；KV block/page size 必须从官方环境读取，decode 和 chunked prefill 的最佳配置可能不同。若实际 backend 不是该实现，本节候选不进入实验，阶段 2 应对真实路径重新归因。

首轮 profile 应验证：

- 4K/16K prefill 分别落入哪个配置；
- decode 的 query-group swap 是否稳定生效；
- BLOCK_M/BLOCK_N、warps、stages 的实际选择；
- attention、KV-cache reshape/write、metadata 准备各占多少时间；
- kernel 是否为带宽受限、occupancy 受限或 launch-bound。

只有拿到这些数据后，才进行小范围候选配置搜索，并把最终选择写成针对固定 shape 特征的确定性 heuristic，而不是依赖 benchmark 特判。

### 5.3 C500 使用独立的 MACA attention

C500 的 dispatch 配置只允许 `vendor:metax` attention。该实现会区分 prefill/decode，并在 decode 路径使用 MACA 的 `flash_attn_with_kvcache`。因此 BI-V150 的 FlagGems attention 调优不能迁移到 C500。

C500 的优化优先级应由 profile 决定：若 MACA attention 已高度优化，则先检查 42 层反复调用的 RoPE、RMSNorm、SiLU×mul 与 CPU/metadata 开销；若长上下文 decode attention 占主导，再分析 MACA 路径的分支、layout、cache 写入及 launch 配置。

### 5.4 KV-cache 更新是条件化版本风险

比赛分支的 `AttentionFLBackend` 实现了 `do_kv_cache_update()`，`forward()` 又明确假定 cache 已经预先写入，但该类没有声明：

```python
forward_includes_kv_cache_update = False
```

上游基类默认值是 `True`，这会让 vLLM 认为 `forward()` 自己完成 cache 更新，从而跳过独立更新调用。

比赛分支提交日期为 2026-09-07；仓库在 2026-09-09 的 `3a0da4e` 才加入上述 5 行修复，提交说明明确记录未修复时会出现乱码/重复输出。该提交不在比赛指定分支内。

这个风险只在官方 BI-V150 实际命中 `AttentionFLBackend` 时成立；如果日志确认使用其他 backend，应把 C0 标为“不适用”，而不是设置开关切换后端来制造问题。处理原则：

1. 不把后续提交直接 cherry-pick 到参赛分支；
2. 先在官方镜像核对实际 backend、环境变量、安装文件、包路径和 hash；只有命中目标实现时再确认镜像是否已额外修复；
3. 先做极小生成 sanity check，再跑规定准确率；
4. 若官方环境确实缺失修复，应向赛事方确认指定分支与官方镜像的预期状态；必要时把它作为独立、可解释的 correctness patch，而不是性能创新提交。

这是适用时必须在调优前清掉的 P0 风险；不适用时则以 backend 日志闭环并转向实际路径，否则吞吐数字仍可能建立在错误归因上。

### 5.5 共享小算子是第二梯队候选

- fused residual + RMSNorm 已经是单 kernel，hidden size 2048 走直接路径；仍可检查 BF16 向量化、reduction 结构和输出写回。
- SiLU×mul 当前使用通用 pointwise 生成路径；MiniCPM 的固定连续 shape 可测试专用实现是否降低通用调度开销。
- RoPE wrapper 每次 forward 都包含 cache/device、flatten/view/chunk 等 Python/张量准备操作；在 decode 和 graph 边界处可能是可见开销。
- KV reshape-and-cache 对本模型每 token 只搬运 2×128 的 K 和 V，适合检查更宽向量访问、合并访存与 launch 数，但不得改变 cache layout 或数值语义。

这些算子每层都会调用，微小收益能累计 42 次；但仍需 profile 证明，不应盲目重写。

## 6. 下一步执行计划

### 阶段 A：本机继续做的工作（静态，零运行）

1. 从比赛指定 HEAD 建立独立工作分支，保持原始分支不动。
2. 建立实验台账：补丁 ID、平台、源码 hash、服务日志、四次原始结果、后三次均值、TTFT、准确率、profile 链接。
3. 为 attention、RMSNorm、SiLU、RoPE 和 KV-cache update 建立最小源码映射与补丁边界。
4. 准备官方环境采集脚本，但不在本机执行；脚本只采集版本/config/log，不改系统状态。

### 阶段 B：拿到每一种官方算力后的第一个工作日

按以下顺序执行，并对两种平台分别留档：

1. **环境指纹**：记录硬件型号/显存、容器信息、Python/Torch/vLLM/plugin/FlagGems 版本、源码路径、git HEAD、关键文件 hash。
2. **最终配置**：从启动日志记录实际 attention backend、`max_num_batched_tokens`、`max_num_seqs`、chunked-prefill、KV block size、graph mode。
3. **正确性门禁**：先跑 3～5 个确定性短 prompt，确认无乱码/重复；再执行官方 MATH-500 Level 3 命令并保存逐题结果。
4. **原始基线**：完全使用官网命令运行两个场景；每场景四次，保留 raw CSV、summary CSV 和 server log。不要用仓库里旧的 Qwen 通用脚本替代官网脚本。
5. **波动评估**：计算后三次均值、标准差和变异系数。若 1% 小于环境自然波动，则先控制温度、后台进程和冷启动状态。

### 阶段 C：诊断 profile

使用单独的诊断服务启用 vLLM profiler；这一步允许加入 profiler 配置，但其结果不能作为正式成绩。分别截取：

- 4K 场景的稳定 prefill 段和稳定 decode 段；
- 16K 场景的稳定 prefill 段和稳定 decode 段；
- CPU scheduler/metadata 时间、device kernel 时间、graph replay 情况、同步和空洞。

输出一张按平台与阶段拆分的热点表。只有累计占比足够且理论上能贡献超过 1% 端到端收益的候选，才进入实现。

### 阶段 D：按风险与收益做单变量优化

推荐顺序：

1. BI-V150 实际 Attention backend 的首要候选；仅在命中目标 FlagGems 实现时研究 paged attention 配置；
2. BI-V150 实际命中的 KV-cache reshape/write 路径；
3. C500 MACA attention 或其 metadata/cache 路径（由 profile 决定）；
4. 两平台共享的 RoPE；
5. 两平台共享的 RMSNorm；
6. 两平台共享的 SiLU×mul；
7. graph/metadata 的纯开销优化。

每个补丁只改一个假设，先做算子级正确性对比，再跑小规模服务 smoke test，最后才跑完整四次 benchmark。不得把多个未经证实的改动堆在一起。

### 阶段 E：每个候选补丁的准入门槛

补丁只有同时满足下列条件才保留：

- 两个官方场景均无失败请求；
- MATH-500 Level 3 ≥ 0.95，且输出行为无异常；
- 目标场景 total tok/s 的提升超过自然波动，并尽量达到 ≥1%；
- Mean TTFT 不突破保守上限；
- 无新增 OOM、graph capture 失败或非确定性错误；
- 修改属于真实实现优化，不依赖改 benchmark、改 serve 参数或切换已有后端。

### 阶段 F：提交前复验

1. 从干净的比赛指定版本重新应用最小补丁。
2. 两个平台分别冷启动，按官方命令完成全套准确率与吞吐测试。
3. 保存源码 diff、环境指纹、原始日志、CSV、profile 前后对比和复现说明。
4. 对规则边界敏感的 correctness patch、平台专用分支或自研量化方案，提前向赛事方书面确认。

## 7. 当前决策

当前不建议立即写 kernel。最优下一步是：**申请/进入官方 BI-V150 和 C500 环境，先完成版本指纹、实际 backend 与条件化 KV-cache correctness 核对、官方准确率和原始基线，再用 profile 决定第一个算子补丁。**

若只能先拿到一张卡，可优先 BI-V150：首要目的不是假定其落入 FlagGems，而是尽快冻结真实 backend，并在适用时消除 KV-cache 更新声明的版本疑点；若不适用，则立刻按实际路径重新排候选。
