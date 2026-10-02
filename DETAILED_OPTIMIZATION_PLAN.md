# MiniCPM5-2B 双平台推理吞吐优化路线

本文把 FlagOS 2026 S2 MiniCPM5-2B 优化收敛为七个连续阶段：先冻结规则、环境和可信基线，再用分阶段 Profile 建立瓶颈模型，随后清理确定性开销，分别优化 BI-V150 实际命中的 Attention 路径与 C500 MACA 路径，最后处理共享算子、通用调度、图执行、可选量化和组合回归。每一步都要从真实端到端证据出发，并同时守住正确性、精度、TTFT、4K/16K 泛化和规则合规。本文是总路线；赛题规则、调用链和官方命令见 [S01 赛前静态准备](docs/S01-赛前静态准备.md)，实验记录要求见[项目 skill](skills/flagos-minicpm-competition/SKILL.md)。

本机只用于源码阅读、版本比对、文档和静态检查；不安装比赛工程依赖、不编译、不导入项目模块、不加载模型，也不运行测试、Profile 或 benchmark。租赁目标卡只承担诊断和预验证，正式精度、吞吐、TTFT 与阶段出口只在组委会认可的 BI-V150 与 C500 环境执行。

S1 专家反馈只作为历史方法经验，不是 S2 规则或许可来源；S2 的合规边界只来自当前赛题页和组委会书面答复。OOT 包装、dispatch/model runner 集成、切换现有 vendor/native/Inductor 实现或单纯打开已有能力，都不单独构成创新；性能策略必须包含可审查的实质代码改动，并在报告中映射到实际激活的运行路径。

## 1. 目标、模型与不可越过的边界

[源码事实] [MiniCPM5-2B 配置](https://modelscope.cn/models/OpenBMB/MiniCPM5-2B/resolve/master/config.json)中的 `architectures=LlamaForCausalLM`、`model_type=llama`，是 BF16 稠密 Llama 模型。它有 42 层、hidden size 2048、intermediate size 6144、16 个 Query Head、2 个 KV Head、head dim 128 和 131072 最大上下文，GQA 比例为 8:1。每层主要经过 QKV GEMM、RoPE、KV Cache 写入、Paged Attention、O-Proj、RMSNorm、gate/up GEMM、SiLU×gate 和 down projection。

KV Cache 每 token 的理论占用为：

```text
42 layers × 2(K/V) × 2 KV heads × 128 head_dim × BF16(2 bytes)
= 43,008 bytes ≈ 42 KiB/token

43,008 × 17,408 tokens/request × 64 requests
= 47,915,728,896 bytes ≈ 44.6 GiB
```

[分析假设] 第二式按 16K 输入加 1K 输出，即 17,408 token/request，并用 `1024^3` 换算 GiB。44.6 GiB 只含 KV Cache，不含约 5GB BF16 权重、激活、Attention workspace、图缓冲、运行时和 allocator reserve；因此 C500 64GB 的分页布局、临时张量和调度节奏很关键。BI-V150 的实际显存、token budget 与 scheduler 配置必须从官方环境读取，不能根据型号猜测。

[官方规则] 当前页面给出的四行基线与 1% 参考门槛如下。固定 benchmark 每个场景运行 4 次、丢弃第一次并统计后三次；精确评分和 TTFT 判定仍以组委会最新书面口径为准。

| 平台 | 场景 | 官方 total tok/s | 1% 最小目标 | 官方 Mean TTFT | 保守 TTFT 上限 |
|---|---:|---:|---:|---:|---:|
| BI-V150 | 4K/1K | 2028.01 | 2048.29 | 11573.53 ms | 11689.27 ms |
| BI-V150 | 16K/1K | 915.15 | 924.30 | 599262.03 ms | 605254.65 ms |
| C500 | 4K/1K | 5089.645 | 5140.54 | 3199.435 ms | 3231.43 ms |
| C500 | 16K/1K | 7029.675 | 7099.97 | 27197.135 ms | 27469.11 ms |

[官方规则] [EvalScope MATH-500 数据统计](https://evalscope.readthedocs.io/en/latest/benchmarks/math_500.html#data-statistics)显示 Level 3 共 105 题。基线约 0.962 对应 `101/105=0.9619`，而 `accuracy≥0.95` 至少需要 `100/105=0.9524`，相对基线大致只容许再错一道题；任何改变数值路径的补丁都必须跑完整 Level 3。

[源码事实] BI-V150 的本地固定 dispatch 配置会优先调用 FlagGems Attention selector，但 selector 在未显式启用 `VLLM_FL_USE_FLAGGEMS_ATTN` 时返回 `TRITON_ATTN` 枚举路径；官方镜像的环境变量、注册覆盖和实际 backend 尚待日志确认。C500 的本地配置选择 MetaX/MACA vendor Attention。两张卡共享证据格式和通用调度设计，但必须先按实际 backend 分别归因；不能把一张卡的结论外推到另一张卡，也不能为启用候选而擅自切换后端。

[官方规则] 不得调优正式 benchmark 脚本、test cases、serve 参数和模型行为，不得通过修改超参数开启量化、投机采样或已有前缀缓存功能，不得在没有实质算子优化时仅切换现成算子或删除框架主要算子选择逻辑，也不得直接合并 FlagOS 推理框架最新分支取得性能。不得按 `4096`、`16384` 或 case ID 写死逻辑；技术报告中的策略必须真实存在于最终代码。

[分析假设] 本项目采用更保守的内部门槛：scheduler、graph、固定 metadata buffer 和自研量化在取得组委会书面确认后才进入成绩分支；局部收益必须超过实测噪声并转化为端到端收益。Profiler、缩短请求和 microbenchmark 只用于诊断，正式成绩只来自官方命令。

本地静态基线固定为：

| 仓库 | 指定版本 | 固定 HEAD | 用途 |
|---|---|---|---|
| `vllm-plugin-FL` | `flagos-2026-s2` | `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0` | 比赛主仓库 |
| `FlagGems` | `v5.3.5` | `a7620cc191a0b42e040194622c5758b22a7a25dc` | 算子库 |
| `vllm` | `v0.24.0` | `ee0da84ab9e04ac7610e28580af62c365e898389` | 只读上游参考 |

## 2. 七阶段总表

| 阶段 | 状态 | 核心交付 | 阶段出口 |
|---:|---|---|---|
| 1 | 租赁 C500 已预验证；官方双平台待验收 | 规则答复、环境指纹、正确性与官方基线 | 后端和源码可追溯；Level 3≥0.95；波动可识别约 1% 变化 |
| 2 | 已定位 C500 4K Decode 排序热点；完整双平台归因待做 | 四类时间窗 Profile、热点与 Amdahl 模型 | 至少 85% 时间完成归因；冻结每个平台前三候选 |
| 3 | C500 M1 已预验证；阶段整体待验收 | C500 同步、FlagGems 临时分配与 wrapper 开销清理 | 同步/分配按设计消失；输出一致；无 TTFT 回退 |
| 4 | 待执行 | BI-V150 GQA Decode、Prefill Attention 与 KV write | 16K 提升、4K 不退化、Attention/KV 正确性矩阵通过 |
| 5 | MetaX 排序 kernel 已获租赁 C500 配对、回归与 4K 重复支持，已推送 fork；跨平台待验 | C500 MACA 路径、共享 RoPE/Norm/SiLU 与 graph 分析 | C500 两场景稳定收益；共享路径跨平台回归通过 |
| 6 | 待执行 | Shape-aware 调度、图覆盖与可选自研 W8A16 | 无 case 硬编码；TTFT 合规；量化若启用则完整精度通过 |
| 7 | 待执行 | 双平台组合、消融、干净复现与技术报告 | 四个正式场景、精度、复现和规则合规全部通过 |

七个阶段仍必须依次通过出口；租赁平台上提前做单一热点候选的预验证，不代表跳过前序出口或正式双平台验收。量化只是阶段 6 的可选项；没有书面许可、成熟自研 kernel 和完整精度证据时，不进入最终提交。

## 3. 各阶段实施与出口

### 阶段 1：规则澄清、环境冻结与可信基线

**问题分析**

- 官网指定 branch/tag，但远端引用可能移动；容器实际 import 的包也可能不等于 `/workspace` checkout。
- `C0` 是条件化正确性疑点：只有官方日志确认 BI-V150 实际命中 `AttentionFLBackend/AttentionFLImpl` 时，独立 KV 写入与上游默认声明的差异才构成阻断项；若实际 backend 不同，应记录“不适用”并对真实路径重新归因。
- Accuracy 使用 `temperature=1.0`，而 105 题只留下约一道题余量；seed、重复、TTFT 和双卡 70% 计分方式均需书面确认。

**本阶段目标**

让两个官方环境的代码身份、真实 backend、模型行为、精度与性能波动均可追溯，建立后续所有同机器单变量配对实验的可信起点。

**实施步骤**

1. 使用 [S01 的八项规则问题](docs/S01-赛前静态准备.md#4-待确认问题与提交要求)向组委会确认评分、TTFT、Accuracy、C0、调度/graph、自研量化与提交边界，并归档原文。

2. 在 BI-V150 和 C500 分别记录硬件、显存、驱动、容器 ID、Python、Torch、Triton/FlagTree、vLLM、plugin 与 FlagGems 版本，并以实际运行证明“模型类 → 注册选择 → dispatch → 已编译实现 → 真实设备 kernel”的完整链路。

3. 记录三个仓库的实际 import 路径、HEAD、dirty 状态和关键文件 hash；同时保存 `USE_C_EXTENSION`、FlagGems `has_c_extension/use_c_extension` 以及 server log 中的 Attention backend、KV block、token budget、chunked prefill 与 graph mode。未确认 Python/Triton 分支实际命中前，不按本地源码路径解释 Profile。

4. 在 BI-V150 先用原始服务命令和环境白名单确认完整运行链路与实际 Attention backend。只有命中 `AttentionFLBackend` 时才检查 `forward_includes_kv_cache_update` 并调查 `C0`，不直接 cherry-pick 后续修复；未命中时将 C0 标为不适用。随后运行 3～5 个确定性短 prompt 和至少一个长 prompt，核对 token、EOS、重复和乱码。

5. 执行完整 MATH-500 Level 3，保存逐题输出、采样参数、seed 信息、失败请求和最终分数。

6. 严格使用 [S01 的官方命令](docs/S01-赛前静态准备.md#3-官方复现命令)建立两个平台、两个场景的基线。每个 wrapper 内保持四轮、丢弃第一轮；整套 wrapper 独立重复 3～5 组以估计自然波动。

7. 按[项目 skill 的验证与记录要求](skills/flagos-minicpm-competition/SKILL.md#implementation-and-verification-discipline)汇总 raw CSV、summary CSV、server log、硬件状态、逐题结果、源码 hash 与规则答复，生成不可变 `baseline-manifest`。

**验收条件**

- 两个平台实际加载的源码、import 路径与 backend 均可追溯。
- BI-V150 实际 Attention/KV Cache 路径已确认；若命中 `AttentionFLBackend`，C0 已确认正确或已有组委会认可的独立 correctness patch，未命中则有可复核的“不适用”证据。
- 短、长 prompt 无乱码、异常重复或错误生成长度，所有正式请求成功。
- 完整 Level 3≥0.95，逐题输出与采样身份齐全。
- 至少三组独立基线具有均值、标准差和 CV，足以识别约 1% 的变化。

**退出与回退**

- 实际 backend 未确认、适用时 C0 未闭环、输出异常或精度不过线时停止性能优化，先恢复可追溯性与正确性。
- CV 接近或超过 1% 时先处理热状态、后台负载和冷启动，不进入阶段 2。

### 阶段 2：分阶段 Profile 与瓶颈归因

**问题分析**

- `total tokens/s` 只说明最终速度，不能区分 Prefill、Decode、收尾、CPU 或设备端瓶颈。
- 4K/16K、纯 Prefill、混合 Prefill/Decode、稳态 Decode 和尾批的热点可能不同。
- 只看 device kernel 会漏掉 GPU→CPU 同步、metadata 和调度空洞；只做 microbenchmark 又会漏掉 allocator 与 graph break。

**本阶段目标**

将四个正式场景的大部分端到端时间归入可解释类别，并用收益上限与风险冻结阶段 3～5 的候选顺序。

**实施步骤**

1. 为诊断启动独立服务并启用对应平台 profiler；正式评分服务继续使用原始命令，不添加 profiler 参数。Profile 前再次锁定模型类、注册、dispatch、已编译实现和设备 kernel，采样期间不得切换 backend。

2. 在每个平台、每个场景分别标记初始纯 Prefill、混合 Prefill/Decode、稳态 Decode 和请求收尾四类时间窗。

3. 使用官方 shape 预热至稳态后，在同一平台、服务参数、代码版本、backend 和热状态下同步采集 CPU timeline、device timeline、内存水位、graph capture/replay 与 graph break，不只截取最快片段。

4. 把时间归入 GEMM、Attention Prefill/Decode、RoPE、KV write/read、RMSNorm、SiLU、scheduler/metadata、allocation/copy 与同步空洞。

5. BI-V150 额外记录 Attention shape、GQA swap、tile、warps、stages、寄存器、occupancy 与带宽。

6. C500 额外记录 MACA Prefill/Decode kernel、metadata 构建时间，以及满足非 cascade、DCP=1、`num_prefills>0` 时 `.tolist()` 的同步次数。

7. 使用官方 shape 的暖态时间占比 `p` 与 Amdahl 公式 `1 / ((1-p)+p/s)` 估算收益上限；局部加速仍不足以越过实测噪声的候选先暂缓。

**验收条件**

- 至少 85% 的端到端时间能归入明确类别。
- 四个正式场景均形成分开的 Prefill/Decode 热点表。
- 每个平台列出前三候选、时间占比、理论上限、风险和修改文件。
- BI-V150 16K 的首要瓶颈有 trace 证据。
- C500 `.tolist()` 是否真实影响端到端性能已有对应分支的 trace 证据。

**退出与回退**

- 归因不足 85% 时补采 CPU、设备或内存证据，不凭直觉进入改码。
- 多候选收益接近时按“预期收益×置信度÷实现风险”排序，保留未选项而不并行混改。

### 阶段 3：低风险确定性开销清理

**问题分析**

- C500 的 `M1` 只涉及非 cascade、DCP=1 且 `num_prefills>0` 的路径；这里每层 `.tolist()` 可能造成 device→host 同步并重建 tensor。
- 若 BI 实际命中目标 FlagGems paged Attention，`I3` 路径会创建 dummy cumulative-length tensor，但 kernel 是否读取该参数必须先确认；未命中时该候选不进入实验。
- RoPE wrapper 中同 device、同 dtype 的 `.to()` 不等于发生 copy；`R1` 必须以 CPU timeline 或分配证据为前提。
- BF16 descale view、flatten、view 或 chunk 是否形成可见开销同样需要 Profile，不能仅凭源码形态判断。

**本阶段目标**

先移除数学语义不变、证据明确的同步与临时分配，为后续 kernel 优化提供更干净的 trace 和稳定地址。

**实施步骤**

1. 每个候选使用“未改原版 → 单一候选 → 恢复原版复测”做单变量验证，但不为每个候选建立 commit；两次原版复测漂移明显时不采信候选。未提交的候选必须绑定仓库、完整 base SHA、完整归档 scoped diff 的 SHA-256、全部改动或新增文件的 SHA-256、实际远端 import 路径和不可变实验 ID。相关修改积累到一个完成必要验证并得出保留、继续验证或回滚决定的小阶段交付后再提交，用户明确要求时除外。对外学习结果只展示 A0（官方公布基线）和 B1（当前改进）。

2. 实施 `M1`：在 C500 metadata builder 中一次计算 `prefill_cu_seq_lens` 并供 42 层复用。优先复用已有 cumulative tensor，否则使用 device cumsum；只改目标分支。

3. 实施 `I3`：先确认实际进入 `ops/attention.py` 的纯 Python 分支；该分支当前无条件分配 dummy，而 `flash_api.py:mha_varlan_fwd` 会断言 cumulative tensor 非空和 shape，不能直接传 `None`。确认 `seqused_k` 路径不读取数值后，再复用稳定兼容 tensor 或研究已有 `_opt` 路径，避免逐层创建。

4. 检查 BF16 kernel 是否读取 q/k/v scale；只有确认未读取且 trace 显示价值后，才移除无用 descale expanded view。

5. 仅当 CPU Profile 证明 wrapper 开销可见时实施 `R1`。缓存 cos/sin view 或减少 view/chunk 时，仍保留 device、dtype、partial rotary 与非连续布局边界。

6. 为量化 KV、非连续 tensor、partial rotary、device 变化和未知 shape 保留原实现 fallback，并分别触发验证。

7. `M1` 第一版只移动同步；固定容量 buffer、slice 复用和 graph 地址稳定留给 `M2`，不混入同一实验。

**验收条件**

- C500 目标分支中每层 `.tolist()` device synchronize 按设计消失。
- `I3`、descale 或 wrapper 的目标 allocation/copy 计数按设计下降。
- metadata 与 wrapper 改动通过 token-for-token、边界 shape 和逐题检查。
- 两个平台对应 4K/16K 无失败请求、TTFT 回退或新 graph break。
- 完整 Level 3≥0.95，单项端到端收益超过噪声或是后续必要前置。

**退出与回退**

- 单项只改善源码外观、未改变目标计数器且不支撑后续工作时删除。
- `M1` 语义、地址或混合 batch 出错时回到原 per-layer 构造，再缩小适用条件。

### 阶段 4：BI-V150 Attention 与 KV Cache 主线

**问题分析**

- 本阶段先以日志和 trace 冻结 BI-V150 实际 Attention backend。以下 I1～I5 只在目标 FlagGems 实现确实命中时成立；若未命中，保留阶段目标但按实际 backend 重新建立候选，不通过设置开关切换后端。
- 目标 FlagGems 的单 token GQA 分支把 `[B,16,128]` Q 经过 reshape/transpose/reshape；是否发生实体 materialization 必须由 BI trace 确认，不能仅据源码下结论。
- `I2` 当前会为 swapped 分支创建临时 output，并在调用方提供 output 时执行尾部 `copy_`；42 层和 1024 Decode token 会放大固定开销。
- Iluvatar 当前启发式主要改变 `BLOCK_M=16/32/64/128`，难以同时适配 GQA=8、D=128、4K/16K KV 与 chunked Prefill。
- 16K 是内存与 Attention 主战场，但 4K 同样计分，不能用 4K 回退换单点收益。

**本阶段目标**

在不改变官方 backend 选择的前提下优化 BI-V150 实际 Attention 与 KV 路径；若命中目标 FlagGems 实现，再让 GQA Decode 直接使用正确布局，并分别优化 Decode、Prefill 与 KV write，同时保持通用 shape 条件和完整 fallback。

`I1/I2` 的建议 element stride 合并如下；它描述逻辑解释方式，不代表补丁已经实现：

| Tensor | 逻辑视图 | batch stride | KV-head stride | group/row stride |
|---|---|---:|---:|---:|
| Q | `[B,2,8,128]` | `16×128` | `8×128` | `128` |
| output | `[B,2,8,128]` | `16×128` | `8×128` | `128` |

**实施步骤**

1. 先用 trace 判断 Q reshape 链是否 materialize，并分别计量 Q reorder、临时 output、尾部 copy 和 allocator；`I1` 与 `I2` 独立消融。

2. 实施 `I1`：让 kernel 按表中 stride 读取原始 Q，不先复制成 `[B×8,2,128]`。Fast path 仅覆盖单 token、Q heads 可整除 KV heads、最后维连续的 BF16。

3. 实施 `I2`：让 kernel 按表中 stride 写入 vLLM 预分配 output，消除临时 tensor 与尾部 `copy_`。必须逐 head 验证 Q0～Q15 顺序。

4. 无 ALiBi、滑窗和 dropout 时才进入直接布局；调用方未提供 output、stride 不匹配或 shape 未知时回退原实现。

5. 实施 `I4` Decode heuristic：由 trace 缩减到 4～8 组候选，再比较 `BLOCK_M=8/16`、`BLOCK_N=32/64/128`、warps=4/8、stages=1/2/3。

6. 实施 `I5` Prefill heuristic：比较 `BLOCK_M=32/64/128` 与相应 BN/warps/stages，并同时覆盖 4K、16K 和 chunked Prefill。

7. 实施 `K1` 的 BI-V150 优先 fast path，为 `2 KV heads×128=256` 元素优化 mask、向量宽度和 warps。C500 走 vendor 路径，是否需要独立实现另行 Profile，不能直接共用结论。

8. 只有 Profile 证明 RoPE(K) 到 KV write 的中间往返占比足够高，才研究融合；该融合不作为首轮交付。

**验收条件**

- Attention 差分覆盖 B=1/8/64/256、KV=1/16/4K/16K/17K、混合长度和 block 边界。
- `I1/I2` 验证全部 16 个 Q head 顺序，trace 证明目标 reorder、临时 output、copy 或 allocation 消失。
- KV write 覆盖连续、乱序、负 slot 和跨 block，且未写区域保持不变。
- BI-V150 4K/16K 均按官方四轮流程验证；16K 稳定提升，4K 无明显回退，TTFT 合规。
- 8K/12K shadow shape 无灾难性回退，完整 Level 3≥0.95。

**退出与回退**

- 任一 head 顺序、layout、slot 或输出错误立即关闭 fast path 并回到原实现。
- heuristic 只对正式 case 有利或端到端落入噪声时删除；不得保留 4096/16384 特判。

### 阶段 5：C500 MACA 路径与共享 Decode 算子

**问题分析**

- C500 本地固定配置走 MACA vendor Attention，不能照搬仅在 BI-V150 命中目标 FlagGems backend 时才成立的 kernel 结论；M1 后要重新判断 MACA、metadata 与共享小算子的占比。
- C500 官方 16K 吞吐高于 4K，批处理与长上下文特征明显不同，不能把 BI 结论外推。
- MetaX FP32 排序直方图已完成自研重写、双场景与精度预验证；改后 4K 稳态 Decode 的 8 次迭代中，新直方图累计 3.288 ms，而 `sweep` 累计 117.280 ms。此限定窗口只用于确定下一调查顺序，不是端到端时间占比；16K 的改后分相 Profile 尚未完成。
- 共享 RoPE 当前可能在小 Decode batch 下 program 数不足，但这是待 Profile 的假设。
- 当前 C500 日志与 trace 中 Norm/SiLU 由 native 或编译融合路径执行，不能假定修改 FlagGems 的同名算子会命中；只有真实调用链与热点都确认后才值得重写。
- 固定 vLLM `llama.py` 已用 `MergedColumnParallelLinear` 实现 gate/up 投影，并用 `QKVParallelLinear` 实现 QKV 投影；这两项是现有源码事实，不能写成新实现的创新。

**本阶段目标**

在不切换现成后端的前提下优化 C500 的真实热点。先把已保留的 MetaX 排序改动在官方平台复核，再调查其后续 `sweep` 与 MACA 路径；共享 RoPE、RMSNorm 与 SiLU 只有实际命中且值得优化时才实施。

**实施步骤**

1. 在 `M1` 与排序直方图改动后补齐 C500 的纯 Prefill、混合、稳态 Decode 与收尾 Profile，分别查看 `sweep`、MACA、metadata、RoPE、Norm 和 SiLU。当前仅完成租赁 C500 的 4K 稳态 Decode 限定窗口；下一次上卡须补 16K 和其余阶段，不能从单窗口外推。

2. 若改后 Profile 再次确认排序 `sweep` 是主要可优化成本，先分析 `FlagGems/src/flag_gems/ops/sort.py:sweep` 中逐桶循环、局部扫描、lookback 等待和 scatter，构造仅针对 MetaX FP32/4-bit radix 的单变量候选。原排序路径必须保留回退；无严格稳定索引、特殊浮点位模式、top-p 与同随机状态 token 的差分结果，不进入完整模型测试。

3. 若分配或地址变化仍显著，实施 `M2`：使用固定容量 metadata buffer 与有效 slice，保证 graph 地址稳定并清除陈旧数据。

4. 若 MACA Attention 主导，只优化其 Prefill/Decode 分支准备、layout、KV write 和 launch 配置；不切换到另一现成 backend。

5. 为 contiguous、full-rotary、BF16、小 Decode batch 实施 `R2` 二维 grid；当前 plugin 从 `vllm_fl/ops/rotary_embedding.py:RotaryEmbeddingFL.forward_oot` 传入 `inplace=True`，因此入口是 `FlagGems/src/flag_gems/fused/rotary_embedding.py:apply_rotary_pos_emb_inplace_kernel`。建议候选为：

```text
(num_tokens, ceil((num_q_heads + num_kv_heads) / HEADS_PER_PROGRAM))
HEADS_PER_PROGRAM ∈ {1, 2, 4}
```

该 grid 的第一维分开 token，第二维把 Q/KV heads 切成小组，以增加小 batch 并行度。两个平台可以选择不同 `HEADS_PER_PROGRAM`，但必须共用通用 shape 条件并保留原 kernel fallback；Prefill 和非连续布局不默认进入该路径。

6. 实施 `N1` 时先确认目标运行确实沿 `RMSNormFL.forward_oot → FlagGems dispatch → modules/normalization.py:gems_rms_forward → fused/fused_add_rms_norm.py` 的 residual 路径，而不是 native/Inductor 或其他实现；再比较 hidden 2048 下 4/8 warps、连续 fast path 与 reduction 效率，保持 FP32 累加、epsilon 和原地 residual 语义。

7. 实施 `S1` 时先确认目标运行实际命中 FlagGems 路径，而不是 native/Inductor 实现；再按真实布局处理：父 tensor 末维为 12288，split 后两个 6144 元素 view 的行 stride 仍为 12288。继续用 FP32 sigmoid，不采用近似 exp，也不能把两个 view 误判为独立 contiguous tensor。

8. 同步检查 Decode graph 覆盖和 Attention eager boundary，记录动态 shape、metadata 与 wrapper 导致的 graph break，但把正式图策略留给阶段 6。

9. `sweep/M2/R2/N1/S1` 各自进行“未改原版 → 单一候选 → 恢复原版复测”，先做算子差分，再做两场景端到端；共享小算子在另一平台同步回归。不从 S1 历史反馈复制 `torch.compile` 或 greedy sampler 方案，因为 S2 的官方服务参数和采样行为必须保持不变。

**验收条件**

- C500 Prefill、混合、Decode 和收尾热点均有修改前后对比。
- 已保留的 MetaX 排序直方图在两场景、完整精度和 C500 回归通过；下一候选 `sweep` 只有在自己的差分、完整 Level 3、双场景配对与 TTFT 通过后才能保留。
- `M2` 不新增每 step 分配，地址稳定且 metadata 无陈旧内容。
- vendor Attention 改动属于实质开发，不是后端切换。
- `R2/N1/S1` 分别具有算子误差、局部性能和端到端消融证据。
- [项目验收目标] 最终保留的组合在四个场景分别达到 BI-V150 4K `2048.29` tok/s、BI-V150 16K `924.30` tok/s、C500 4K `5140.54` tok/s 和 C500 16K `7099.97` tok/s；同时 Mean TTFT 不超过对应保守上限、Level 3≥0.95，所有请求成功且重复统计超过实测噪声。这是项目目标，不冒充组委会最终评分规则。

**退出与回退**

- `M2` 破坏地址、slice 或 graph 时回到动态构造，先修正所有权模型。
- `sweep` 任一稳定索引、特殊值、top-p/token 或长上下文性能失败时恢复现有 radix sweep，不影响已保留的直方图改动。
- `R2/N1/S1` 只有 microbenchmark 收益或造成另一平台回退时关闭该平台 fast path或删除。

### 阶段 6：Shape-aware 调度、图执行与可选量化

**问题分析**

- 16K prompt 要经历多轮 chunked Prefill，Prefill/Decode 混排同时影响利用率、KV 水位和 TTFT。
- 请求收尾时 batch 变小，设备可能低占用；graph break 又会放大小模型的 CPU/launch 开销。
- 调度容易变成 benchmark 特调，量化则可能消耗仅约一道题的精度余量。

**本阶段目标**

在书面规则允许下建立通用 shape-aware 调度和可解释 graph replay；仅在证据充分时评估自研 W8A16。

**实施步骤**

1. 取得组委会对内部 scheduler、graph 和自研量化边界的书面确认，不改变正式 serve 参数。

2. 实施 `G1`：以 phase、当前 token、活动序列、最大 KV 长度、KV 水位、token budget 和硬件容量建立调度模型。

3. 分别分析 chunked Prefill、混合 batch 和 request 收尾的利用率/TTFT，保存每轮 batch 组成、设备空洞和 KV 峰值。

4. 使用 2K、8K、12K 与混合长度作为 shadow cases；代码不得判断 4096、16384 或 case ID。

5. 实施 `G2`：逐个定位 Decode graph break，提高 capture/replay 覆盖；动态或不支持 shape 保留 eager fallback。

6. 只有阶段 1～5 稳定、规则允许且 Profile 证明 GEMM/权重带宽主导时，启动 `Q1` 自研 weight-only INT8/W8A16。

7. `Q1` 先验证单层权重格式、反量化与 GEMM，再进入整模；每次候选立即运行完整 Level 3。W4A16 和 KV 量化默认不做。

**验收条件**

- 四个正式场景与 shadow cases 使用同一通用策略，无 case 硬编码。
- 吞吐提升不以 TTFT 越线为代价，两个平台无 OOM、死锁、饥饿或失败请求。
- 每轮 token budget、Prefill/Decode 组成、设备空洞与 KV 峰值可复核。
- graph replay 比例提高，capture、break 与 fallback 均可解释。
- 若启用 `Q1`，完整 Level 3≥0.95，且端到端收益覆盖反量化开销。

**退出与回退**

- 调度导致 TTFT、饥饿、OOM 或 shadow shape 灾难性回退时恢复原策略。
- `Q1` 任一精度、token 行为、shape 支持或净性能失败即删除；量化不是必需交付。

### 阶段 7：组合回归、干净复现与技术报告

**问题分析**

- 单项有效不代表组合有效，shared buffer、graph、tile 与调度可能相互影响。
- 报告策略、最终 diff 和实验身份不一致会形成合规与复现风险。
- 最终结果必须同时覆盖两张卡、两个场景与完整精度，不能只展示最好单点。

**本阶段目标**

只组合已验收补丁，在干净环境完成双平台复现、消融和报告到代码的一一映射。

**实施步骤**

1. 从第一个候选启动时就维护消融顺序与“候选 → 实际源码/提交 → 最终报告章节”映射；最终再从固定比赛 commit/tag 建立干净分支，按 correctness、低风险清理、平台 Attention、共享算子、调度/graph 的顺序应用补丁。

2. 每加入一项都运行短 prompt、对应 operator test 与快速同机器配对对照；发现交互时回到最后一个已知正确组合。

3. 组合完成后在两平台冷启动，运行完整 Level 3 与官方 4K/16K，整套流程独立复现至少两轮。

4. 输出 baseline、单项、平台组合和最终组合的完整消融，不以单次峰值代替稳定统计。

5. 从第二个干净 checkout 按复现说明重放全部补丁，核对 backend、commit、命令、精度和性能。

6. 技术报告按“问题证据→代码文件/commit→正确性→性能→规则依据”逐项映射，规则敏感补丁附书面答复。未激活、没有实质代码、只有 microbenchmark 收益或无法映射最终 diff 的策略不得写成性能成果。

7. 删除无收益代码、诊断开关、case 硬编码、未使用分支和报告未解释的逻辑，整理源码 diff、manifest、原始数据与复现说明，并完成提交包中的 `report.pdf` 与 `readme.md`。

**验收条件**

- 两个平台、两个场景和所有规定重复均无失败请求，生成长度与 token 行为正常。
- Level 3≥0.95 且逐题结果留存，Mean TTFT 符合组委会确认口径。
- 最终吞吐超过自然波动，4K/16K 无未解释的严重回退。
- 至少两轮独立复现和第二个干净 checkout 均能得到相同 backend、精度与相近性能。
- 报告每项策略映射实际 diff，代码每项优化具有实验与规则证据。

**退出与回退**

- 组合退化时按逆序移除补丁并重做消融，不用其他场景收益掩盖问题。
- 第二 checkout 无法复现或报告无法映射时停止提交，修正文档、依赖或补丁序列。

## 4. 关键优化项与验收矩阵

| ID | 平台 | 优化项 | 主要问题 | 保留门槛 |
|---|---|---|---|---|
| C0 | BI-V150；仅命中 `AttentionFLBackend` 时 | KV Cache update 正确性声明 | 适用时 cache 可能未写导致错误输出 | 实际 backend 证据；适用时确认或认可修复，不适用时闭环记录 |
| M1 | C500 | cumulative lengths 移入 metadata builder | 仅非 cascade、DCP=1、`num_prefills>0` 路径每层 `.tolist()` 同步 | 同步消失、逐题一致、TTFT 不退化 |
| I1 | BI-V150；仅命中目标 FlagGems Attention 时 | GQA Decode 直接读取 Q | reshape/transpose 是否 materialize 待 trace | Head 顺序正确；trace 证明目标重排消失 |
| I2 | BI-V150；仅命中目标 FlagGems Attention 时 | GQA Decode 直接写 output | 临时 output 与尾部 `copy_` | Allocation/copy 消失、输出一致 |
| I3 | BI-V150；仅命中目标 FlagGems Attention 时 | 删除 paged Attention dummy tensor | 每层无效 allocation | Allocation 消失、两场景无回退 |
| I4 | BI-V150；仅命中目标 FlagGems Attention 时 | Decode tile/heuristic | GQA=8、D=128 未特化 | 16K 稳定提升、4K 不明显回退 |
| I5 | BI-V150；仅命中目标 FlagGems Attention 时 | Prefill tile/heuristic | 粗粒度 `avg_rows_per_cta` | 4K/16K 与 shadow shapes 泛化 |
| K1 | BI-V150 目标 KV write 路径优先；C500 独立确认 | KV write n=256 fast path | mask、向量宽度和 launch 开销 | Slot 矩阵逐元素一致、端到端有效 |
| M2 | C500 | 固定 metadata buffer | 每 step 分配与 graph 地址变化 | 地址稳定、无新增分配 |
| R1 | 两平台 | RoPE wrapper 清理 | 同 device `.to()` 不等于 copy；仅处理 trace 证明的 view/chunk/分配 | CPU 热点下降、输出一致 |
| R2 | 两平台 | 原地 RoPE Decode 二维 grid | `inplace=True` 路径的小 batch program 数可能不足 | 两平台分别选型、原地语义正确、Prefill 不回退 |
| N1 | 两平台 | fused add + RMSNorm warps/连续 fast path | 真实 residual 路径的 Decode CTA/归约效率 | FP32 累加与原地 residual 不变、端到端收益可见 |
| S1 | 两平台 | SiLU×mul strided-view kernel | 12288 父 tensor 切成 6144 view 后行 stride 仍为 12288 | FP32 sigmoid、stride 正确、完整精度通过 |
| G1 | 两平台 | Shape-aware 调度 | Chunked Prefill、混排与尾批空洞 | TTFT 合规、无 case 硬编码 |
| G2 | 两平台 | Decode graph 覆盖 | Graph break 与 launch 开销 | Replay 比例提高、fallback 正确 |
| Q1 | 分平台 | 自研 W8A16（可选） | 权重带宽与显存 | 规则确认、Level 3≥0.95、净性能为正 |

每个候选内部使用“未改原版 → 单一候选 → 恢复原版复测”：两次原版复测漂移明显时不采信候选；每次保存四轮 raw 数据、后三轮均值/标准差/CV、total/output throughput、Mean/median/P99 TTFT、TPOT、ITL、成功请求、峰值内存、日志、trace、commit/hash 和 Accuracy。公开学习结果只分 A0（官方公布基线）与 B1（当前改进），官方未公布字段填“—”。

| 维度 | 必测组合 | 判定 |
|---|---|---|
| 环境/版本 | 两平台、实际 import、容器与关键 hash | 可追溯、无隐式版本漂移 |
| Attention | Prefill/Decode，B=1/8/64/256，KV=1/16/4K/16K/17K | 数值、head 顺序、layout 与 fallback 正确 |
| RoPE | 长位置、Q16/KV2、in-place、Neox layout | 无 NaN/Inf，误差与基线一致 |
| KV write | 连续/乱序/负 slot、跨 block、未写区域 | 写入逐元素一致，未写区域不变 |
| RMSNorm | 有/无 residual、BF16、N=2048 | FP32 累加与原地 residual 语义不变 |
| SiLU×mul | token=1/64/2048，末维 12288 | FP32 sigmoid，误差和 layout 合规 |
| 模型行为 | 短/4K+/16K+/混合 prompt | Token、EOS、请求数和生成长度无异常 |
| 准确率 | 完整 MATH-500 Level 3 | ≥0.95，并保存逐题输出 |
| 性能 | 两平台×4K/16K，官方四轮流程 | [项目验收目标] BI 4K≥2048.29、BI 16K≥924.30、C500 4K≥5140.54、C500 16K≥7099.97 tok/s；Mean TTFT 不超过对应保守上限，Level 3≥0.95，无失败请求且稳定超过实测噪声 |
| 泛化 | 2K/8K/12K/混合长度 | 无 benchmark 特判或灾难性回退 |
| 稳定性 | 冷启动、重复运行、graph/eager fallback | 无 OOM、失败请求、死锁或偶发错误 |

以下退出规则适用于全部候选：

- 任何 Accuracy 或 token 行为异常都会使对应性能数据作废。
- microbenchmark 有收益但端到端无收益时删除补丁；4K 有利而 16K 回退时只能依据通用 phase/shape 建立策略。
- `M1/M2` 破坏地址或 metadata 语义时回退；`R2` 导致 Prefill 回退时只在 occupancy 支持的小 Decode batch 启用。

## 5. 规则问题、风险与参考边界

[待平台验证] 下列问题必须取得组委会书面答复，详细问题见 [S01](docs/S01-赛前静态准备.md#4-待确认问题与提交要求)：

- 两张卡的性能分怎样组成 70%，是否各有权重或上限。
- TTFT 的 1% 是逐平台逐场景、单轮、后三轮均值还是聚合判定。
- `temperature=1.0` 的 seed、重复次数与随机波动怎样处理。
- 原始 BI-V150 服务命令实际解析到哪个 Attention backend，是否预设 `VLLM_FL_USE_FLAGGEMS_ATTN` 或注册覆盖；仅在命中 `AttentionFLBackend` 时，官方镜像是否包含 C0 等价修复及独立 correctness patch 如何验收。
- scheduler、graph、固定 metadata buffer 与自研 W8A16 的允许边界。
- 平台专用 kernel 如何合并提交，最终镜像、驱动和编译器是否冻结。
- API `race_start=2026-09-11` 与正文开发起点 2026-09-21 哪个生效，并确认 2026-11-20 23:59（UTC+8）为提交截止。

常见风险与防范方式分开记录，避免把官方禁令和项目内部策略混为一谈：

| 风险 | 防范方式 |
|---|---|
| 报告策略未进入最终代码，或代码存在报告未解释的逻辑 | 报告、commit、实验 ID 与 diff 一一映射 |
| 只切换现成后端、合并新分支或修改 benchmark/serve 参数 | 固定官方命令与 HEAD；每项必须有实质开发 diff |
| 错误输出、少生成 token 或失败请求制造虚假高吞吐 | 性能前检查 token、EOS、长度、成功请求与完整 Accuracy |
| 只取单次高点，或 4K 收益掩盖 16K 回退 | 官方四轮丢首；报告后三轮统计并并列呈现两场景 |
| 按 4096/16384 写死或把 BI 方案硬套 C500 | 使用 phase/shape/KV 水位；验证 2K/8K/12K 与平台独立路径 |
| 把单变量验证等同于逐候选 commit，或在同一次候选运行中混入多个改动 | 每轮只验证一个候选，并做“未改原版 → 单一候选 → 恢复原版复测”；用完整 diff/hash 保持未提交实验可追溯，在阶段性决策后批量提交 |
| 未确认规则就进入调度、graph 或量化 | 保存组委会书面答复；未确认时采用保守回退 |

| 官方资料/源码 | 采用内容 | 不越过的边界 |
|---|---|---|
| [比赛页面](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn) | 指定仓库、命令、指标、硬件和规则 | 不自行改变评分口径、命令或输入 |
| [FlagOS dispatch 使用指南](https://docs.flagos.io/projects/vllm-plugin-FL/zh-cn/latest/dispatch_user_guide/dispatch-user-guide.html) | 注册、dispatch 和 OOT 集成机制 | 文档配置只说明候选链路；仍须在目标运行证明已编译实现与实际设备 kernel |
| [MiniCPM5-2B config](https://modelscope.cn/models/OpenBMB/MiniCPM5-2B/resolve/master/config.json) | 模型 shape、dtype、GQA 与上下文 | 不修改模型结构或行为 |
| [EvalScope MATH-500](https://evalscope.readthedocs.io/en/latest/benchmarks/math_500.html) | Level 3 样本数与评测语义 | 不用子集替代完整正式精度 |
| [S01 赛前静态准备](docs/S01-赛前静态准备.md) | 官方命令、调用链与规则问题 | 不把静态分析或非官方平台结果写成官方平台结论 |
| `vllm-plugin-FL@13eb9be` | Dispatch、平台 Attention、metadata 与 graph 接口 | 不直接合并后续性能分支 |
| `FlagGems@a7620cc` | Attention、RoPE、KV write、RMSNorm、SiLU | 不把后端切换冒充原创优化 |
| `vLLM@ee0da84` | Scheduler、AttentionBackend 与模型调用参考 | 只读参考，不作为提交依赖 |

当前执行优先级为：

1. 先冻结 BI-V150 实际 backend；命中 `AttentionFLBackend` 时完成 `C0`，未命中时以“不适用”证据闭环，再建立可信双平台基线并用 Profile 冻结真实热点。

2. C500 `M1` 已以提交 `47882ba` 在租赁卡完成预验证：16K 重复收益可见，4K 尚无涨点，正式平台出口仍待完成；当前代码和结论见 [S02](docs/S02-C500基线运行与内存问题复现.md)。仅当 BI-V150 确实命中目标 FlagGems Attention 时验证 `I3/I1/I2`，其中 `I1` 是否有 materialization 以 trace 为准，否则按实际 backend 重新排序候选。

3. 仅当 BI-V150 确实命中目标 FlagGems Attention，且直接布局与固定开销证据成立时，才推进 `I4/I5`；`K1` 则以实际 KV write 路径证据为前提，不直接外推 C500 vendor 路径。

4. 随后评估 `M2/R1/R2/N1/S1`，最后才研究 `G1/G2` 和可选 `Q1`。
