# MiniCPM5-2B 双平台推理吞吐优化路线

## 1. 当前重点

本计划按七个阶段推进 FlagOS 2026 S2 MiniCPM5-2B 优化。当前只完成了租赁 C500 上的 `M1`、排序全局直方图/E2 和 int32 中间索引预验证；Attention、KV 与新融合主线尚未实施，BI-V150 实际 backend 也未确认，因此七阶段远未完成，正式双平台验收仍待组委会环境。

模型为 BF16，42 层，hidden 2048、intermediate 6144，16Q/2KV、head dim 128、context 131072。KV 约 42 KiB/token，64 并发、每请求 16K+1K 时约 44.6 GiB，不含权重、激活、workspace、图缓冲和 allocator reserve。

当前 B1 固定为 `M1` 与排序 `hist/E2/I32` 的组合，暂停新的排序迭代。其租赁 C500 有效均值为 4K `7795.68 ± 16.17 tok/s`、16K `8551.02 ± 3.40 tok/s`；实现与复现见 [S02](docs/S02-C500基线运行与内存问题复现.md) 和 [S03](docs/S03-C500排序索引优化.md)。

改后 16K 稳态窗口覆盖 8 次 Decode：MACA Attention 主核占设备 kernel 时间 `63.26%`，独立 KV write 占 `0.38%`。paged-KV 读取已包含在主核内，不能重复相加；这些比例只用于当前设备窗口的 Amdahl 排序，不能外推为端到端收益。

下一步顺序固定为：

1. 优先优化 C500 Attention 内部 paged-KV 读取、GQA 跨 Q head 复用与 split/combine 分工。

2. 仅在可消访存量足够时实施 KV layout/write 或 RoPE(K)+KV-write 融合。

3. 再处理实际激活 kernel 的计算/访存瓶颈、尚未融合的数据流或 graph 空洞，不把已有融合冒充新成果。

租赁平台可以继续做独立预验证，但七阶段出口仍须依次通过；BI-V150 暂不可访问不阻断当前 C500 工作，也不代表阶段 4、阶段 5 或正式双平台出口完成。

## 2. 统一约束与验收

本机只做源码、文档和 Git 静态工作，目标卡才运行导入、编译、测试、服务、Profile 与 benchmark。指定版本、官方命令、规则问题和调用链统一引用 [S01](docs/S01-赛前静态准备.md)；实验与文档纪律遵循[项目 skill](skills/flagos-minicpm-competition/SKILL.md)。只修改 `vllm-plugin-FL` 和 `FlagGems`，`vllm` 保持只读参考。

必须保持官方 benchmark、test cases、serve 参数、模型、采样、输出长度和评分输入不变。禁止通过超参数开启量化或投机采样，禁止调 benchmark，禁止只切换现成 operator 或删除主要选择逻辑取得性能，禁止直接合并最新推理框架分支；同时不得启用现成前缀缓存或按 `4096/16384`、case ID 写死分支。

每项成果必须包含可审查的实质计算或访存代码，报告精确映射实际激活路径，并为未知 shape、layout、dtype 和平台保留原实现 fallback。不得为命中候选而切换既有 backend；租赁结果只称预验证，规则解释和正式结论以比赛页面、组委会书面答复及官方平台为准。

官方 A0 与保守门槛如下；A0 始终只指组委会公布值：

| 平台 | 场景 | A0 total tok/s | A0×1.01 | A0 Mean TTFT | 保守 TTFT 上限 |
|---|---:|---:|---:|---:|---:|
| BI-V150 | 4K/1K | 2028.01 | 2048.29 | 11573.53 ms | 11689.27 ms |
| BI-V150 | 16K/1K | 915.15 | 924.30 | 599262.03 ms | 605254.65 ms |
| C500 | 4K/1K | 5089.645 | 5140.54 | 3199.435 ms | 3231.43 ms |
| C500 | 16K/1K | 7029.675 | 7099.97 | 27197.135 ms | 27469.11 ms |

所有候选采用同一验收链：

1. 先做算子差分、边界 shape、layout 和 fallback 测试；任何输出、token、EOS、长度或请求错误都会使性能结果作废。

2. 数值路径改变后运行完整 MATH-500 Level 3，至少 `100/105` 正确，并保存逐题输出、采样身份和失败请求。

3. 在目标平台依次运行固定 B1、单一候选、恢复 B1。4K/16K 均用官方 wrapper 跑四轮、丢弃第一轮，报告后三轮 mean、SD、CV，并保留所有轮次，包括慢轮。

4. 要求所有请求成功、无 OOM，Mean TTFT 不越表中上限，两个场景的吞吐收益均超过实测噪声；最后补改后 Profile，并在另一平台执行适用路径与 fallback 回归。

5. 只有 microbenchmark 收益、路径未激活、无法映射最终 diff 或端到端落入噪声的补丁不保留。相关修改积累成完成正确性、性能和保留决定的阶段成果后再批量提交，除非用户另行授权。

每次远端实验结束后立即同步原始结果和源码身份，并原位更新相关学习文档；沿用现有 SNN，不为一次小实验新建文档或小提交。

## 3. 七阶段实施

### 阶段 1：规则、环境与可信基线

**问题与目标。** 容器实际 import、设备 backend 和仓库 checkout 可能不同；先让双平台代码身份、模型行为、精度和波动可追溯。

1. 按 S01 确认规则与官方命令，记录设备、驱动、镜像、Python、Torch、编译器、plugin、FlagGems、实际 import 路径和源码状态。

2. 从模型类、注册、dispatch、编译实现到设备 kernel 核实实际链路。KV update 的 C0 只在 BI-V150 真正命中 `AttentionFLBackend` 时检查，否则记录不适用并分析真实 backend。

3. 运行确定性短、长和混合 prompt smoke，核对 token、EOS、生成长度、失败请求和重复异常，再执行完整 Level 3。

4. 两个平台的 4K/16K 基线至少独立重复三组，保存四轮原始数据与后三轮统计，用于识别约 1% 的变化。

**验收。** 实际 import/backend 可追溯，适用时 C0 正确，smoke 和 Level 3 通过，双平台基线波动足以判断约 1% 收益。

### 阶段 2：分阶段 Profile 与瓶颈归因

**问题与目标。** 总吞吐无法区分 Prefill、Decode、CPU 空洞或内存压力；需用联合测量解释至少 85% 端到端时间并冻结真实热点顺序。

1. 在两平台、两场景分别标记纯 Prefill、mixed、稳态 Decode 和 tail，请求阶段必须由 token 与 annotation 共同确认。

2. 同步采集 CPU timeline、device kernel、内存水位、allocation/copy、同步、graph capture/replay 和 break，不只截取最快片段。

3. 将时间归入 GEMM、Attention、RoPE、KV read/write、Norm/激活、metadata/scheduler 与空洞，并核实每个热点对应的实际源码路径。

4. 只用匹配时间窗的占比计算 Amdahl 上限；收益不足以越过噪声的候选暂缓，每个平台冻结前三项实际热点。

**验收。** 四个正式场景都有分相热点与收益上限，至少 85% 端到端时间可解释，候选顺序由 trace 而非源码外观决定。

### 阶段 3：低风险同步与分配清理

**问题与目标。** `M1` 处理 C500 非 cascade、DCP=1 且有 Prefill 的路径；其余同步、分配和 wrapper 只有被 Profile 证明可见才处理。

1. 在 metadata builder 一次构造并共享 `prefill_cu_seq_lens`，供 42 层使用，不在每层 `.tolist()` 后重建 tensor。

2. 对 dummy tensor、descale、view/chunk、`.to()` 和固定 buffer 分别核实读取关系、allocation/copy 与地址需求，不直接删除接口必需对象。

3. 仅对已测得的开销做单变量修改，并为 mixed batch、非连续布局、量化 KV、未知 shape/device 保留边界 fallback。

4. 用目标计数器、token 差分、Level 3 和两场景配对验证；源码更短但计数器与端到端不变的补丁删除。

**验收。** 目标同步或分配按设计消失，metadata 语义和 graph 地址正确，4K/16K 无稳定回退；当前租赁 C500 结果仍待官方双平台出口。

### 阶段 4：BI-V150 Attention 与 KV

**问题与目标。** BI-V150 实际 backend 尚未确认；只有目标实现真实激活后，才能优化其 GQA Decode、Prefill、layout、direct output 和 KV write。

1. 先用未修改官方服务锁定 BI backend、调用链与设备 kernel；不得设置开关切换 backend 来命中候选。

2. 若目标 FlagGems 路径激活，分别计量 Q layout/reorder、临时 output、尾部 copy、Decode/Prefill tile 与 allocation，再逐项实施直接读写或通用 heuristic。

3. KV write 只在实际路径和占比成立时优化，覆盖连续、乱序、负 slot、跨 block 和未写区域；所有 layout、head 顺序与未知 shape 保留 fallback。

4. BI 暂不可访问时继续 C500 独立预验证，但阶段 4 和后续正式顺序保持 pending，不外推另一平台结论。

**验收。** 候选路径实际激活，Attention/KV 差分、完整精度与 BI 4K/16K 四轮配对通过，16K 稳定提升且 4K、TTFT 和 shadow shape 不回退。

### 阶段 5：C500 Attention、KV 与融合主线

**问题与目标。** 当前 C500 主热点是 MACA Attention，但安装版 `flash_attn` 的签名、binding 和可提交源码边界未确认；本阶段必须交付至少一项非排序、实际激活的实质代码，不能用已完成的排序成果代替。

1. 在目标容器核实 MACA 的 actual import、函数签名、编译 binding 与源码权限，并分别 Profile 主核、combine、copy、KV write 和 CPU 空洞。实际 Decode 接口未证明支持 `num_splits` 或 direct output 前，不把 metadata 字段当调优旋钮。

2. 首攻 paged-KV 读取、GQA 跨 Q head 的 K/V 复用和 split/combine 工作分配，保持 BF16 输入输出、FP32 softmax 状态、scale 与 causal 语义。

3. 若 MACA 主核在不可编辑边界外，评估 FlagGems 自主受限 BF16 paged-GQA Decode kernel；只能从 plugin 原调用点做 scoped dispatch，并保留原 backend selection 和 fallback，不能仅切换现成实现。

4. KV/RoPE 融合先过成本门，核实 slot mapping、cache layout、ownership、alias 与可消访存量；保持 Q 旋转、K 原地语义和 cache contract，避免重复写 cache。独立 KV write 占比很小，不能单独支撑融合。

5. Norm/SiLU 当前已有 Inductor 融合，QKV 与 gate/up 也已合并；不得重复宣称这些融合，但新的 Profile 若证明已激活 kernel 仍有计算或访存瓶颈，可以做实质优化。graph 变更留待阶段 6 规则门。

**验收。** 至少一项 Attention、KV 或融合的非排序代码在 4K/16K 都独立稳定生效；覆盖 mixed、2K/8K/12K、head mapping、paging 边界、layout、workspace 与 fallback，并通过统一 Level 3、配对 wrapper 和改后 Profile。租赁结果不代替官方双平台验收。

### 阶段 6：通用调度、图执行与可选量化

**问题与目标。** 调度和 graph 容易变成 benchmark 特调，量化又有较高精度与规则风险；只有书面规则允许且前五阶段稳定后才进入。

1. 先取得 scheduler、graph 和自研量化边界的书面确认，不改变正式 serve 参数。

2. 调度只使用 phase、活动序列、当前 token、token budget、最大 KV 长度、KV 水位和 graph eligibility，不读取 case 名或固定长度。

3. 提高 graph capture/replay 覆盖时保留动态 shape 与不支持路径的 eager fallback，并用 2K/8K/12K、mixed 和 tail 检查泛化。

4. 自研 W8A16 只在规则允许且权重带宽为热点时最后评估，先做单层差分再整模精度；不得通过超参数开启量化、投机采样，W4 与 KV 量化不作为早期候选。

**验收。** 四个正式场景和 shadow shapes 使用同一语义策略，无硬编码、饥饿、OOM 或 TTFT 越线；graph fallback 可解释，量化若启用则完整 Level 3 与净端到端收益均通过。

### 阶段 7：组合、复现与提交

**问题与目标。** 单项通过不代表组合有效；最终只组合已验收补丁，并让正式数据、最终 diff 和技术报告一一对应。

1. 按单项验收顺序组合补丁，每加入一项都做消融、短 prompt、算子差分和快速配对，出现交互时回到最后一个已知正确组合。

2. 在 BI-V150 与 C500 分别运行完整 Level 3、4K/16K 官方流程和全精度回归，整套正式流程独立复现至少两次。

3. 从第二个干净 checkout 重放补丁，核对 backend、实际 import、命令、精度、性能和 fallback。

4. 报告逐项映射“问题分析→实际代码→正确性→性能→规则依据”，保留完整消融，不用单次峰值掩盖场景回退。

5. 删除未激活、无收益、仅 microbenchmark、诊断开关、case 硬编码和报告无法解释的路径，再整理提交包。

**验收。** 两平台、两场景均无失败请求并稳定超过噪声，完整精度与 TTFT 合规；两次独立复现和第二 checkout 结果一致，最终报告精确对应提交代码。
