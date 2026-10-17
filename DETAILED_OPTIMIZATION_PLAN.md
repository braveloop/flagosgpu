# MiniCPM5-2B 双平台推理吞吐优化路线

## 1. 当前重点

- 2026-10-16 BI-V150本版KV128与既有Mixed/runtime/GEMM/SQL组合已完成完整105题，103/105正确，105唯一成功非空、HTTP200/stop各105、请求错误0。原wrapper于21:41:42–22:57:51完成双场景各四轮、整体exit0；去首后三轮4K3819.66±8.35、16K3020.37±2.41 tok/s，CV0.218634%/0.079785%，MeanTTFT4250.15/165812.15ms，全部请求tokens完整、评分服务日志未见OOM或错误。主审独立核原件后已批准保留并更新当前B1；不把同日单轮control比较称为稳定因果增量。
- 本轮基底为 `FlagGems@900531d91192bc5095a9838c9f3b03fc4b5098a0`、`vllm-plugin-FL@c75ae7e02507a2e4cee3b3369ff142318dca2bd0`；生产只增加Prefill完整共同历史128→64→32和page16/32/64的8/4/2页映射，Q2、因果32尾、Decode、selector及fallback不变。局部独立FP32、边界和完整成本门已完成，细节见现有S01；所有修改未提交、不推送，原生vLLM保持只读。
- 当前唯一正式A0/B1表、精度及统计统一维护于 [B1-V150 S03第5节](docs/B1-V150/S03-天数混合注意力与矩阵访存及配置加载优化.md#5-精度与官方正式结果)。本版正式测试无Profile；主审批准的唯一非评分单4K初始8iteration窗于23:00:31–23:07:49完成并清理，256成功0失败、tokens完整，总用时约438秒且无重试/额外窗口；原件与源/元数据/device联合证据已审。该初始混合窗确认Q16及部分CTA共同历史≥128，未观察Q8；可验证kernel sum1431.281ms，GEMM57.85%、Prefill31.04%，不外推16K/纯Decode/全程。10/12旧Profile不移植为本版比例，FULL内部仍保持opaque边界；本版跨平台适用路径和最终双平台出口未完成。
- MetaX既有租赁预验证7795.68/8551.02 tok/s和102/105保持原身份，见 [MetaX S03第3章](docs/MetaX/S03-C500采样前缀判定与连续写回优化.md#3-正确性与性能结果)。
- device prefix证明+连续mask写已完成105（103/105、错误0），但原4+4的16K未确认稳定增益，见 [MetaX S03](docs/MetaX/S03-C500采样前缀判定与连续写回优化.md)。geometry-v2已过局部门与原key复核，组合仍未验收；新实例 `4ceeedda09c6` 部署前原装Torch实算失败，官方macainfo仅CPU2/GPU0；sGPU-M已开启但租户逻辑设备未识别，需核对sGPU资源注入，暂停部署、无新成绩。选卡变量未见不证明注入错误，物理parent访问拒绝可能是正常隔离。恢复后按既定原版同日1+1→冻结组合105→原4+4；accepted不变。
- 历史候选和失败仅保留内部ledger与immutable evidence。后续按真实可消耗时、覆盖、完整成本和风险筛选有界实质优化，最终组合仍须完整105题、原wrapper4+4及适用跨平台门，评分不带采集工具。
- 用户确认本轮BI-V150于北京时间2026-10-16 24:00（10月17日00:00）到期，23:40开始最终同步；已批单窗Profile已完成，现阶段仅文档/归档与最终同步，不擅自续租或关闭实例。MetaX四份预先存在dirty源码保留原状，不在MetaX执行本轮测试，全部原始结果与冻结源码持续同步。

## 2. 统一约束与验收

本机只做源码、文档和 Git 静态工作，目标卡才运行导入、编译、测试、服务、Profile 与 benchmark。指定版本、规则问题、调用链及实验文档纪律统一遵循[项目 skill](skills/flagos-minicpm-competition/SKILL.md)与本节统一约束。只修改 `vllm-plugin-FL` 和 `FlagGems`，`vllm` 保持只读参考。

必须保持官方 benchmark、test cases、serve 参数、模型、采样、输出长度和评分输入不变。禁止通过超参数开启量化或投机采样，禁止调 benchmark，禁止只切换现成 operator 或删除主要选择逻辑取得性能，禁止直接合并最新推理框架分支；同时不得启用现成前缀缓存或按 `4096/16384`、case ID 写死分支。

每项成果必须包含可审查的实质计算或访存代码，报告精确映射实际激活路径，并为未知 shape、layout、dtype 和平台保留原实现 fallback。不得为命中候选而切换既有 backend；租赁结果只称预验证，规则解释和正式结论以比赛页面、组委会书面答复及官方平台为准。

官方A0与门槛如下；A0始终只指组委会公布值。吞吐有效性下限为A0×0.99；A0×1.01是涨点>1%的参考线，不是官方否决阈值，稳定收益还须超过实测噪声：

| 平台 | 场景 | A0 total tok/s | 官方下限A0×0.99 | 涨点参考A0×1.01 | A0 Mean TTFT | TTFT 上限 |
|---|---:|---:|---:|---:|---:|---:|
| BI-V150 | 4K/1K | 2028.01 | 2007.73 | 2048.29 | 11573.53 ms | 11689.27 ms |
| BI-V150 | 16K/1K | 915.15 | 906.00 | 924.30 | 599262.03 ms | 605254.65 ms |
| C500 | 4K/1K | 5089.645 | 5038.75 | 5140.54 | 3199.435 ms | 3231.43 ms |
| C500 | 16K/1K | 7029.675 | 6959.38 | 7099.97 | 27197.135 ms | 27469.11 ms |

所有候选采用同一验收链：

1. 先做算子差分、边界 shape、layout 和 fallback 测试；任何输出、token、EOS、长度或请求错误都会使性能结果作废。

2. 数值路径改变的单项或同主线组合候选通过局部差分后，统一运行完整 MATH-500 Level 3，至少 `100/105` 正确，并保存逐题输出、采样身份和失败请求；不为组合内每个小改动重复完整评测。

3. 在同一目标平台与不变服务参数下，用必要的短配对、恢复 B1 对照或消融排查漂移与交互；对待验收的最终组合统一运行官方 4K/16K wrapper，各四轮、丢弃第一轮，报告后三轮 mean、SD、CV 并保留所有轮次，包括慢轮。不要求组合内每个子项重复三臂全程，最终组合的稳定收益与完整验收门不变。

4. 要求所有请求成功、无 OOM，Mean TTFT 不越表中上限，两个场景的吞吐收益均超过实测噪声；最后补改后 Profile，并在另一平台执行适用路径与 fallback 回归。

5. 只有 microbenchmark 收益、路径未激活、无法映射最终 diff 或端到端落入噪声的补丁不保留。相关修改积累成完成正确性、性能和保留决定的阶段成果后再批量提交，除非用户另行授权。

每次远端实验结束后立即同步原始结果和源码身份，并原位更新相关学习文档；沿用现有 SNN，不为一次小实验新建文档或小提交。每个平台最高编号现存 SNN 末尾维护唯一 A0/B1 四行表、完整精度摘要与统计口径；依既有创建门槛或用户明确许可新建专题后立即整段迁移而非复制，旧页只保留链接，索引与计划同步。迁移不改变数值、接受状态或租赁预验证边界。

## 3. 七阶段实施

### 阶段 1：规则、环境与可信基线

**问题与目标。** 容器实际 import、设备 backend 和仓库 checkout 可能不同；先让双平台代码身份、模型行为、精度和波动可追溯。

1. 按项目 skill 与本计划统一约束确认规则与调用链，记录设备、驱动、镜像、Python、Torch、编译器、plugin、FlagGems、实际 import 路径和源码状态。

2. 从模型类、注册、dispatch、编译实现到设备 kernel 核实实际链路。KV update 的 C0 只在 BI-V150 真正命中 `AttentionFLBackend` 时检查，否则记录不适用并分析真实 backend。

3. 运行确定性短、长和混合 prompt smoke，核对 token、EOS、生成长度、失败请求和重复异常，再执行完整 Level 3。

4. 两个平台的 4K/16K 基线至少独立重复三组，保存四轮原始数据与后三轮统计，用于识别约 1% 的变化。

**验收。** 实际 import/backend 可追溯，适用时 C0 正确，smoke 和 Level 3 通过，双平台基线波动足以判断约 1% 收益。

### 阶段 2：分阶段 Profile 与瓶颈归因

**问题与目标。** 用目标版本mcTracer短窗区分Prefill、Decode、CPU空洞和设备工作，建立真实可消耗时与候选顺序；开发不以先解释85%全程为前置条件。

1. 本轮C500先核实已有mcTracer实际help和GPU EngineCore子进程，复用匹配组件、不升级SDK；在原服务/图模式下用2–3秒、最多5秒窗口标记Prefill/mixed、稳态Decode和tail，以token及可用annotation确认阶段。

2. 不叠Torch profiler或硬件计数器；分开记录runtime/API、device kernel、stream/correlation及graph节点。FULL未展开部分保留opaque，不将API aggregate伪拆成算子，也不从时长推出带宽/spill；采集与stop/export干扰不得混入评分。

3. 只按可归因的设备证据和实际源码路径归类GEMM、Attention、RoPE、KV、Norm/激活、sampling、metadata/scheduler与空洞；选择活跃可改且净收益有依据的热点，不受固定算子/阶段顺序限制。

4. 只用匹配窗口的可消耗时与覆盖估计收益上限，纳入dispatch、workspace、编译/调优等完整成本和风险；正确性或成本门失败即转不同机制/热点，不盲重旧变体，采集能力不足不伪造热点结论。

**验收。** 候选须有真实分相热点、活跃路径和完整净收益依据；四个正式场景的分相解释与85%覆盖保留为最终质量目标，不阻塞已具充分证据的开发。正式成绩不带采集工具，参数和请求/token合同不变。

### 阶段 3：低风险同步与分配清理

**问题与目标。** `M1` 处理 C500 非 cascade、DCP=1 且有 Prefill 的路径；其余同步、分配和 wrapper 只有被 Profile 证明可见才处理。

1. `M1` 已在 metadata builder 一次构造并跨42层共享 `prefill_cu_seq_lens`，消除逐层 `.tolist()` 后重建 tensor；`seq_lens/block_table` 亦已按batch共享，不重复立项。

2. 对 dummy tensor、descale、view/chunk、`.to()` 和固定 buffer 分别核实读取关系、allocation/copy 与地址需求，不直接删除接口必需对象。仅当 `prefill_query_start_loc` 减法及 `prefill_cu` 的 cat/cumsum 分配或launch成本实测可见时，考虑自研一次填充与固定容量双int32元数据缓冲，大小为 `2×(Rmax+1)×4` 字节；明确owner、stream、graph固定地址及batch生存期，不添同步，净收益不越噪声即停止。

3. 仅对已测得的开销做单变量修改，并为 mixed batch、非连续布局、量化 KV、未知 shape/device 保留边界 fallback。

4. 用目标计数器、token 差分、Level 3 和两场景配对验证；源码更短但计数器与端到端不变的补丁删除。

容量依据：本轮C500 KV池约120.57万token，已有日志峰值使用90.7%、抢占0；`64×(16384+1024)=1114112` 约占池容量92.4%，尚无容量限速依据。plugin `model_runner.py::_allocate_kv_cache_tensors/_reshape_kv_cache_tensors` 已预分配并通过views/shared_by复用，worker已核算graph池；不重造池、不改配置、不启用量化/prefix。仅后续出现因KV不足等待/抢占，或可消除临时峰值挤占KV预算时，才立项容量开发，不改核算数字制造容量；区分初始化、峰值与稳态搬运收益。

**验收。** 目标同步或分配按设计消失，metadata 语义和 graph 地址正确，4K/16K 无稳定回退；当前租赁 C500 结果仍待官方双平台出口。

### 阶段 4：BI-V150 Attention 与 KV

**问题与目标。** 当前KV128组合的Q8/Q16完整共同历史128→64→32、原Q2/因果尾、直接序列索引Decode及数量参数编译策略已完成局部数学、本版完整105和BI双场景正式四轮，并获准保留。正式结果统一在天数S03第5节；10/12旧组合的初始/稳态/tail Profile只保留原日期与范围，本版独立单4K初始混合短窗已完成并经原件审查，FULL内部未展开部分仍如实保留。跨平台适用路径与最终双平台出口留待后续阶段。

1. **优先级 1：P1 自研 paged-GQA Prefill/mixed Attention。** FlagGems 的 iluvatar 实现已组合跨 query KV 复用、完整历史/因果边界拆分、页粒度寻址与冗余 mask 削减、Q2/Q8/Q16 选择四项工作；`max_query_len == 2` 优先 Q2，否则只有最大查询至少16且整个batch平均查询至少16才选Q16，其余Q8。保持 BF16、FP32 online softmax、原 scale/causal 语义与输出 buffer contract，保留 physical page 负值/越界保护、Q 读取与输出写入 mask。plugin 仅通过原 `TRITON_ATTN` 枚举的受限实现子类接入，保留原 selector、KV update、mixed FULL graph 与未知 shape/layout/feature 的 fallback，不修改 vLLM，不启用现成 FlagGems Attention 开关。

2. **优先级2：当前KV128/Mixed/runtime/GEMM/SQL组合正式保留。** Q8/Q16的完整共同历史按128→64→32连续处理；128-key在page16/32/64正确映射8/4/2页，其他合法布局沿用原页号广播或逐lane读取，因果32尾/Q2保持原分段；Decode按设备长度选择32或64，固定16段与三个FP32 workspace。Prefill数量参数为runtime且禁止按值特化，数学定义、布局与兼容guard保持不变，128-key分块改变浮点归约顺序并重新完成FP32与整模数值门。本版103/105及原wrapper双场景各四轮门已通过；完整组合包含本版Prefill与既有Decode/Mixed/GEMM/SQL，未提交；本版改后仅完成单4K初始混合短窗，不外推其他窗口。

3. **当前Mixed与后续稳态Decode研究。** 实际高q1 Mixed及自身KV长度和Attention设备热点已经支持原16段Decode/q1与Q16 skip的三文件关联包，builder及全部launch/merge成本已纳入局部门，最终组合已通过天数正式双场景验收。后续同源非评分Profile保持FULL内部opaque边界，不切graph/后端求可见性。仅在真实稳态长度分布及partial/merge设备计时证明工作失衡后，才研究固定CTA队列思路；现阶段不实施该路线，不扩大当前冻结源码。

4. paged-KV 读取已计入 Attention 主核，不能重复相加。实际 mixed 窗口中的独立 KV write 不是首要热点；后续只有实际占比或可消除访存量支持时才研究 KV/RoPE 融合，并保持 slot mapping、负 slot、跨 block、cache alias 与未写区域语义。

5. C500 `M1` 针对的逐层 CPU `.tolist()` 路径在当前 iluvatar Attention 中不存在，不照搬补丁。Norm/SiLU 等已有融合先沿用，只有 BI 实际热点证明还有可消除工作时才继续开发，不重复宣称已有融合；sort/sampler 差分微测未见可分辨回归，本轮不继续排序。双平台沿现有 dispatch、平台 guard 和原实现 fallback 共用同一合并源码，不重造兼容框架或强行统一 backend。

6. 调度、graph 策略与自研量化留待规则许可、适用审批和真实热点支持后最后评估。BI 的 32 GiB 与 KV 压力不是修改正式 `gpu-memory-utilization`、`max-model-len`、KV cache dtype、并发或采样参数的理由；不通过服务配置、量化开关或更换现成 backend 求分。

**验收。** 设备路径实际激活，Attention/KV 差分、必要的完整 105 题精度、BI 4K/16K 官方四轮及跨平台适用路径/fallback 门通过，吞吐提升超过噪声，TTFT、请求正确性与 shadow shape 不回退。当前KV128组合的局部正确性、103/105数值精度和BI正式双场景四轮已完成；本版改后单4K初始混合Profile已完成并清理，10/12旧窗口不充作新证据，双平台门仍待后续阶段。

### 阶段 5：C500 Attention、KV 与融合主线

**问题与目标。** 当前 C500 主热点是 MACA Attention；本轮受限形状/版本的native caller-output合同与full-forward局部门已验证，但1+1整模增益约0.37%/0.04%不足稳定收益，已封存，无新依据不重复立项。本阶段仍须交付非排序、实际激活的实质代码，不能用排序成果代替。

1. 复用已归档接口和本轮已验证的受限output alias/ownership、动态及图执行合同，不泛化其他ABI；后续依据新的主核、combine、copy、KV write或CPU空洞成本证据筛选路线，不重复无整模收益的R6，不凭metadata字段或 `num_splits` 参数宣称收益。

2. 首攻 paged-KV 读取、GQA 跨 Q head 的 K/V 复用和 split/combine 工作分配，保持 BF16 输入输出、FP32 softmax 状态、scale 与 causal 语义。

3. 已失败自研Decode的完整图成本慢约54%，本轮不重启。历史KV读取已包含于vendor Attention主核，不能当额外独立热点重复计算；后续仍保留原backend selection、主dispatch与fallback，不能仅切换现成实现。

4. **有条件的output内存路径。** 在plugin MetaX Attention的 `forward` 与 `fa_utils.py` 边界实现调用者output复用及graph地址、alias/ownership合同，消除临时输出与copy；native OptionalTensor仅为本轮受限形状/版本已核验的入口，其他ABI合同须独立验证。真正的缓冲生命周期与写入编排可作实质框架开发，直接换现成out/append接口不冒充原创kernel，没有实质开发不立项。

   - `64token×16head×128dim×BF16` 每层输出256KiB，相关copy读写最多512KiB/层；42层累计流量不能算作峰值显存释放。vendor append仅核验接口合同，不独立立项；自研KV write/RoPE+slot融合为小备选，不重复现有功能，保持负slot/跨页及当前token只写一次、旋转一次。
   - 旧KV write与相关copy约1.2%仅属旧短kernel窗口，不是端到端比例；先计metadata、dispatch、workspace完整成本，无收益依据不实施。
   - 后续每个候选最多一次≤20分钟差分与eager及实际图执行路径的完整成本验证，不改变C500官方serve graph模式，覆盖负slot、跨页、alias、长短mixed与图重放；净收益不超噪声即止，通过后才纳入组合105题和原4K/16K官方各四轮、后三轮均值验收。规划完成可开展有界实验。

5. Norm/SiLU 当前已有 Inductor 融合，QKV 与 gate/up 也已合并；不得重复宣称这些融合，但新的 Profile 若证明已激活 kernel 仍有计算或访存瓶颈，可以做实质优化。graph 变更留待阶段 6 规则门。

6. **Shape-aware 多内核与有界调优（合规开发路线，待性能验证）。** 在平台内部增加二级选择，保留原主 dispatch、主要选择逻辑与 fallback；原 kernel 仅作回退，不计为新增实质开发。

   - 从实际热点选择2–3个等价但具有实质计算/访存改动的Triton变体，分别建立受益形状范围；原kernel仅fallback，不做配置海扫。
   - 按 M/N/K、dtype、layout、device，以及适用时的 page/query/KV length 等语义条件选型；禁止硬编码 4096/16384 场景、请求编号或基准答案，形状表不能替代泛化 guard。
   - 优先复用本仓现有 LibTuner 能力，按目标实际版本核对接口；官方 v5.3.5 材料仅作流程参考，不合并 master，不把新接口未经核验移植到当前环境。
   - 正确性先于性能：每个变体先做原实现差分及边界、mask、尾块、layout、别名与 graph 安全验证，再按同输入、同配置和配对顺序比较；仅保留完整路径净收益超过噪声的语义区间。
   - 成本包含查表、dispatch、workspace、编译/调优冷启动及其摊销、eager 与 graph；预先限定候选数、单次调优时间和总预算，超限立即回退，不靠基准前隐蔽预调优或忽略 TTFT 获益。
   - 缓存命名空间/键包含设备、编译器及实现版本，并隔离 dtype/layout 和影响正确性的元数据；配置缓存不得改实际输入、编译键或 mask 语义，且必须能在干净环境按相同预算再生。
   - 最终仍须通过完整 105 题、原参数 4K/16K、冷启动与稳定态、跨平台适用路径/fallback 及正式验收；局部胜出不写成整模或官方收益。
   - 据用户于 2026-10-13 确认，组委会已书面允许自行实质优化多个等价 Triton kernel，并按 Tensor Shape 选择内核/配置，前提是模型行为、Benchmark、vLLM 启动参数及主要选择逻辑不变；本项目据此开展，许可不等于已实现或通过验收。
   - “C500 超过 80%”暂无公开代码或报告，不作本项目收益保证。

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
