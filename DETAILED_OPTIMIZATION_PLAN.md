# MiniCPM5-2B 双平台推理吞吐优化路线

用户投入禁令：严禁新增小算子优化，包括 sort/sampling 扩展、RoPE、Norm、SiLU、KV write 及其微融合；严禁继续投入复制消除、输出缓冲复用、小型 metadata/分配清理，以及权重物理布局 packing/prepack/repack/交错实验。不得以“低优先级”“备选”“有条件”或“附带优化”重新立项。已验收代码及必要正确性、兼容性回归可以保留，但不据此重开优化。这是用户的资源投入限制，不是组委会新增禁令，覆盖旧路线和 skill 中冲突的选项；本轮不修改 skill。

## 1. 当前重点

- **10月29日当前结论：v2+extent双景稳定吞吐预验证PASS，完整精度未验。** 原wrapper各双景4轮、固定去首后三：4K5034.83±31.43tok/s/CV.624326%，相对同机retained BF163930.61为+28.092949%；16K4293.03±5.28/CV.122938%，相对3162.79为+35.735575%。候选慢有效4999.42/4286.97均高于对照快有效3987.94/3164.49；16轮请求/token/TTFT完整通过。16:54:53全部自有服务清理no-live；外层长期SSH传输255与远端正常wrapper/监督完成收据分别保留，无重跑/扣时。原v2的83模块/recipe/数值路径、上游vLLM、官方benchmark与serve/缓存/采样参数未改；physical reserve计入原准入，实际batch可改变。源码已冻结92579d25…b754f1d，生产BF16/v2/canonical及所有dirty保留。正式3864.10/3164.70、102/105不换，无新105/提交/推送/SNN；租期18:22:55 freeze/sync、18:37:55 owned-exit、18:42:55结束不延长。
- **10月28日历史租期与13:24授权。** 当日用户授权至18:24，18:04 freeze/sync、18:19退出自有计算；保留dirty，无新105/提交/推送。允许新量化路线、先收益依据和短门，取代10-26会话的禁止扩quant范围；旧失败仍关闭。075000 fastquant首单层4K成本失败，BF16 Q/P+对齐I8KV原型因旧数学假设纠正静态撤回。K-only中心化先冻结42μ，首layer35过、后train4k/layer8最差10.016006%超门关闭（9过1败/余74未跑），无成本/模型；固定Qmean范围统计也未过预注册门。capture只作校准/开发筛查，不继承held-out/模型质量，v2/canonical/正式表当日均未改变。
- 10月26日本轮最新结论：原83模块native W8候选v2的原wrapper四轮严格去首后三，4K为`4063.04±202.30`、16K为`3378.75±5.97 tok/s`，相对同机BF16提高`6.59%/8.68%`。16K后三轮与对照稳定分离，4K的CV为4.98%且慢轮与对照重叠，未达到双景稳定保留条件。独立自然文本test PPL增加0.238%，不替代MATH精度门。后续v3/v4均未改善双景净收益，两个down通道保护信息门未达到实施覆盖门。正式表与保留BF16源码不变；完整原件索引见`evidence/bi150-20261025-152700Z-native-w8-fourround/session-summary.md`。
- **10月26日历史会话收敛目标：只稳定现有83模块W8 v2。** 当时固定recipe、量化范围和数值路径，保留16K增量而只解决4K慢轮，停止新增INT8 KV、Q/P Attention、混合精度缓存及其它量化扩展。075000Z快速Q/P量化包在84组训练capture数值验证通过后依此停止，未跑完整成本，不是吞吐成果。该扩quant禁令现由10-28用户新授权取代，旧事实与失败不改；没有实际修正时仍不追加同源热轮，不用快轮替换慢轮或通过降低快轮速度宣称稳定。
- 4K已有证据是同逻辑KV、同请求长度的四组真实页布局使42层Decode相差4.75–4.91ms，尚未证明整模慢轮全部由此造成。已失败的KV32、四page16直接SME计算、拼接KV64不重试；旧仅排序已选新块的覆盖不足结论保留。不得用prefix-cache/调度/CLI参数游戏碰成绩；10-29另行批准的extent-first是明确改变cached-free选择的实质放置策略，须完整计费reserve与潜在命中损失。候选须通过正确性/多轮复用/完整成本，再作原始4K对照及16K回归，不能用初始pool连续性冒充稳定收益。
- C500 历史 4K 慢轮的一次取栈窗口出现 `LibTuner.bench → default_policy → NT mm_out`。它只关联该窗口，不能证明所有慢轮同因或重复 cache miss；既有 M1828 候选局部门通过，但端到端约 +0.0568%，ROI 未过，不外推为整模收益。
- 2026-10-19 BI-V150 当前 B1 为 Q32 与既有 KV128/Mixed/runtime/GEMM/SQL 组合，完整精度 102/105；原 wrapper 双场景各四轮通过，去首后三轮 4K `3864.10±5.04715`、16K `3164.69667±1.54196` tok/s，CV `0.130616%/0.048724%`，Mean TTFT `4104.1467/156931.9867 ms`。唯一正式 A0/B1 表、请求与精度统计仍在 [B1-V150 S03 第5节](docs/B1-V150/S03-天数混合注意力与矩阵访存及配置加载优化.md#5-精度与官方正式结果)。
- BI 当前基底48b7ac36/FGcf2e930b/plugin d88e5eed，历史租期为用户于10-25 23:03续授7小时，截止10-26 06:03，05:43转同步、05:58退出自有实验；主代理直接实施已审定阶段6，不等待慢执行代理。18:00已触发，部署诊断与真实激活采集已完成；本轮 fresh BF16 控制4024.06/3260.86，SmoothQuant候选3625.28/3005.63 tok/s退化，候选接入已撤。canonical保留BF16＋Decode/slabs/shared-A五源07561/14de9/8401/55af/8a4c及其它dirty。`180402Z-retained-BAB` 保守开发重复性通过：4K B1/A/B2=4026.01/3793.94/4023.13（最差+6.04%），16K3257.50/3167.77/3244.20（最差+2.41%），增量均大于两B差，六场请求/token/TTFT全过，非正式稳定/孤立slabs归因。`202807Z` head-only局部虽通过，fresh控制4026.34/3262.60 tok/s、TTFT5443.24/151493.10ms，候选4K3420.98（−15.035%）/TTFT14628.08超官方11689.27，完整256请求/token后按原门关闭、未启16K，canonical从未启用head。唯一恢复对照 `211719Z-bf16-recovery` 原4K于05:25:32通过：4025.06 tok/s、TTFT5447.96ms、256成功0失败且tokens完整；前后B吞吐差−0.03179%，KV521744→head455520→恢复521840。05:25:35自有进程remaining=[]，不追加轮次；恢复只支持候选关联，不确定约58秒具体根因，不扣时重算成绩。全部原件/冻结源/IR留ledger。保留开发参考为4K4025.06（恢复轮）/16K3262.60（前BF16轮），非同轮新组合；正式仍3864.10/3164.70，旧3939.84/3267.40原件不改。生产保持BF16，量化精度未知，无105/SNN/提交/推送。
- MetaX 历史租赁 B1 保持 `7795.68/8551.02 tok/s` 与 `102/105`，唯一结果表和统计口径仍在 [MetaX S02 第4节](docs/MetaX/S02-C500排序索引优化.md#4-完整精度与历史预验证结果)，只代表该阶段原环境。历史控制为 sort892f7/sampler305355/mm2d4f；当前本地仍有未提交 down-pack 候选（mm c38c24/runner3756488），LM 已撤。down-pack 冻结为未验收候选，停止后续验证、晋级和集成，本轮不删除其代码，不冒称历史 B1 源码或继承旧精度。
- 本文是调整后的路线，不表示七阶段已完成，也不产生新成绩。开发不自动运行新 105，包括数值路径变化；最终 105 必须取得用户重新明确授权，旧租赁截止时间不构成当前执行授权。
- BI down-projection 的双输出面板共享 A 分块已通过局部正确性、通用形状与 fallback、输入/输出归属及改输入图重放。真实 `M/N/K=2048/2048/6144`、42 份独立层数据的完整 eager/graph 设备耗时分别降低 `16.27%/16.31%`。同日原双场景各一轮，当前 B1 `3804.84/3163.94` → 改后 `3836.40/3193.68 tok/s`，增量为 `0.83%/0.94%`，全部请求/token 完整，Mean TTFT `5594.52/156977.49` → `5464.30/155272.99 ms`。只作为开发配对，不宣称稳定正式收益、不继承旧精度。O/QKV 扩展在首 QKV 完整 eager 设备成本门慢约 `4.98%`，已按计划停止，未扩大生产 guard。后续转向有实际活跃几何支持的 Decode 主计算，不继续扫描这一扩展。上轮主代理接手许可不外推为后续模型路由豁免；10-23 本轮已完成 Astra 规划与 Sol 执行交接。
- 用户最新要求优先寻找大幅端到端涨点。BI 短诊断已把实际 FL graph 的 `M64` 四类 GEMM 捕获与 16 次同图 Decode replay 对应；42 份独立层数据的完整 graph 成本分别为 QKV `1.4483`、O `1.2120`、gate/up `3.8837`、down `2.3209 ms`。down 只有 48 个 K 步且可省时间有限，不开展 split-K，不把上述合成成本当整模占比。Decode QK/PV 双转置确使完整 KV64 步的矩阵指令由每 warp 64 降到 16，寄存器 113→77、spill 0，但正确性通过后的首个完整 42 层 graph 成本由 `40.78` 增至 `90.49 ms`，已立即关闭并还原。不能以指令减少代替净收益；只审阅由该结果支持的主计算/归约布局机制，不扫参数。首次观察误用 upstream graph 类，已用实际 FL `GraphWrapper` 修正；生产数值、服务和 benchmark 均未修改，90 秒内的诊断前缀不算性能轮。

- Decode 固定单 warp 的完整 42 层 graph 耗时由 `40.78` 增至 `53.35 ms`，慢 `30.84%`，正确性通过仍撤回。连同成对归一化、双转置共三次失败，关闭该组局部变体，不再扫描 warp、tile 或转置组合。
- 10-23 历史原 Decode 主核计数器诊断记录：LSU 利用率 `86.31%`、shared store bank conflicts `16,962,048`、访存依赖与限流停顿合计 `46.83%`；实际驻留率 `18.51%` 接近 shared 容量限定的理论 `18.75%`，TCU 与 DRAM 读取效率为 `18.94%/36.23%`。当时据此研究 K/V 从 shared 写入线程到 TCU 操作数的映射，后续显式 shared-layout 完整成本已失败关闭（见下一条），不再作为当前优先级；不把冲突次数、单核时长或 replay 缓存命中率换算成整模收益。旧信息门只证明 generic `set_layout/local_load` 路径缺失；10-23 依据官方 [FlagTree #724](https://github.com/flagos-ai/FlagTree/pull/724)、[#880](https://github.com/flagos-ai/FlagTree/pull/880) 及现装源码确认 vendor `alloc + local_ptr` 和所需 passes 存在，因此不能写成全部 TLE 不可用。
- 2026-10-23：Decode 显式 shared-layout（local_ptr store/load）候选已成功编译并通过首个严格正确性门，资源为 shared 20 KiB、120 registers、0 spill；但 42 个独立层的完整 partial＋merge graph 首成本由 40.77097 ms 增至 138.31121 ms（慢 239.24%），成本门失败，已恢复原 Decode，本轮关闭该候选。此结果是局部诊断，不是新的整模成绩；已验收 B1 不变。编译排障细节与原始记录保留于 failure ledger。
- 官方固定 Iluvatar 源码复核已关闭两条救跑：现装八参 TLE builder 无 useTcu，而 TCU composeShared 固定 vec=512/bitwidth、perPhase=maxPhase=1 和专有 useTcu 映射；原物理 KV 为 block-major K/V 交错，即使页号连续也不满足 KS0=BLOCK_SIZE×KS1，不能把整个 KV64 作单 base＋固定 stride 的 SME 读取。依据为 [FlagTree 固定源码](https://github.com/flagos-ai/FlagTree/blob/acd391c7ff8/third_party/iluvatar/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp)及同提交 TLE binding、TritonGPUToLLVM Utility；不改编译器/IR、不拆页拼装。
- SIMT 8 头共享 KV 的固定 Decode 候选已通过首 FP32 .02、分类与所有权门，资源为 179 registers、0 spill、shared 20 KiB；IR 确认每 KV64 tile 仅一份 K/V global load，已移除该分支 TCU staging，但仍有跨 warp 归约和 shared 通信。42 独立层完整 partial＋merge 的 7ABA graph 耗时由 40.79296 增至 156.42195 ms（慢 283.45%），首成本门失败，已恢复原 Decode 并关闭该固定候选；未进入 eager、扩展形状或整模，旧 B1 不变，不扫描参数或救跑。

主 GEMM 的 shared-B 双 M-panel 已关闭：原尺寸 256/256/32、W16/S2 与 K 累加顺序不变，实际 IR/ISA 保留 SME/TCU 并让两组输出行共享一次 B 加载，首形状输出逐位一致、149 registers、0 spill、96 KiB shared。但 42 独立层的真实 mm_out 7ABA eager 设备耗时由 49.56234 增至 53.45934 ms（慢 7.86%），未进入 graph 或整模，已恢复本轮前的 mm，保留原 shared-A 与其他工作区改动。减少逻辑读取不等于减少完整耗时，不调 tile/warp/stage 救跑。后续仅依据新的主热点机制立项；本轮用户已允许主代理接手执行，Astra 规划保持，正式 B1 不变。

## 2. 统一约束与验收

10-23 已完成的历史单包为 KV 分离物理布局与连续页 Decode 直接加载，Prefill 保留原 gather。原严格位门发现极少量 BF16 差异，失败记录保留；随后采用项目原有的独立 FP32 `atol=.02, rtol=.02` 合同，两臂对称验证，分类、输入、归属与 canary 仍为硬门，不继承整模精度。首个 4K 短窗口的 8 个相邻 pure Decode 步中，完整 KV64 访问连续率为 `97.5804%`；保留全部真实页号与长度、仅裁去未访问尾部容量的 42 层局部复核，8 个样本均通过数值与 7ABA 成本门，设备耗时降低 `27.91%～27.99%`。这是实际元数据＋合成 BF16 内容的局部成本，不是整模成绩；完全随机页约 `5.63%` 的回退仍保留。插件已加入受限分配：普通 BF16/NHD/单卡 FullAttention 使用独立 K/V slabs，跨层打包、padding、zeroing、connector/offload 等配置保留旧布局；新增接入与既有 bridge 共 8 项测试通过。16:12–16:52 同机原 CLI 双场景各一轮已完成：当前 shared-A 对照 `3611.98/3100.86 tok/s` → Decode-slabs `3939.84/3267.40 tok/s`，增量 `9.08%/5.37%`，所有请求/token 完整。Mean TTFT `5464.60/159521.63` → `6197.32/151321.94 ms`，两景低于官方上限，但 4K 相对同机对照变慢，不称为时延改善。仅单轮开发结果，无新完整精度或正式稳定结论。唯一 fresh 回切 4K 得到 `3830.87 tok/s`、Mean TTFT `5494.62 ms`，请求/token 完整；候选仅比该回切高 `2.84%`，两次 control 差 `218.89 tok/s` 大于候选超过较快 control 的 `108.97 tok/s`，因此不能宣称 4K 稳定 `+9.08%`，不继续回切。17:03–17:05，Q32/KV128 连续完整 tile 首门通过：两臂相同 slabs，独立 FP32 合同通过，最大 1 ULP 差异，真实 SME→TCU 直接操作数路径成立；q2048/context2048 的 42 层完整 eager 设备耗时 `67.1373→56.0508 ms`（降低 `16.51%`），含主机调用为 `68.7316→57.8408 ms`（降低 `15.85%`），零 spill、shared 96 KiB。17:18–17:21 补门中，全部 8 个真实 Prefill/Mixed 元数据状态及 5 组边界通过数值检查，完整 builder/bridge＋42 独立层设备成本降低 `13.47%～16.65%`；长历史连续页 `305.7193→251.7823 ms`（降低 `17.64%`），但完全碎片页 `305.2611→319.1855 ms`（慢 `4.56%`），显著超过噪声，按预设完整成本门关闭本包。仅撤回 Q32/KV128 两处启用改动，Prefill 精确恢复 Decode-only slabs 的 `14de9cb7…`；未启动组合整模，不扫参数救跑。保留前述 Decode-only 单轮结果，不自动运行 105/正式四轮或改写正式 B1。

10-23 CPU 页表证明＋编译期 Prefill 分派的局部门通过，但唯一原 CLI 单轮为 `3920.52/3266.15 tok/s`，未体现整模新增收益。随后独立副本的 16 步有界诊断确认真实 hook/资格/核分派接通，此 4K 早期窗口按完整 KV128 工作计覆盖 `65.63%`；仅为窗口观测，不能外推全程或 16K，亦未确定整模无收益的原因。observer 未进入 canonical，完整局部门、诊断与源码身份均留 raw/ledger。

三段 Prefill 的固定 batch 级 `7/8` 工作资格、一次计划 H2D、顺序 affine/gather/affine 与原 fallback 已通过全部批准局部门。首 step11 的 per-row bridge `3d3878d3` 与后续 batch bridge `68940f0a` 原件分开，不合称同源全过。另行批准的原 CLI 同服务 1+1 于 19:04:14–19:22:18 完整完成：4K/16K `3930.98/3248.05 tok/s`，相对保留 Decode-only 开发单轮 `3939.84/3267.40` 为 `−0.2249%/−0.5922%`，相对 pure-policy `3920.52/3266.15` 为 `+0.2668%/−0.5542%`。256/128 请求全成功、零失败，全部 input/output token 完整，Mean TTFT `6520.75/153168.83 ms` 在官方上限内；独立单轮不可称稳定或单变量归因。未见新增整模净收益，按批准停止条件关闭整个 Prefill policy/三段扩展路线，不做第四变体、新路径诊断或参数扫描。仅将本任务三文件经 apply_patch 精确恢复为 Decode-only slabs：Prefill `14de9cb7`、bridge `55af1e64`、runner `8a4c2469`，保留 Decode `07561a17`、mm `8401b9c4` 及全部无关 dirty。raw/冻结候选/恢复身份已同步，owned PID 清理完成；`3939.84/3267.40` 仍为保留开发组合而非正式成绩，正式 B1 表不变，无新 105、正式四轮、提交或推送。

有界 FULL 图计时首包因 RMS observer 未映射关闭；随后独立批准的唯一机械纠偏移除 RMS/整层事件，32 节点复用 QKV:start→down:end 为 body（不含输入 RMS，不称整层）。M64 uniform FULL 捕获的 42 层四 GEMM 及 Decode/partial/merge 映射通过，但首个合资格 replay 的完成 query 通过后，elapsed-time 读取报 `ixErrorInvalidResourceHandle`；原 4K 前缀仅运行 63.66 秒即停，0 有效样本，无法给分项/body/full 中位数、波动、长度轨迹或 Attention/GEMM 排序。按约定永久关闭此观察门，不再修事件或扩采样，不据此猜测厂商根因。两包 raw/身份与各自清理 receipt 均保存并同步，owned 无存活；canonical 五源及 `3939.84/3267.40` 开发组合保持，正式表不变，等待独立机制规划。

固定 KV32-direct-SME 首原型因 shared 仍20KiB在编译机制门关闭。另行批准的唯一 V 生命周期纠偏仅使 eligible KV32 的短/长、完整/尾 gather 在 QK/softmax 后读取 V（短 eligible 时序也改变；数学、mask、原长64量子分段、W4/S1/16splits/merge及非eligible路径不变），实际得到105regs、spill0、shared12KiB，直接SME→TCU保留。保存real sample0、真实页号/长度/table8192/完整32609-block pool、42独立层且两臂共用只读KV；42层双方独立FP32 .02/.02及分类/查询/输出归属/canary通过，最大FP32误差双方同为0.00050211，归约粒度改变存在BF16位差。完整partial+merge graph7ABA的event29.76345→32.27796ms（慢8.4483%，noise+drift0.07183ms），wall29.79925→32.31381ms（慢8.4384%），相对当前Decode-slabs原核显著回退。按首完整成本门关闭整个KV32路线，不第三修正、扫参数或扩补门；两文件已apply_patch精确恢复Decode07561/Prefill14de9，mm/plugin/无关dirty保持。两包原件、IR、身份和退出状态独立同步，owned无存活；未启动服务/105/正式四轮，保留3939.84/3267.40开发组合与原正式B1。

随后固定 Decode 整 split 证明＋无分支 affine KV64 两级流水包，8份相邻真实页表在当前保留 slabs 下的完整 KV64 工作覆盖为 `28528/33064=86.28%`，过75%信息门；仅代表该4K窗口。原16splits/64量子边界/KV64/W4/launchS1不变，整完整段一次GPU证明后，唯一 `tl.range(num_stages=2)` 调用既有affine helper，失败原循环、尾原mask。唯一完整编译为122regs、spill0，但shared69632bytes（68KiB）超过36KiB硬界，立即在编译资源门关闭；未声称异步重叠/等待依赖门已过，未作FP32、42层完整成本或后续补门，不换stage/tile/API救跑。仅Decode经apply_patch恢复07561，其余Prefill14de9/mm8401/plugin及无关dirty不动；CPU原件、完整IR、身份/退出状态与恢复源已同步，owned无存活，3939.84/3267.40开发组合和正式B1保持。

融合A8尾行补零接入包 `150955Z-w8a8-fusedpad` 已因完整两景退化关闭：固定两配置/M256阈值、量化数学不变，仅原量化核补零物理尾行以恢复A侧SME，无额外padding核。M255/M257独立数学、零区/canary/改A图和42层完整成本分别省device17.08%/38.26%、wall17.00%/30.62%；M1/63/65数值及真实loader/fake/fullgraph/配置hash/BF16KVslabs通过。实际服务仅42层gate/up为W8A8，QKV/O/down、Embedding/Norm/head为BF16，无长期BF16权重镜像，live参数3978612736B、KV550288tokens（控制521840；不当作VRAM峰值或退化归因）。候选原CLI1+1于23:29–23:48完成，4K/16K3703.40/3073.64tok/s，相对本次独立冻结Decode-only控制4021.71/3265.83为−7.9148%/−5.8849%；MeanTTFT10250.72/157294.10ms相对5441.63/151408.66增加88.38%/3.89%，虽过官方门但无整模净收益，256/128请求全成功、全部tokens完整。只属单轮探索，模型精度未知；不把局部PASS冒充端到端/精度通过。限定既有日志/默认Triton缓存未能确认逐M编译开销，不定因、不追加诊断或确认。23:48:04 owned清理remaining[]；仅本包增量apply_patch回撤：原型恢复bd9d98ea未注册、platforma4307747、bridge55af1e64，自有接入module移出生产但冻结候选/source/IR/raw保留；Decode07561/Prefill14de9/mm8401/runner8a4c及所有其他dirty保持。既有3939.84/3267.40开发组合、正式表不变，无105/正式四轮/SNN/提交；本固定接入失败不等于W8A8方向证伪。

独立 W8A16 大M配套门 `132253Z-w8a16-large` 只复用同一W8A8 adapter的raw Wq/Sw，scale一维view零复制，未注册/未接模型。固定M2048、256/256/32 W16S2/grouped8，BF16 TCU、A SME/B非SME、112regs/0spill/56KiB；tile内I8 cast后staging原件保留，不称数学原创。42层真实权重＋合成A通过CPU FP64量化数学对照（layer0的16完整输出行覆盖384CTA、后41层固定跨gate/up子集），尾257/259/65、zeroScale/input/owner/canary通过；不是模型精度。完整42层eager7ABA对当次mm8401，event `49.82010→107.64722 ms`（慢116.07%）、wall `53.96607→110.38909 ms`（慢104.55%），两者显著超过3SD+drift；设备中位数比 `107.65578/49.82415=2.160715` 超过1.01。按防退化门关闭本固定配套包，无第三配置/救跑/模型接入；原型作为未注册实验保留，既有W8A8包原件不改。raw/身份已回传、owned三PID全退出，生产五源、3939.84/3267.40开发组合及正式表保持，无105/整模/SNN/提交，后续独立规划另审。

本机只做源码、文档和 Git 静态工作，导入、编译、测试、服务、Profile 与 benchmark 只可在另获算力与实施授权后于目标卡运行。指定版本、规则问题、调用链及实验文档纪律遵循[项目 skill](skills/flagos-minicpm-competition/SKILL.md)，与用户最新投入禁令冲突的旧选项停止执行。性能实现仍限定于 `vllm-plugin-FL` 和 `FlagGems` 允许 fork，`vllm@ee0da84ab9e04ac7610e28580af62c365e898389` 保持只读；历史目标卡实施窗口自10-25 18:00开始，当时延长至10-26 05:43后只收尾；10月26日新增12:01–18:01租期及17:41/17:56截止点见阶段6；保留全部dirty，不提交或推送，未经重新明确授权不运行新105。

必须保持官方 benchmark、test cases、serve 参数、模型、采样、输出长度和评分输入不变。禁止通过超参数开启量化或投机采样，禁止调 benchmark，禁止只切换现成 operator 或删除主要选择逻辑取得性能，禁止直接合并最新推理框架分支；同时不得启用现成前缀缓存或按 `4096/16384`、case ID 写死分支。

每项成果必须包含可审查的实质计算或访存代码，报告精确映射实际激活路径；未知 shape、layout、dtype 和平台的未量化路径保留原实现，量化模块须有阶段6规定的同ABI正确通用fallback，不能在丢弃原权重后承诺回原BF16。不得为命中候选而切换既有 backend；租赁结果只称预验证，规则解释和正式结论以比赛页面、组委会书面答复及官方平台为准。

用户已确认：自研多个等价 Triton 内核并按泛化 shape guard 分派属于既有书面允许范围。后续主热点开发须有实质算子 variant、完整成本验证及原 fallback；该许可不接受单纯 backend 开关、缓存键捷径、隐蔽预热或 benchmark 特调，也不解除本文开头的投入禁令。

2026-10-23 用户补充转述组委会答复，纳入后续设计边界：

- 模型参数量化、自研投机采样机制优化可研究，仍需实质框架/算子开发与最终数据集精度，不能只改超参开启现成功能，不未经允许引入其它模型。用户最新要求学习并改进量化代码，后续主线改为阶段6的激活感知主矩阵量化；旧纯absmax W8A8 gate/up、固定down、W8A16配套和head接入包均已关闭，不泛化为整个量化方向无效。生产仍BF16，原型/evidence保留；本轮已获实施授权，仅在隔离候选中接入经过验证的量化配置，保留slabs精确兼容guard，不自动105，开发与正式结果保持分开。
- 官方 benchmark 自然产生的后续轮次高缓存命中率可以计入结果。保持原脚本与运行流程，不额外预热、不识别测试轮次、不改输入或缓存命中条件。
- 自研 Attention 可以按实际 query 长度选择分块，也可以在算子内部将已组成混合批次的 Prefill/Decode 分组计算并恢复原输出顺序，不改变请求调度、批次成员或主体选择与回退逻辑。现有 Q32/Mixed 和本轮 Decode 快路采用这一边界。
- 算子优先用 Triton 实现在 FlagGems，插件负责接入。正式正确性以规定的数据集评测为准，局部 FP32 差分用于工程风险筛查，不把“逐位相同”擅自升级为赛事规定，也不以局部通过代替完整精度。

官方A0与门槛如下；A0始终只指组委会公布值。吞吐有效性下限为A0×0.99；A0×1.01是涨点>1%的参考线，不是官方否决阈值，稳定收益还须超过实测噪声：

| 平台 | 场景 | A0 total tok/s | 官方下限A0×0.99 | 涨点参考A0×1.01 | A0 Mean TTFT | TTFT 上限 |
|---|---:|---:|---:|---:|---:|---:|
| BI-V150 | 4K/1K | 2028.01 | 2007.73 | 2048.29 | 11573.53 ms | 11689.27 ms |
| BI-V150 | 16K/1K | 915.15 | 906.00 | 924.30 | 599262.03 ms | 605254.65 ms |
| C500 | 4K/1K | 5089.645 | 5038.75 | 5140.54 | 3199.435 ms | 3231.43 ms |
| C500 | 16K/1K | 7029.675 | 6959.38 | 7099.97 | 27197.135 ms | 27469.11 ms |

所有候选采用同一验收链：

1. 先做算子差分、边界 shape、layout 和 fallback 测试；任何输出、token、EOS、长度或请求错误都会使性能结果作废。

2. 数值路径改变先完成局部差分和获批开发性能门，不自动启动完整 MATH-500 Level 3。只有用户重新明确授权最终105后才运行，仍须至少 `100/105` 正确并保留逐题原件；未重验时只能称性能预验证，不能继承历史精度。

3. 开发优先复用有效证据，默认只做必要的一至两轮原 4K/16K 诊断；用短配对、恢复 B1 对照或消融解决明确的漂移与交互问题。最终组合统一运行官方 4K/16K wrapper，各四轮、丢弃第一轮，报告后三轮 mean、SD、CV 并保留所有慢轮；不要求每个子项重复三臂全程，也不建立穷尽冷暖状态的测试框架。

4. 要求所有请求成功、无 OOM，Mean TTFT 不越表中上限，两个场景的吞吐收益均超过实测噪声；只有现有证据不能支持改后结论时补必要 Profile，并在另一平台执行适用路径与 fallback 回归。正式评分不带采集工具。

5. 只有 microbenchmark 收益、路径未激活、无法映射最终 diff 或端到端落入噪声的补丁不保留。相关修改积累成完成正确性、性能和保留决定的阶段成果后再批量提交，除非用户另行授权。

每次远端实验结束后立即同步原始结果和源码身份，并原位更新相关学习文档；沿用现有 SNN，不为一次小实验新建文档或小提交。每个平台最高编号现存 SNN 末尾维护唯一 A0/B1 四行表、完整精度摘要与统计口径；依既有创建门槛或用户明确许可新建专题后立即整段迁移而非复制，旧页只保留链接，索引与计划同步。迁移不改变数值、接受状态或租赁预验证边界。

## 3. 七阶段实施

### 阶段 1：规则、环境与可信基线

**问题与目标。** 复用已确认的规则、运行路径和基线，避免重复核查挤占主热点开发；只补当前变更影响的证据。

1. 沿用已冻结的设备、软件、源码、实际 import 和 backend 身份。只有环境、相关代码、规则发生变化或出现矛盾证据时，才复查受影响部分。

2. 新主核候选核对模型类、注册、dispatch、编译实现与设备 kernel 的实际连接，复用已有合同与 fallback 测试，不重建环境或辅助验证体系。

3. 按改动范围执行必要的局部差分与短、长、mixed smoke，核对 token、EOS、长度和请求成功。新 105 仅在用户重新明确授权后执行；未经重验的变更不继承历史精度。

4. 使用同环境有效基线和已知波动；只有对照失效或待解释异常才补短控制，不强制重跑多组基线。

**验收。** 当前候选身份、活跃路径与适用正确性有据可查，未完成的完整精度和官方平台门明确保留；不重复验收未变化的历史结果。

### 阶段 2：主热点定位与端到端归因

**问题与目标。** 用现有有效证据区分主计算与真实等待，按可消除的端到端时间选择工作，不以任意覆盖率或重复重型 Profile 阻塞开发。

1. 优先复用同源码、同路径的有效窗口，分别标明 4K/16K、Prefill/mixed、Decode 和 tail。新开发只面向主 GEMM、Attention/KV 主计算，或已证明阻塞主工作的系统等待。

2. 必须补采集时，使用目标版本匹配的 mcTracer 与实际 GPU EngineCore 子进程，窗口 2–3 秒、最多 5 秒，不叠 Torch profiler、不升级 SDK。分开记录 runtime/API、device kernel、stream/correlation 和 graph 节点，隔离采集与导出干扰。

3. FULL 未展开部分保留 opaque；设备窗口占比不等于端到端占比，也不代表全部可改或可消除。仅凭持续时间不推断带宽、spill 或容量瓶颈，KV 读取不与 Attention 重复计数。

4. 对候选估算真实可省工作，计入 dispatch、workspace、编译/调优及图执行完整成本。正确性或净成本失败即关闭该机制，转向另一个有证据的主热点；不盲重参数变体。

**验收。** 候选有活跃路径、主工作占用与完整净收益依据，推断边界清楚；已有充分证据即可推进，正式评分始终不带采集工具。

### 阶段 3：C500 4K 主 GEMM 与真实冷调优停顿

**问题与目标。** 优化真实主 GEMM 的计算与调优总成本。一次慢轮取栈只支持继续定位，不足以把所有停顿归因于 LibTuner 或缓存；历史 `M1` 已保留，不重开同步或分配清理。

1. 从既有记录关联真实 M/N/K、dtype、stride 与 `NT mm_out` 调用，区分主核执行、首次编译和真实冷 shape 调优等待。权重物理布局保持不变，不把历史局部冷成本收益当作整模收益。

2. 仅当现有证据不足时，补一次不带 profiler 的定向原 4K 诊断，关联停顿窗口、调用和 shape。仍无法定位即停止该诊断分支，转向另一主热点，不无限重复取栈或长跑。

3. 最多准备两个有明确机制依据的实质 Triton GEMM 变体，通过分块、计算流水或块内访问减少主工作成本；按泛化 M/N/K、dtype、layout、device guard 分派，保留原主要选择逻辑与 fallback。不扫参数，不走缓存键捷径、隐蔽预热或现成 backend 开关。

4. 先比较受影响真实 shape 的完整成本，包含 dispatch、workspace、编译/调优冷启动、首次调用及实际 eager/graph 路径。局部成本或适用覆盖失败即关闭候选；有净收益依据才进入必要的原 4K/16K 开发验证。

**验收。** 同一权重物理布局上，实质主核或真实冷调优总成本下降，收益覆盖完整路径且超过噪声，正确性与 fallback 通过、另一场景无稳定回退。无法定位或成本失败时有明确停止结论，不以重复诊断代替优化。

### 阶段 4：BI-V150 已验收组合与主计算深化

**问题与目标。** 保留 Q32/KV128/Mixed/runtime/GEMM/SQL 当前 B1，只在新证据指向主 Attention 或 GEMM 时继续开发；已验收范围与尚缺的跨平台出口分开陈述。

1. 保持自研 paged-GQA 主核：16Q/2KV、BF16 输入输出、FP32 online softmax 和原 scale/causal。`max_query_len == 2` 时优先 Q2；最大 query 与 batch 平均 query 均至少 32 才选 Q32，否则沿既有条件选 Q16/Q8。Q32 使用 16 warps/1 stage，实测 shared 96 KiB、reg 124、spill 0，所有页边界与读写 mask 保留。

2. 保持 Q8/Q16/Q32 完整共同历史的 128→64→32 处理，page16/32/64 的 8/4/2 页映射、Q2 与因果尾，以及原直接序列索引 Decode。Mixed 仍由原 16 段 Decode 写 q1，再由 Prefill skip_q1 写较长 query；数量参数为 runtime，不按值特化，builder/bridge/merge/launch 均计入完整成本。

3. 既有 24 项加新增 11 项 FP32/所有权/输入不变性及 7 项仓内回归通过，完整 102/105 与原双场景各四轮已验收。改后唯一单 4K 初始混合窗为 4292 kernel、合计 1374.981841 ms，其中 GEMM 59.832315%、Prefill 27.977644%；metadata、资源和源码联合确认 Q32 与部分 128-key 共同历史。该比例只属于此设备 kernel 窗，不外推 16K、纯 Decode 或全程，FULL 内部仍 opaque。

4. 后续仅针对证据支持的主 Attention 计算、KV 主核读取复用或主 GEMM 提出实质机制，按阶段 2 的完整成本排序。通过原 `TRITON_ATTN` 受限实现接入，保留主要 selector、平台 guard、FULL 资格及未知 shape/layout/feature fallback，不为命中候选切换 backend。

5. 跨平台只补适用路径与 fallback 兼容性回归，沿用现有分派边界。当前 BI 结果和冻结源码保持原身份，唯一正式结果继续维护在 [S03 第5节](docs/B1-V150/S03-天数混合注意力与矩阵访存及配置加载优化.md#5-精度与官方正式结果)，不以新候选继承旧分。

**验收。** 已完成的 BI 局部正确性、102/105、原双场景四轮和请求/token/吞吐/TTFT 门保持有效。未来主核变更单独验证；现有跨平台适用路径回归、双平台干净复现与最终出口仍未完成。

### 阶段 5：C500 16K Attention/KV 主计算

**问题与目标。** 面向长上下文的主 Attention 计算和 KV 重复读取寻找实质收益。历史 vendor Attention 约占采样设备 kernel 窗的 61%–63%，不是端到端占比，也不代表全部可编辑或可消除；KV 读取已在其中，不重复计数。

1. 在真实 query/KV 长度、分页和 GQA 头映射上核算主 QK/PV 计算与重复 KV 读取。候选必须写明能省掉哪部分主工作、适用范围和收益上限，不能用局部 kernel 百分比代替端到端收益。

2. 只为能减少主计算或重复读取的新机制实现有界原型，先核算 planner、padding、workspace、launch、partial/merge、编译与实际图执行的全链成本。首个真实长上下文几何成本失败即关闭，不继续参数试探；有依据才进入必要的原 4K/16K 开发验证。

3. context 分段工作重分配仅为待证假设：先证明跨请求/跨 KV-head 的真实不均与尾波，再证明分工和归并的净收益；未有上述证据前不评为高概率或接入原型，不假设共享前缀，不把已有 split/GQA pack/cascade 当新开发。

4. 失败机制保持关闭：R2 paged online Decode 完整图成本慢约 54%；同 CTA 全段 online 重算首长形状明显回退；native 单生产者 KV 原型首真实几何完整 forward 增加约 24.2 ms；直接省 PV shared 往返的窄机制亦已关闭。没有实质新机制不重开，不以 tile/split/warps 微调重试。[固定 native 源码](https://github.com/MetaX-MACA/flashattn/tree/8f279843ce18cb5c1c9e695267e28060d0b6273c)只作已记录的布局研究依据，公开 pin 未绑定安装二进制。

5. 保持 BF16、FP32 softmax、scale/causal、head mapping、分页和输出归属语义，覆盖边界与 mixed、代表性非基准长度及 fallback。实现限定于允许 fork，保留原 backend selection、主 dispatch 与 BI/native 兼容边界；不修改 SDK、服务参数或评分输入。

**验收。** 实际激活的主 Attention/KV 代码减少完整路径成本，局部正确性和兼容性通过，端到端收益超过噪声且另一场景无稳定回退。租赁结果只称预验证，完整精度与官方双平台验收按统一门执行。

### 阶段 6：激活感知量化与模型级部署成本

**目标与当前实现。** 用真实激活校准主要Linear，以完整部署吞吐决定保留。固定alpha=.5，对X/s与W*s配对缩放，inverse-s、RNE A8和必要尾零在FlagGems融合，加载时生成W8与FP32尺度。实际保留候选白名单为42个QKV及41个gate/up，第7层gate/up保持BF16。native INT8 GEMM归属于已安装ixformer，插件负责真实QuantizationConfig、白名单、版本、ABI与同量化ABI回退。Embedding/Norm/head、Attention与KV保持BF16。

**本轮结果。** v2的原wrapper每景四轮严格去首后三，4K `3811.89→4063.04 tok/s`（+6.59%，候选CV4.98%），16K `3108.93→3378.75`（+8.68%，CV0.1767%）。16K有效轮稳定分离，4K末轮与对照范围重叠，双景稳定条件未完成。全部请求/token/TTFT过门。独立test PPL增加0.2381%，只属有限自然文本质量证据，未跑新105、不继承旧分数。原BF16组合及正式表保持。

完整冻结源、原始统计和后续实验索引见[本轮汇总](evidence/bi150-20261025-152700Z-native-w8-fourround/session-summary.md)。精简前详细过程与全部旧失败原数留在同目录`plan-before-session-condense.md`及各独立证据包。

**已解决的实现问题。**

- 原逻辑M加载谓词使INT8 A-side SME失效；按已初始化物理tile加载、输出按逻辑M截断后恢复SME，边界整数oracle通过。首版自己的W8计算局部有收益，但整模−9.91%/−7.83%，未保留该计算路线。
- native W8计算与融合量化组合使完整84模块M2048图成本62.37→26.38ms，余下M32/40/64及角色门完成后才接入v2。完成wall与设备Event跨度分别保留，不把主机提交间隙当kernel self time。
- 真实数据使用固定WikiText revision、train32/validation8，test不回流校准。算术oracle、量化输出误差、整模质量与吞吐各自判断，真实权重加随机A不能代替模型质量。

**已关闭的扩展。**

- v3增加40个O和6个down，M<256保留原BF16、M≥256使用W8。完整局部成本通过，但同机整模4K/16K为−4.59%/−2.88%，关闭。
- v4将40个O改为全M单份W8，准确释放320MiB BF16副本，Worker非Torch增量却增加约284MiB，实际KV仅略增。整模4K/16K为−4.47%/+2.36%，不保留，非Torch综合指标未定因为workspace。
- W4 per-channel投影重构0/84模块通过原精度门。AWQ G128大M完整成本显著回退，group64 W8A16小M也回退，停止格式轮换。
- down误差分解以A量化误差为主。训练极值通道并集、固定32步训练输出误差反馈分别只新增6/10个模块通过train与validation，均未达18个投入门，没有残差性能核或额外模型轮。

**尚未解决的部署问题。** 4K在多次fresh量化配对中出现约25秒首段额外耗时。原有配对timeline显示匹配Prefill/mixed GPU活动略有收益，也记录到不同Attention变体的首次编译，但没有解释全部慢时。宿主诊断一次未进入真实step，一次约7秒后EngineCore退出，窗口不完整，不再追加第三次采样。

本轮末对现有四轮原日志作只读分解：v2的4K run4比run2总耗时多30.98秒，完成请求1/64/128/192/256处的累计时间差约为7/12/16/24/31秒。额外耗时分散在多个区间；对照run2在77→78也有约6秒间隔。该模式不能与fresh首段延后直接合为同一根因，也不支持仅用第一次native加载解释四轮波动。证据见本包`fourround-request-milestones.json`，显示精度为整秒，没有据warning数量猜测实际编译或调优耗时。

runtime BLOCKS旧补丁虽复用容量编译hash，首完整成本慢1.125%而关闭；三次Prefill编译对应不同Q tile/SKIP_Q1，不能把10.2秒全当作容量常量可消除收益。native二进制只读检查确认调用关系及24字节主机handle，没有各内部步骤的耗时或实际grid，不支持猜测性handle缓存、预热、split-K或修改SDK。

**10月26日新增租期与限定诊断。** 用户新增北京时间12:01–18:01六小时租期，17:41停止新实验并同步、17:56前退出自有计算。经Astra ultra计划`BI-20261026-LATE-WARM-ATTRIBUTION`批准，仅在隔离证据包复用冻结v2同一服务，连续执行两次完整且不改参数的4K原CLI，两轮均观测；每CLI含输入准备限480秒、启动180秒、总墙钟20分钟并保留30秒清理，不因首次调度等待超过旧60秒而截断。按run_id分别记录实际phase/graph descriptor、wall与thread CPU、既有JIT/LibTuner/native聚合、稀疏图外ordinary events（首4步后每64步、每轮至多96对、pending至多8），外部1Hz采集已有ixsmi及精确Engine `/proc`/schedstat/cgroup计数。相同步点在计时前只读现有CPU页表统计affine KV64覆盖并记录自身开销。结果只作时间归因诊断，不作新成绩；无第三轮、16K、105、生产核变更或SNN/commit/push。新实质候选须另立规划。

**新增诊断完成。** 上述两轮于12:11–12:23完成，同一Engine原4K为307.94/314.61秒、4256.39/4166.18 tok/s、Mean TTFT4737.25/3214.12ms；两轮256成功/0失败、tokens完整、CLI0，70/70图外Event排空且observer错误0。按64完成请求cohort的M64纯Decode设备均值42.97/45.36/56.21/54.36→48.13/52.20/56.39/53.40ms，与affine覆盖87.84/75.93/0/15.26%→52.84/28.36/0/21.70%相伴；精确phase/descriptor/tokens/seq_sum/max匹配分层保留，不能据此宣称全部6.67秒同因。主线程run_delay10.57/8.39ms、时钟1500/1600MHz稳定、最高54°C；cgroup计数不可用，不声称零throttling。原件已同步并校验，supervisor0、精确自有进程remaining_live=[]；该包只作归因诊断，不更新四轮成绩或完整精度。见`evidence/bi150-20261026-041000Z-late-warm-attribution/README.md`。

**保留核布局成本门。** 随后的`BI-20261026-RETAINED-DECODE-LAYOUT-COST`复用160900Z四对完全相同query/seq向量、原始page16页表与同42物理池基址，用跨四对CPU等价类初始化保证逻辑KV内容相同。07561 Decode/14de9 helper零改动；首对7ABA设备及完成wall差超过原3SD+drift才继续后三对。四对42层FP32/分类/归属/canary均过，round2相对round1完整设备差4.914/4.752/4.762/4.823ms，完成wall也过门；仅确认保存布局的相对代价，不将其全量换算整模慢时。22.59秒自然exit0，精确owned remaining_live=[]、完整原件同步校验；未改生产/调度/服务或开启105。见`evidence/bi150-20261026-042800Z-retained-layout-cost/README.md`。

**固定H128新机制信息门。** 经另行批准的`BI-20261026-DOWN-H128-INFORMATION`仅对原192200Z train/validation快照与alpha=.5训练尺度作48个独立H128块的配对FP32自然七阶段旋转，归一化一次、量化前无BF16舍入，沿原CPU FP64 scale/RNE oracle及native M256 ABI评估真实快照位置。42/42层在两split均过原整体3%/最差行10%门，36个原拒绝down同时恢复，原6层零损失；最大joint整体相对RMS为0.016964/0.016599，最差行为0.038779/0.039777。保存未量化旋转、A-only、W-only与joint及完整来源身份，28.46秒自然exit0、remaining_live=[]、raw回传校验。只放行独立融合实现/编译成本规划；未实现生产FHT、未作模型吞吐/新105，不宣称完整质量或新B1。见`evidence/bi150-20261026-043100Z-down-h128-information/README.md`。

**融合H128固定包关闭。** 在上述信息门后隔离实现同一A/W融合diag→七阶段H128→RNE核，编译85regs、spill0、16KiB shared、无global scratch或全激活中间量；首真实W及M1/33/257 CPU码/尺度、INT8点积1BF16ULP、fallback物理尾零、归属/canary与两次变输入图均过。完整42个down的7ABA成本却为M2048 graph25.2153→33.0715ms、M32 graph2.0220→3.1357、M64 graph2.3069→3.3817；M2048 eager完成wall28.1931→35.7535ms。预注册wall/device屏幕指标S分别−11482.95/−8595.73ms且G为负，固定all-M单W8包关闭；不得以局部质量或显存节省晋级，不开BF16镜像、M-cut或参数扫描救跑，无真实模型/生成/新105。初次schema和CPU默认dispatch的setup失败原件保留，机械修正后只恢复成本、未重复已过数值门；实际成本16.88秒自然exit0、remaining_live=[]、全部raw同步校验。见`evidence/bi150-20261026-044000Z-h128-fused-cost/README.md`。

**矩阵H128新机制完成并关闭。** 另行批准的`down_h128_bf16dot_v1`以实际BF16矩阵硬件替换已关闭scalar butterfly跨lane阶段。前置050000Z在diag后BF16 RNE的CPU surrogate信息门42/42 train/validation通过，36旧拒绝层恢复且旧6零损失。050200Z固定A64×128/H128×128实际BI matrix.mad编译170regs、0spill、32KiB shared、globalScratch0；FP64矩阵oracle码差实际0、scale2ppm、I8dot1BF16ULP、fallback/尾零/H归属与两变图均过。用实际GPU dot重验全部42层train/validation同样全过、36恢复/旧6零损失，不能替代整模质量。完整42down 7ABA却为M2048 graph25.2171→33.5846ms、M32 2.06292→3.10281、M64 2.30109→3.35922；wall/device S为−9361.46/−8535.88ms且G为负，固定all-M矩阵包关闭，不做H/warp/stage/M-cut/BF16镜像救跑。19.28秒自然exit0、精确remaining_live=[]、全raw同步校验，无服务/模型/105或新B1。见`evidence/bi150-20261026-050200Z-h128-matrix-cost/README.md`。

**四page16加载拼接单KV64机制关闭。** 051000Z从07561/14de9只加181700合法guard，四K/V页的自然join/permute/reshape后仍沿原single QK/64softmax/single PV，固定W4/S1/16splits，源码guard与顺序静态通过。实际完整Decode编译候选169regs/0spill/20KiB，但new fragment最终K64/QK及V64/PV均useSme=0；LLVM八次SMEload全部映射保留affine helper，而非新分支，aggregate标记不能冒充新路径激活。主代理确认严格IR门失败，未跑数值合同或成本，不作性能结论、不扫layout/tile/warp救跑；这只关闭本拼接lowering，不泛化为page16 SME硬件能力缺失。11.92秒编译结束、外监督15.62秒exit0、remaining_live=[]、raw同步校验。见`evidence/bi150-20261026-051000Z-fragment-fourload-one64dot/README.md`。

**固定32步学习对角缩放信息门关闭。** 052200Z仅学习每层6144个正对角参数s=s0*expδ，alpha_init=.5只表示初始s0；固定全train32步Adam、lr.01/原betas/eps、δ夹±ln4，max/scale可微的rowwiseW8/A8 STE，W/X/FP32参考冻结，无LWC/bias/扫描/validation反馈。看见任何学习输出前已修正投入门为交集≥24且旧36恢复≥18，旧6得失只作诊断。42层均完成32步并以最终s/CPU FP64 reciprocal及原q8/native TN评分，但训练仅21过门（旧6全保留、旧拒绝15恢复），训练上界已不足24/18，所以未动held-out验证或成本/模型；不能把未验证记为验证失败。最终尺度/δ/逐步loss与完整原件同步，20.63秒正文/26.79秒监督自然exit0、remaining_live=[]，没有参数救跑、新105或生产改动。见`evidence/bi150-20261026-052200Z-learned-diagonal-information/README.md`。

**独立INT8 KV隔离线。** 用户新授权仅在独立候选研究INT8 KV，临时BF16 KV研究限制在该范围被取代；当前v2/production仍保持原BF16 KV。053600Z沿冻结v2原serve以两个不相交TRAIN token区间捕获全部42层4K/16K完整post-RoPE KV及32真实Q，82.01秒exit0、prefix洞检测全过，全部683MiB raw已回传校验。真实inline132 ABI（128 I8+4FP32 scale）自研RNE fused writer初版在负BF16subnormal发生−128/−127差；055300独立诊断证明graph1=graph2=eager，仅K slot2双head256 codes不同、scale bits0x810完全相同，不是graph特有故障。060100唯一精确2^24等比例normalization保留原scale/RNE/flags/normal路径，原序列及固定正负极小/zero/tie/maxfinite/invalidslot逐字节全过。060800以正确KV32/64分区reference及两次改Q/length/table/无效页图核验，42层×两context全部84实际QKV噪声门通过，最大整体RMS0.013864/最差head行0.074066，无全局dequant cache。

061300获批的一层full-pool 7ABA信号为4K设备.83293→1.63714ms、wall1.05245→1.83157；constructed17408 B32/B62设备peractualtoken.03772→.10172ms（约2.70×慢），wall约2.36×慢。未触及pilot扣噪声>2×单独关闭门，不能说全42冻结10%门已经失败；Astra与主代理随后批准`CLOSE_POOR_PILOT_ECONOMICS`，基于16K每实际token约2.70×变慢而自愿提前拒绝当前inline132方案；全42、Prefill/Mixed、native fallback均未运行，4K预注册>2×扣噪声停止门未触发，不能记为全42门失败。实际IR I8→BF16→shared→TCU useSme0、旧affine路径恒不命中，作为机制线索，不声称全端到端归因。各owned进程均exit/no-live、三组raw已同步校验，无KV8模型/TEST/105/正式成绩或生产注册。见`evidence/bi150-20261026-061300Z-int8kv-full42-cost/README.md`。

**Q/P INT8 独立信息门。** 用户随后明确允许独立验证 Q/P INT8 Attention。固定 signed Q8/P8、FP32 softmax/累加的真实数据模拟在第78组（16K、layer35）出现最差查询头误差10.2323%，超过预注册10%门槛，整体误差1.9581%；前77组通过、其余6组未跑。6.44秒完成并保存原件，未进入编译/成本或模型，门槛不变。下一步仅依据该失败样本区分Q与P误差来源，再决定是否有实质量化修正，不能按层挑选或放宽门槛。见`evidence/bi150-20261026-065300Z-qp8-information/README.md`。

固定失败样本旧分解是在KV8基础上再加Q8，使最差行误差由4.48%升至10.13%，不是纯Q8+原BF16 KV的误差；P8分支为4.17%，非负P完整无符号8位范围仍为10.12%，关闭该修正。固定Q-D64分组使原失败样本降至8.35%，但全组在16K/layer08达10.71%而停止，不扫描组大小。两包合计正文约6.7秒，未进入内核/模型；原件`070000Z-qp8-decomposition`与`070400Z-q64p8-information`保留，本轮真正纯Q诊断见下。

**两项残差 Q8/P8 原生实现与舍入闭环。** 070900固定机制信息门通过后，071200首原生 I8 核为177regs/0spill/13KiB，affine K/V最终operand保留SME、没有BF16 KV展开，但low-scale probe先失败，未进入真实84组或成本。原probe未固定/保存输入；072800另建seed33固定样本，CPU FP64归属证明GPU low-scale匹配单次FMA而非两次舍入，不能冒称原输入精确复现。073300改用mul_rn后，同一保存输入仍未达到两次舍入合同，关闭该修正；本包无额外ASM，不声称直接汇编证明。

073600另立显式FMA残差定义，以独立远端CPU FP64 oracle重新核验，信息门84组及两个已知失败预检、实际原生核84/84真实QKV、边界与两次变输入图均通过，最大整体RMS1.77491%/最差行7.59350%，原3%/10%门不变。随后一层full-pool完整writer+Decode+merge的4K 7ABA设备0.8330896→2.7690571ms、完成wall1.0564645→3.0276331ms，均失败于本包noise-inclusive≤10%成本门，`CLOSE_COMPLETE_COST`；16K容量场景及全42/Prefill/Mixed/模型均未运行。此为独立算子开发筛选，不新增模型质量或正式成绩。四包README保存具体source、JSON/日志与退出凭据，后续fastquant独立源不由此继承数值通过。

**10月28日原v2信息门与上轮收尾。** 本轮租期12:29授权后至18:24，18:04冻结同步、18:19退出自有compute。`BI-20261028-V2-NATIVE-GRID`仅观察冻结83模块v2真实native调用的quant/native名字、correlation、grid/block与设备时间，主体M64并以M2048作对照；完整graph直接计量quant+GEMM+output，不把eager分项相加冒充graph。旧M64全83模块3.8593ms不能承诺消除页布局4.75–4.91ms差异。无候选源码/serve/benchmark/recipe变化，正式表不动。

本信息门在ixsys前置处关闭：首启动打印GPU trace缺cuda_device，按现装help机械补正后仍因容器tracefs/ftrace control权限exit1，两次应用均未启动（约.054秒），没有导入加载/实际profiler窗口或无profiler图成本。`CLOSE_IXSYS_CONTAINER_PERMISSION`不构成GPU或native性能结论，不支持split-K立项；未改权限/容器或第三次采样。两次exact-owned清理及source/命令/raw gzip归档见`evidence/bi150-20261028-043019Z-v2-native-grid/README.md`。

唯一替代`043900Z-v2-native-grid-torchprofile`的采集及自有计算退出在原5分钟界内完成，CPU归属/文档同步延至12:43，包收尾超过12:41:46界但仍早于原12:45绝对界。首次夹具误用空plugin令v2 op注册CPU而在profile前停，原件保留；机械恢复真实服务FL platform后，仅一次CPU+CUDA eager窗口.09899秒，334 GPU kernels/168模块markers，83W8×两M各quant+native成对、2个BF16调用，grid/block/correlation全唯一对应。16SM上M64 QKV grid20/block256，native中位23.5595µs；gateup96/block256、62.615µs；M2048分别10×8与48×8/block1024。M64 QKV供给1.25CTA/SM是实测grid事实，非整模根因或收益预算。独立无profiler全84逻辑（83W8+1BF16）graph为M64 3.72683ms/M2048 25.18588ms，不合计eager分项冒充；成功正文15.49秒/监督21.01秒exit0、remaining_live=[]，raw SHA577ffa57双端一致。此信息门当时尚无split-K候选、服务或新模型质量结论，后续固定候选见下一段。

固定结构`044633Z-v2-splitk2`仅隔离M≤64/K2048两原投影形状，BM64/BN128/BK64、W4/S1、两1024片INT32 partial与整数merge后原尺度次序。partial首编译63regs/0spill/12KiB、A/B SME与I8→I32 TCU过；layer0 gateup M64 A8/scale/两片及total严格CPU位等、final独立分步FP32 oracle位等，首输入vs native位差0/maxULP0、owner/canary/两变图过。首41W8+layer7BF16完整42 gateup graph设备2.791111→3.878977ms（慢38.98%）、completedwall2.827856→3.918249，原margin .055061/.051963，`CLOSE_FIRST_GATEUP_COMPLETE_COST`，立即停止未跑eager/QKV/全84/其余形状/83validation，不扫split4或unsplit救跑。capture reserved增量差6MiB，partial写读计成本；首成本正文15.078秒/监督20.175秒exit0、remaining_live=[]，源码/IR/raw回传，canonical/v2/正式表不动，无服务/模型质量继承。见该包README。

独立`045947Z-v2-fullk-lowshared`基于native gateup96CTA/16SM与24KiB shared的驻留供给问题另行批准，仅gateup M≤64、固定64/128/64 W4S1，去掉跨CTA通信，完整32步I32后原尺度。生产核63regs/0spill/12KiB且A/B SME/TCU过，首layer0数值与owner/canary/两变图过；完整gateup42 graph却为2.702702→2.707923ms、wall2.746668→2.752910，差在噪声内且无正收益，`CLOSE_FIRST_GATEUP_COMPLETE_COST`。未继续eager/guard/fallback/全84成本/83validation，不把资源更低或驻留上界提高当收益。无partial仍出现与split2相同的B16/C22MiB capture reserve增量，不能据此把reserve差归因partial；逻辑partial写读的成本事实不改。正文14.913秒/监督20.015秒exit0、remaining_live=[]，生产v2与吞吐不变，见新包README。

`050851Z-real-gather-counterfactual`只比较原gather control与retained，源码仅ALLOW_AFFINE一行差、共用14de9 helper，原slabs/42独立池/原页号/query完全相同。按数据前CPU计数选160900 round2/sample3（四态中最碎片，完整KV64 affine56.39%），只测该一态：完整42层graph retained35.060931→gather40.572370ms、wall35.099599→40.604656，gather慢5.511/5.505ms且显著过噪声，`CLOSE_NO_GATHER_NET_ADVANTAGE`。首层FP32与42层分类/输入/canary通过，正文12.828秒/监督18.603秒exit0、remaining_live=[]，raw同步；其余三态未测，不追加服务或新kernel。结论只限所选真实布局+合成等逻辑KV，不得覆盖041000的0% affine cohort、逐segment上界或全部慢轮/所有fission机制，见该包README。

13:24用户新许可解除本轮量化扩展范围限制后，`052434Z-qresidual-fastquant-cost`只补冻结075000核110025a2未完成成本，writer/双项Q残差FMA/P8/nativeRNE/tinyfallback不变，KV理论1024→528 B/token/layer仅为投入依据。复用已有84 TRAIN capture筛查而不冒称held-out质量；summary module遮蔽只机械改candidate_times，数据前4K event与completedwall均须超原3SD+drift净正。首单层原4K metadata/B64/full32615页完整eager链却为.837877→2.578545ms、wall1.071162→2.846063（noise .033784/.074793），`CLOSE_COMPLETE_COST`立停，未启动16K constructedB32/B62、42层或模型。正文18.988秒/监督23.068秒exit0、remaining_live=[]，全源/raw同步，生产v2与最新吞吐/正式表不变；不扫参数、不继承105/PPL。

`054000Z-bf16qp-kv8-scalemoved`仅保留本地8badc75b原型，状态`UNRUN_STATIC_CLOSED`。Astra撤回此前运行批准：旧061300 reader第138/153行早已在QK后乘Ks、P中乘Vs后BF16 PV，并非逐D128 dequant；新组合仅同528 B/token/layer的存储对齐/scale分离，缺乏弥补旧约2×完整成本的依据。未上传或启动远端/GPU、未作范围/编译/数值/成本检查，不记为numerical/performance失败；旧原件与生产结果不动，之后只读待命。

`060429Z-konly-centered-information`按固定单Q8/Kc8/VBF16/PBF16，先以train4k全部K在CPU FP64→FP32冻结全部42μ与身份，再看任何输出。首train16k/layer35纯Q诊断整体1.179352%/最差7.644662%（6870/head14），不是门或严格下界；候选整体1.560658%/最差8.475081%（6870/head13），指定head14为8.048415%，不支持μ直接修纯Q误差。首候选过后一次筛查在train4k/layer8最差10.016006%（792/head10，整体2.386526%）首败，`CLOSE_KONLY_CENTERED_QUALITY`，10案9过1败、余74未跑，不删query或改3%/10%门。μ冻结2.409秒、正文3.266秒/监督6.346秒exit0/no-live；源/μ/原件回传，无性能kernel/成本/模型/105。1024→776B仅理论容量依据，train16k仍是开发筛查，见该包README。

最后固定Qmean范围门`061720Z-fixed-qmean-range`只远端CPU/8线程，无设备或Attention计算。全部42份μQ由train4k各32条Q的FP64均值→FP32先冻结，随后train16k统计，Q身份/位置覆盖全过；所有分位数预先固定nearest-rank。仅30/42层满足13/16heads的absmax-ratio median≤.80（要求34），全体row/head P90=1.019650/P99=9.109945，均不满足≤1/≤1.10；`CLOSE_FIXED_QMEAN_BRANCH`，不挑head/换μ/新增变体。RMS仅报告，不能将median范围下降当模型/吞吐门。正文5.041秒/监督7.987秒exit0、remaining_live=[]，源/42μ/逐行与层头统计raw回传；无GPU、性能核、模型或生产改动，见该包README。

上轮`085300Z-fragment-segment`原件现已完整回传：编译5.4225秒、168regs/0spill/20KiB，只有首metadata状态42层数值合同通过，其余四状态未运行。42层完整partial+merge graph设备34.74623→40.05743ms（+15.2857%）、完成wall34.78476→40.09505ms，`CLOSE_FIRST_FULL_COST`；主16.34588秒、监督21.80886秒exit0且remaining_live=[]。只隔离candidate，从未改canonical，不以首状态数值通过冒称全部补门完成。见`evidence/bi150-20261026-085300Z-fragment-segment/README.md`。

**10月29日四页物理容量CPU门。** `045423Z-fourpage-reserve`只增加隔离自有FullAttentionSpec/Manager，统一总容量target并沿super计数/touch/free/cache，new_block_ids为子类补一次；仅逻辑四页对齐、全为本次新增且ID集合连续的组排序，两种返回表同步，旧prefix/KV不移动。当前已授权的实质KV管理不等于改淘汰策略或缓存参数碰成绩。

首CPU固定序列使用现装类、32615总块（含null），64req/4096 prompt每16tokens轮流增长、1024 decode、三轮cache/free/reuse；baseline三轮prompt/decode均0%，candidate prompt为100/74.0479/0%、decode加权100/66.2661/0%。第三轮无改善，`CLOSE_NO_REUSE_COVERAGE_GAIN`，只证明此合法复用模式不稳定；真实首轮约98%布局不同，不作完整慢轮因果结论。日志原FIFO跨边界/descending非连续ID组不能被批准规则排序，可用32614=4n+2仅为相关机制线索。预测/实际容量、新tail/IDs、reserve/hash及基础refcount/freequeue检查过，但32req、prefix/cancel/低容量/maxlen/影子KV完整硬门未跑；不称全正确。正文15.169秒/监督20.614秒exit0/no-live，源/raw/身份回传，无GPU或生产接入，不改pool/registry/scheduler救跑，见该包README。

`051419Z-bounded-extent-first`只查前8 free anchors、固定physical quad，经public remove/prepend与原pool分配；旧KV不移。四页page16匹配既有KV64；额外≤3页/请求，相对原逐page容量最多48tokens，64并发最坏3072tokens/约126MiB预分配池容量，不是新增显存或恒定占用。cached-free选择确实改变，沿原调度/准入完整计费，实际batch可能改变；不改cache开关/CLI/上游。

原CPU三轮覆盖/必要计数与内容硬门、actual public grouping、42层模拟布局完整成本门均过；模拟布局不冒充真实trace或整模占比。原CLI v2→组合1+1完整4413.38→4919.82、3639.70→4089.26tok/s（+11.475105%/+12.351567%），这是放置包相对v2的开发增量。历史FIFO候选关闭和认证/INFO凭据设置过程原件保留，不重写为性能轮，不重复测试。

同机retained BF16/原manager→组合原四轮已全部完成，严格只弃各自首轮、所有慢轮保留。有效4K3930.6067±55.9367→5034.83±31.4337，+28.092949%、candidate CV.624326%；16K3162.7867±1.54079→4293.0267±5.27775，+35.735575%、CV.122938%。候选慢有效4999.42/4286.97均高于对照快有效3987.94/3164.49，差1011.48/1122.48tok/s；MeanTTFT4165.32/157162.98→3203.80/115257.03ms，16轮请求/token/TTFT门全过。此为组合性能稳定验证，不把全部组合增量归于量化或allocator；完整精度未知，旧正式表/102分不换。

完整candidate源包candidate-source-frozen.tar.gz双端SHA92579d259797686cba06d29bc886fb4c164d8ca935b271817fb7eb908b754f1d，12声明源/recipe匹配；deploy-src隔离运行，生产BF16/v2/canonical保留。源/规则/最终比较与原件见本包README/final-summary.json；S03开发段仍在唯一正式第5节前。远端两wrapper0、16:54:53两臂shutdown no-live/no升级KILL，监督器rc未留存；外层长期SSH transport255独立记录且最终原件取档成功，后续PID补读连接banner超时、未执行，不扩大为后来所有进程实时状态；无新GPU/105/提交/推送/SNN，按原租期收尾。


**后续立项条件。**

1. 只重开有新因果证据的部署修复，或有独立质量与完整主矩阵成本依据的新量化机制。复用已完成的校准、算术和调用证据，不重复整版v2热轮、旧失败格式或参数扫描。
2. 原始模型目录、serve/benchmark/prefix-cache/采样参数、调度成员、上游vLLM及SDK保持；实质算子改动在FlagGems，插件保存真实数值配置和完整身份。单份W8没有原BF16权重时只允许同量化ABI回退。
3. 新机制先作最低必要算术与独立质量检查，再计入量化、分配、dispatch、workspace、graph、加载及显存代价。局部门通过才做原4K/16K各一次配对，出现明确异常才追加第二轮。
4. 最终双景四轮取后三轮。数值路径变化的新完整105须获用户重新明确授权，不能以局部误差或旧102/105替代。

**参考。** [SmoothQuant](https://arxiv.org/abs/2211.10438)用于配对缩放，[FlagOS-Compressor兼容合同](https://github.com/flagos-ai/FlagOS-Compressor/blob/main/docs/source/reference/compatibility.md)用于产物与运行时接入，[AWQ](https://arxiv.org/abs/2306.00978)及[Marlin](https://github.com/IST-DASLab/marlin)只作方法边界参考，不挪用其它硬件加速倍数。比赛规则继续依据用户提供的官方内容及书面答复，不冒称本轮重新取得网页确认。

**验收。** 本轮证明16K量化有可重复收益，未证明双景稳定及完整精度达标，因此没有新增正式B1。

### 阶段 7：组合、复现与提交

**问题与目标。** 最终组合须在双平台、双场景完成正式门，让数据、实际运行源码和技术报告一致；本文档调整不执行这些任务。

1. 只组合有保留依据的主热点补丁，执行受影响合同的差分、smoke 和必要配对；有交互或归因问题时才做短消融，不要求每个子修改重复三臂全程。

2. 在用户重新明确授权最终 105 后，对最终源码完成完整精度门；未获授权则保留待验状态，不自动评测、不继承旧精度，也不宣称最终验收完成。

3. 在两个官方平台分别运行原 4K/1K 与 16K/1K wrapper，各四轮、丢弃第一轮，保留全部原件并统计 mean、SD、CV、TTFT、请求与 token。收益须超过噪声且满足官方门，评分全程不带 profiler。

4. 从干净 checkout 重放最终补丁，核对实际 import/backend、正式命令、结果与跨平台适用路径/fallback；只为新失败、矛盾或未满足的正式门追加检查。

5. 报告逐项映射“主瓶颈→实质代码→正确性→完整成本与性能→规则依据”。未验收候选不进入提交组合，诊断开关、case 硬编码和无收益路径不作为成果；冻结 down-pack 不再推进，本轮也不删除代码。

**验收。** 两平台、两场景请求与 token 完整、稳定收益超过噪声，获授权的新完整精度及 TTFT 门通过，干净复现与跨平台兼容性成立，报告精确对应最终代码。尚缺任一门就保留对应未完成状态，不承诺固定倍数收益。
