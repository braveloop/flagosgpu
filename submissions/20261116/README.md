# bravegpuwinner 团队报告审阅

本分支固定当前两仓候选源码，供组员准备报告；已经获得自有 fork 共享授权。它不表示官方社区 PR、正式赛事 ZIP、当前整理树测试或正式验收已完成。

从[候选复现说明](../../vllm-plugin-FL/docs/competition/bravegpuwinner-s2.md)阅读机制、代码映射、recipe 来源、厂商 INT8 GEMM 归属和目标机验证步骤；既有实现正文从[学习索引](../../docs/S00-学习索引.md)进入。[release-index.json](release-index.json)记录两仓固定 SHA、BF16 参考分支及原实验身份，[static-check-report.json](static-check-report.json)仅记录本机静态检查。

| 结果身份 | 4K/1K tok/s | 16K/1K tok/s | 精度和适用边界 |
| --- | ---: | ---: | --- |
| 历史 BI-V150 BF16，2026-10-19 Q32 | 3864.10 | 3164.70 | 102/105；不继承到量化候选 |
| 历史 MetaX BF16 租赁预验证 | 7795.68 | 8551.02 | 102/105；不代表组委会官方资格或新整理树回归 |
| 2026-10-29 BI 冻结量化+extent 历史 | 5034.83 | 4293.03 | 原双景各四轮去首后三的稳定吞吐；完整模型精度未验证 |
| 当前共享整理树 | 未运行 | 未运行 | 正确性、稳定吞吐、跨平台回归与完整精度待完成 |

10-29 相对同机 BF16 的吞吐提升为 28.09%/35.74%，候选 CV 为 0.624%/0.123%；相对官方基线的 2.48/4.69 倍仅是吞吐比值，不是官方综合分。完整历史数值摘要见[候选历史结果](../../vllm-plugin-FL/submission/historical-results.json)。

本轮没有目标服务器，也未运行项目、构建 wheel、执行正确性测试或 105 题。`submitted_tree_runtime_verified=false`、`submitted_tree_stable_throughput_verified=false`、`model_accuracy_verified=false`、`full105_run=false`。下一步先在目标机核对固定版本/实际导入与激活，复用现有正确性及 smoke，通过后再执行原双景四轮稳定吞吐；105 题仍需单独授权。官方提交继续处于 `blocked_pending_target_runtime`。

报告请按原实验和本树区分性能、精度与验证状态；原始 evidence、日志、私人申请材料和凭据不随本次共享追加公开。根仓库旧历史资料保持原记录，当前源码与本说明的固定身份以本分支的 submodule 指针为准。
