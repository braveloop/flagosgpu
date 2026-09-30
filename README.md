# FlagOS MiniCPM5-2B 优化工程

本仓库记录 FlagOS 第二季 MiniCPM5-2B 算子优化比赛的静态分析、分阶段实验计划、可复现学习文档与已完成证据。项目目标是在不改变官方工作负载和模型语义的前提下，对 BI-V150 与 C500 平台进行可验证、可回滚的性能优化。

## 仓库结构

- `FlagGems/`：FlagGems 子模块，用于 Triton/算子路径开发。
- `vllm-plugin-FL/`：FlagOS vLLM 集成子模块，用于框架与平台集成开发。
- `vllm/`：vLLM 上游只读参考子模块。
- `docs/`：中文学习索引与阶段性可复现记录。
- `evidence/`：按不可变实验 ID 组织的已完成证据。
- `official-materials/`：官方资料快照、清单与校验和。
- `skills/flagos-minicpm-competition/`：本项目的执行约束和资料路由。

## 获取源码

```bash
git clone --recurse-submodules git@github.com:braveloop/flagosgpu.git
cd flagosgpu
git submodule update --init --recursive
```

三个源码目录以 Git 子模块管理；根仓库只记录已审核的子模块提交指针，不复制它们的内部 Git 历史。

## 开发与验证边界

本地工作区只用于源码和文档编辑、静态审查与 Git 同步；不在本机安装项目依赖、编译、导入项目模块或运行模型、测试、性能分析和基准。运行时诊断必须在对应的目标算力环境中完成；正式准确率、基线与提交证据只以组织者认可的 64 GB BI-V150/C500 环境为准。

开始工作前请先阅读 [`skills/flagos-minicpm-competition/SKILL.md`](skills/flagos-minicpm-competition/SKILL.md) 与 [`DETAILED_OPTIMIZATION_PLAN.md`](DETAILED_OPTIMIZATION_PLAN.md)，学习记录从 [`docs/S00-学习索引.md`](docs/S00-学习索引.md) 开始。

## 隐私与敏感信息

仓库不收录队员联系方式、凭据、本地环境文件、模型权重或生成缓存。包含个人信息的算力申请材料仅保留在本地，不进入公开 Git 历史。
