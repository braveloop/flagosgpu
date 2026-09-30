# vllm-plugin-FL 概览

vllm-plugin-FL 是 [vLLM](https://github.com/vllm-project/vllm) 推理/服务框架的插件，基于 FlagOS 的统一多芯片后端构建——包括统一算子库 [FlagGems](https://github.com/flagos-ai/FlagGems) 和统一通信库 [FlagCX](https://github.com/flagos-ai/FlagCX)。它扩展了 vLLM 在不同硬件环境中的能力和性能。在不改变 vLLM 原有接口或使用方式的情况下，同一命令即可在不同芯片上运行模型推理/服务。


```{toctree}
:maxdepth: 2

features.md
operator-dispatch-mechanism.md

```
