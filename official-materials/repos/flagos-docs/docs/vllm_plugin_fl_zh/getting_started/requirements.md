# 要求

## 软件要求

| 要求 | v0.1.0（vLLM 0.13.0） | v0.2.0（vLLM 0.20.0 或 vLLM 0.20.2） | 备注 |
|-------------|----------------------|----------------------|-------|
| Python | 3.10 - 3.13 | 3.10 - 3.13 | 必需 |
| PyTorch | >= 2.7.1 | >= 2.7.1 | 必需 |
| vLLM | 0.13.0 | 0.20.2 | 来自官方发布或分支 |
| FlagGems | >= v5.0.0 | >= v5.0.0 | 算子调度必需 |
| FlagCX | v0.9.0 | v0.9.0 | 可选，用于多芯片通信 |
| FlagTree | 0.4.0 | 0.4.0 | 仅 Ascend NPU |

## 支持的硬件平台

下表汇总了支持的硬件及其验证状态：

| 芯片厂商 | v0.1.0（vLLM 0.13.0） | v0.2.0（vLLM 0.20.0 或 vLLM 0.20.2） | 备注 |
|-------------|----------------------|----------------------|-------|
| NVIDIA | 支持 | 支持 | |
| Ascend | 支持 | — | 需要 FlagTree 和 eager 执行 |
| MetaX | 支持 | — | |
| T-Head | 支持 | — | |
| Iluvatar | 支持 | — | 需要 FlagTree 和 eager 执行 |
| Moore Threads | 支持 | — | |
| Tsingmicro | 合并中 | — | [PR #52](https://github.com/flagos-ai/vllm-plugin-FL/pull/52) |
| Hygon DCU | 支持 | 支持 | v0.2.0 需要 DTK 容器（参见安装指南） |
| Sunrise | 支持 | — | |

## 支持的模型

理论上，如果不涉及不支持的算子，vllm-plugin-FL 可以支持 vLLM 中所有可用的模型。以下模型已经过端到端验证：

| 模型 | 状态 | 示例 |
|-------|--------|---------|
| Qwen3.5-397B-A17B | 支持 | [qwen3_5_offline_inference.py](https://github.com/flagos-ai/vllm-plugin-FL/blob/main/examples/qwen3_5_offline_inference.py) |
| Qwen3-Next-80B-A3B | 支持 | [qwen3_next_offline_inference.py](https://github.com/flagos-ai/vllm-plugin-FL/blob/main/examples/qwen3_next_offline_inference.py) |
| Qwen3-4B | 支持 | [offline_inference.py](https://github.com/flagos-ai/vllm-plugin-FL/blob/main/examples/offline_inference.py) |
| MiniCPM-o 4.5 | 支持 | [examples/minicpm/](https://github.com/flagos-ai/vllm-plugin-FL/tree/main/examples/minicpm) |
| GLM-5 | 支持 | [glm_5_offline_inference.py](https://github.com/flagos-ai/vllm-plugin-FL/blob/main/examples/glm_5_offline_inference.py) |
| Qwen3.5-35B-A3B | 支持 | [qwen3_5_offline_inference.py](https://github.com/flagos-ai/vllm-plugin-FL/blob/main/examples/qwen3_5_offline_inference.py) |
| BAAI/bge-m3 | 支持 | [bge_m3.py](https://github.com/flagos-ai/vllm-plugin-FL/blob/main/vllm_fl/models/bge_m3.py) |
| MiniMax-M2.7 | 支持 | [minimax_m27_offline_inference.py](https://github.com/flagos-ai/vllm-plugin-FL/blob/main/examples/minimax_m27_offline_inference.py) |
| Qwen3.6-35B-A3B | 支持 | [文本 + 图像推理/服务（v0.2.0）](/getting_started/run-inference-task.md#run-a-serving-inference-task) |
| Qwen3.6-27B | 支持 | [文本 + 图像推理/服务（v0.2.0）](/getting_started/run-inference-task.md#run-a-serving-inference-task) |
| Qwen2.5-1.5B | 支持 | [Iluvatar BI-V150 示例](example-qwen2.5-bv150.md) |
