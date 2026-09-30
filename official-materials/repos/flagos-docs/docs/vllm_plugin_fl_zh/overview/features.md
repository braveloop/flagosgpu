# 特性

- **统一多芯片后端**

利用 FlagGems（统一算子库）和 FlagCX（统一通信库）提供芯片无关的推理能力。同一模型可在不同硬件上运行，无需修改代码。

- **灵活的算子调度**

实现了基于优先级的调度系统，在 FlagGems、厂商特定实现和 PyTorch 参考实现之间进行选择。算子可按后端配置，并在失败时自动回退。

- **平台自动检测**

自动检测硬件并加载平台特定配置。支持 NVIDIA GPU、Ascend NPU、T-Head、Iluvatar、MetaX、Moore Threads、Tsingmicro、Hygon DCU 和 Sunrise 芯片。

- **可扩展的厂商后端**

支持内置厂商后端（CUDA、Ascend）、通过 setuptools 入口点的外部插件包，以及基于环境变量的插件模块。
