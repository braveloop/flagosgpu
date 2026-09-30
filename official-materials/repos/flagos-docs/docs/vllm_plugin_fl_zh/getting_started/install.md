# 安装运行推理任务所需的软件

vllm-plugin-FL 可以从源代码安装或通过 Docker 镜像安装。

## 从源代码安装

本节介绍从源代码安装 vllm-plugin-FL 及其依赖项。

1. 从官方版本安装 vllm [v0.20.2](https://github.com/vllm-project/vllm/tree/v0.20.2)（如果已安装正确版本则可选）
2. 安装 vllm-plugin-FL

    2.1 克隆仓库：

    ```{code-block} shell
    git clone https://github.com/flagos-ai/vllm-plugin-FL
    ```

    2.2 安装

    ```{code-block} shell
    cd vllm-plugin-FL
    pip install --no-build-isolation .
    # 或可编辑安装
    pip install --no-build-isolation -e .
    ```

3. 安装 [FlagGems](https://github.com/flagos-ai/FlagGems/blob/master/docs/getting-started.md#quick-installation)

    3.1 安装构建依赖

    ```{code-block} shell
    pip install -U scikit-build-core==0.11 pybind11 ninja cmake
    ```

    3.2 安装 FlagGems

    ```{code-block} shell
    git clone https://github.com/flagos-ai/FlagGems
    cd FlagGems
    pip install --no-build-isolation .
    # 或可编辑安装
    pip install --no-build-isolation -e .
    ```

    ```{note}
    在 Sunrise 平台上，依赖 FlagGems [PR #2949](https://github.com/flagos-ai/FlagGems/pull/2949)。
    在 Hygon 平台上，依赖 FlagGems [PR #3477](https://github.com/flagos-ai/FlagGems/pull/3477)。
    ```

4. （可选）安装 [FlagCX](https://github.com/flagos-ai/FlagCX/blob/main/docs/getting_started.md#build-and-installation)

    4.1 克隆仓库：

    ```{code-block} shell
    git clone https://github.com/flagos-ai/FlagCX.git
    cd FlagCX
    git checkout -b v0.9.0
    git submodule update --init --recursive
    ```

    4.2 使用不同标志构建库以适配不同平台：

    ```{code-block} shell
    make USE_NVIDIA=1
    ```

    4.3 设置环境变量

    ```{code-block} shell
    export FLAGCX_PATH="$PWD"
    ```

    4.4 安装 FlagCX

    ```{code-block} shell
    cd plugin/torch/
    FLAGCX_ADAPTOR=[xxx] pip install . --no-build-isolation
    # 或可编辑安装
    FLAGCX_ADAPTOR=[xxx] pip install -e . --no-build-isolation
    ```

    ```{note}
    [xxx] 应根据当前平台选择，例如 nvidia、ascend 等。
    ```

如果当前环境中有多个插件，可以通过 VLLM_PLUGINS='fl' 指定使用 vllm-plugin-fl。

### 华为 Ascend 额外设置

1. 安装 [FlagTree](https://resource.flagos.net)

    ```{code-block} shell
    RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=https://resource.flagos.net"
    python3 -m pip install flagtree==0.4.0+ascend3.2 $RES
    ```

2. 设置必需的环境变量

    ```{code-block} shell
    export TRITON_ALL_BLOCKS_PARALLEL=1
    ```

3. 启用 eager 执行

    Ascend 需要 eager 执行。在 `LLM` 构造函数中添加 `enforce_eager=True` 或在命令行中传递 `--enforce-eager`。

### （可选）CUDA 额外设置

本节说明如何通过设置环境变量使用 CUDA 运行推理任务。

算子调度环境变量请参见 [环境变量](../dispatch_user_guide/configure-backend-selection.md/#environment-variables)。

#### 使用 CUDA 通信库

本节演示如何通过设置环境变量使用 CUDA 运行推理任务。

```{code-block} shell
unset FLAGCX_PATH
```

#### 使用原生 CUDA 算子

如果您想使用原始的 CUDA 算子，可以设置以下环境变量。

```{code-block} shell
export USE_FLAGGEMS=0
```

## 从 Docker 镜像安装

本节介绍使用预构建的 Docker 镜像运行 vllm-plugin-FL。

### SVT 全栈测试镜像（v0.2.0-rc2）

预构建的 SVT 全栈测试镜像：

| 平台 | 镜像 | 内容 |
|----------|-------|----------|
| NVIDIA GPU | `harbor.baai.ac.cn/flagos21-release/vllm-plugin-fl:v0.2.0-rc2-nvidia-svt` | vllm 0.20.2, FlagGems 5.3.0-rc2.post1, FlagTree 3.6.0, vllm-plugin-FL 0.2.0-rc2.post1, torch 2.11.0+cu130 |
| 海光 DCU | `harbor.baai.ac.cn/flagos21-release/vllm-plugin-fl:v0.2.0-rc2-hygon-svt` | vllm 0.20.0, FlagGems 5.3.0-rc2.post1, FlagTree 0.5.0-rc2.post1+hcu, vllm-plugin-FL 0.2.0-rc2.post1, torch 2.10.0+das |

```bash
# NVIDIA SVT
docker pull harbor.baai.ac.cn/flagos21-release/vllm-plugin-fl:v0.2.0-rc2-nvidia-svt

# 海光 DCU SVT
docker pull harbor.baai.ac.cn/flagos21-release/vllm-plugin-fl:v0.2.0-rc2-hygon-svt
```

### Hygon DCU

适用于 vllm-plugin-FL v0.2.0（vLLM 0.20.0）。

1. 拉取并启动 Hygon DCU Docker 容器：

    ```{code-block} shell
    docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.20.0-ubuntu22.04-dtk26.04-py3.10-MiniCPM-V-4.6

    docker run \
        --name perf \
        --network=host \
        --ipc=host \
        --device=/dev/kfd \
        --device=/dev/mkfd \
        --device=/dev/dri \
        -v /opt/hyhal:/opt/hyhal \
        -v /path/to/models:/models \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -itd harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.20.0-ubuntu22.04-dtk26.04-py3.10-MiniCPM-V-4.6 \
        /bin/bash
    ```

    将 `/path/to/models` 替换为实际的模型存储路径。

2. 在容器内安装 FlagGems：

    ```{code-block} shell
    git clone https://github.com/flagos-ai/FlagGems
    cd FlagGems
    git checkout 2718037d887cd6a3143474da0224648e40c5004f
    pip install --no-build-isolation -e .
    ```

3. 安装 vllm-plugin-FL：

    ```{code-block} shell
    git clone https://github.com/flagos-ai/vllm-plugin-FL
    cd vllm-plugin-FL
    git checkout 48af29e21491700a38020ab031af5d3b90e6795e
    pip install --no-build-isolation -e .
    ```

4. 下载模型：

    ```{code-block} shell
    modelscope download --model Qwen/Qwen3.6-27B --local_dir /models/Qwen3.6-27B
    modelscope download --model Qwen/Qwen3.6-35B-A3B --local_dir /models/Qwen3.6-35B-A3B
    ```

### NVIDIA

适用于 vllm-plugin-FL v0.2.0（vLLM 0.20.2）。

1. 拉取并启动 NVIDIA Docker 容器：

    ```{code-block} shell
    docker pull vllm/vllm-openai:v0.20.0-cu130-ubuntu2404

    docker run -itd \
        --name perf \
        --entrypoint /bin/bash \
        --gpus all \
        --ipc=host \
        --privileged \
        --net host \
        --shm-size 512g \
        -v /path/to/models:/models \
        vllm/vllm-openai:v0.20.0-cu130-ubuntu2404
    ```

    将 `/path/to/models` 替换为实际的模型存储路径。

2. 在容器内安装依赖：

    ```{code-block} shell
    apt-get update
    apt install git
    apt install vim
    pip install -U scikit-build-core==0.11 pybind11 ninja cmake
    ```

3. 安装 vllm：

    ```{code-block} shell
    pip install vllm==0.20.2
    ```

4. 安装 FlagGems：

    ```{code-block} shell
    git clone https://github.com/flagos-ai/FlagGems
    cd FlagGems
    git checkout 1dab11ab1a6671e3132528492d2cc193e78af8f4
    pip install --no-build-isolation .
    ```

5. 安装 vllm-plugin-FL：

    ```{code-block} shell
    git clone https://github.com/flagos-ai/vllm-plugin-FL
    cd vllm-plugin-FL
    git checkout 48af29e21491700a38020ab031af5d3b90e6795e
    pip install --no-build-isolation .
    ```

6. 下载模型：

    ```{code-block} shell
    modelscope download --model Qwen/Qwen3.6-27B --local_dir /models/Qwen3.6-27B
    modelscope download --model Qwen/Qwen3.6-35B-A3B --local_dir /models/Qwen3.6-35B-A3B
    ```

### 华为 Ascend

适用于 vllm-plugin-FL v0.1.0（vLLM 0.13.0）。

1. 拉取并启动 Ascend Docker 容器：

    ```{code-block} shell
    docker pull quay.io/ascend/vllm-ascend:v0.13.0rc1-a3

    docker run \
    --name flagos \
    --network host \
    --ipc=host \
    --privileged \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /path/to/models:/models \
    -itd quay.io/ascend/vllm-ascend:v0.13.0rc1-a3 bash

    docker exec -it flagos bash
    ```

    将 `/path/to/models` 替换为实际的模型存储路径。

2. 在容器内安装 FlagGems：

    ```{code-block} shell
    pip install -U scikit-build-core==0.11 pybind11 ninja cmake
    git clone https://github.com/flagos-ai/FlagGems
    cd FlagGems
    git checkout 6f2585dc9c48d440d856ad75f4aedee66fac365a
    pip install --no-build-isolation -e .
    ```

3. 安装 FlagTree：

    ```{code-block} shell
    RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=https://resource.flagos.net"
    python3 -m pip install flagtree==0.4.0+ascend3.2 $RES
    ```

4. 安装 vllm-plugin-FL：

    ```{code-block} shell
    git clone https://github.com/flagos-ai/vllm-plugin-FL
    cd vllm-plugin-FL
    git checkout ba008211c1c9646e19290e832a7f7775f7d2944f
    pip install --no-build-isolation -e .
    ```

5. 设置环境变量并启动服务：

    ```{code-block} shell
    export VLLM_PLUGINS=fl
    export TRITON_ALL_BLOCKS_PARALLEL=1
    vllm serve --model /models/Qwen3-4B --served-model-name qwen --enforce-eager
    ```

    ```{note}
    Ascend 需要 eager 执行。在 `LLM` 构造函数中添加 `enforce_eager=True` 或在命令行中传递 `--enforce-eager`。
    ```

### Iluvatar BI-V150

适用于 vllm-plugin-FL v0.1.0（vLLM 0.13.0）。

1. 加载并启动 Corex Docker 容器：

    ```{code-block} shell
    docker load -i /mnt/share/images/corex.4.4.0.release.0211.vllm.013.flagos.tar

    docker run --shm-size="32g" -itd \
      -v /dev:/dev -v /usr/src/:/usr/src \
      -v /lib/modules/:/lib/modules \
      -v /mnt/share/user_homes/:/mnt/share/user_homes/ \
      --privileged --cap-add=ALL --pid=host --net=host \
      --name flagos_v2 corex:4.4.0.release.0211.vllm.013.flagos /bin/bash
    ```

2. 在容器内安装 FlagGems：

    ```{code-block} shell
    pip install -U scikit-build-core==0.11 pybind11 ninja cmake
    git clone https://github.com/flagos-ai/FlagGems
    cd FlagGems
    git checkout 6f2585dc9c48d440d856ad75f4aedee66fac365a
    pip install --no-build-isolation -e .
    cd ../
    ```

3. 安装 vllm-plugin-FL：

    ```{code-block} shell
    git clone https://github.com/flagos-ai/vllm-plugin-FL.git
    cd vllm-plugin-FL
    git checkout f11a0f4707aecae245ec81289329b208ede5b06d
    pip install --no-build-isolation -e . --no-deps
    cd ../
    ```

4. 启动服务：

    ```{code-block} shell
    export VLLM_PLUGINS=fl
    export VLLM_ENGINE_ITERATION_TIMEOUT_S=36000
    export VLLM_RPC_TIMEOUT=36000000
    vllm serve /mnt/share/user_homes/zyp/Qwen3-4B/ --served-model-name qwen --enforce-eager
    ```

    ```{note}
    首次启动大约需要 15 分钟。后续启动不到 2 分钟。
    ```

以上步骤提供了在 BV150 上运行 Qwen3-4B 的最小设置。如果您需要完整的 FlagOS 技术栈（含 FlagTree、FlagGems 使用模式、环境验证和故障排除），或者您是首次设置 BV150，请参见完整的端到端指南：[Qwen2.5-1.5B on Iluvatar BI-V150](example-qwen2.5-bv150.md)。

## （可选）调度算子

如果需要，您也可以调度算子。

概念相关信息请参见 [vllm-plugin-FL 概览](../overview/overview.md)。
配置相关信息请参见 [算子调度用户指南](../dispatch_user_guide/dispatch-user-guide.md)。

安装和可选的算子调度配置完成后，您可以继续[运行推理任务](run-inference-task.md)。
