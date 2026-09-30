[[中文版](./install_cn.md)|English]

## 💫 NVIDIA & AMD [nvidia](/third_party/nvidia/) & [amd](/third_party/amd/)

- 基于 Triton 3.1/3.2/3.3/3.4/3.5/3.6，x64

### 1. 构建和运行环境

#### 1.1 使用镜像（适用于 Triton 3.6）

如果您的网络连接可用，则无需执行后续的步骤 1.x，因为依赖项将在构建过程中自动获取。

```shell
# 方案 A：docker pull（35.5GB）
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-py312-torch2.8.0a0_5228986c39.nv25.05-ubuntu24.04:202605-3.6-base
docker pull ${IMAGE}
# 方案 B：docker load（16GB）
IMAGE=flagtree-py312-torch2.8.0a0_5228986c39.nv25.05-ubuntu24.04:202605-3.6-base
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-py312-torch2.8.0a0_5228986c39.nv25.05-ubuntu24.04.202605-3.6-base.tar.gz
docker load -i flagtree-py312-torch2.8.0a0_5228986c39.nv25.05-ubuntu24.04.202605-3.6-base.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --net=host --uts=host --ipc=host --privileged \
    --ulimit stack=67108864 --ulimit memlock=-1 \
    --security-opt seccomp=unconfined \
    --gpus=all \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 LLVM

```shell
cd ${YOUR_LLVM_DOWNLOAD_DIR}
# 适用于 Triton 3.1
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-10dc3a8e-ubuntu-x64.tar.gz
tar zxvf llvm-10dc3a8e-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-10dc3a8e-ubuntu-x64
# 适用于 Triton 3.2
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-86b69c31-ubuntu-x64.tar.gz
tar zxvf llvm-86b69c31-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-86b69c31-ubuntu-x64
# 适用于 Triton 3.3
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-a66376b0-ubuntu-x64.tar.gz
tar zxvf llvm-a66376b0-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-a66376b0-ubuntu-x64
# 适用于 Triton 3.4
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-8957e64a-ubuntu-x64.tar.gz
tar zxvf llvm-8957e64a-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-8957e64a-ubuntu-x64
# 适用于 Triton 3.5
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-7d5de303-ubuntu-x64.tar.gz
tar zxvf llvm-7d5de303-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-7d5de303-ubuntu-x64
# 适用于 Triton 3.6（方案 A）
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-f6ded0be-ubuntu-x64.tar.gz
tar zxvf llvm-f6ded0be-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-f6ded0be-ubuntu-x64
# 适用于所有版本
export LLVM_INCLUDE_DIRS=$LLVM_SYSPATH/include
export LLVM_LIBRARY_DIR=$LLVM_SYSPATH/lib
```

```shell
# 适用于 Triton 3.6（方案 B，推荐）
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.12 -m pip install mlir $RES
```

#### 1.3 手动下载 Triton 依赖项

Triton 依赖项已在镜像中下载并安装。
如果您不需要从源码构建 FlagTree 或 Triton，则无需下载 Triton 依赖项。

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# 适用于 Triton 3.1（x64）
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.1.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.1.x-linux-x64.tar.gz
# 适用于 Triton 3.2（x64）
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.2.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.2.x-linux-x64.tar.gz
# 适用于 Triton 3.2（aarch64）
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.2.x-linux-aarch64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.2.x-linux-aarch64.tar.gz
# 适用于 Triton 3.3（x64）
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.3.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.3.x-linux-x64.tar.gz
# 适用于 Triton 3.4（x64）
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.4.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.4.x-linux-x64.tar.gz
# 适用于 Triton 3.5（x64）
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.5.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.5.x-linux-x64.tar.gz
# 适用于 Triton 3.6（x64）
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.6.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.6.x-linux-x64.tar.gz
```

执行上述脚本后，原始的 ~/.triton 目录将被重命名，并创建一个新的 ~/.triton 目录来存储预下载的包。
请注意，脚本在执行过程中会提示手动确认。

### 2. 安装命令

#### 2.1 免源码安装

```shell
# 注意：首先安装 PyTorch，然后执行以下命令
python3 -m pip uninstall -y triton  # 重复此命令直到完全卸载
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.12 -m pip install flagtree===0.6.0rc1 $RES
```

安装 `flagtree` 后，您可以通过以下命令检查：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
apt update; apt install zlib1g zlib1g-dev libxml2 libxml2-dev nlohmann-json3-dev
cd ${YOUR_CODE_DIR}/FlagTree
python3 -m pip install -r python/requirements.txt
cd python  # 对于 Triton 3.1、3.2、3.3，需要进入 python 目录进行构建
git checkout main                                   # 适用于 Triton 3.1
git checkout -b triton_v3.2.x origin/triton_v3.2.x  # 适用于 Triton 3.2
git checkout -b triton_v3.3.x origin/triton_v3.3.x  # 适用于 Triton 3.3
git checkout -b triton_v3.4.x origin/triton_v3.4.x  # 适用于 Triton 3.4
git checkout -b triton_v3.5.x origin/triton_v3.5.x  # 适用于 Triton 3.5
git checkout -b triton_v3.6.x origin/triton_v3.6.x  # 适用于 Triton 3.6
unset FLAGTREE_BACKEND
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
# 如果之后需要构建其他后端，应清除 LLVM 相关的环境变量
unset LLVM_SYSPATH LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR
```

### 3. 测试和验证

参考 [nvidia 后端测试](https://github.com/flagos-ai/FlagTree/blob/triton_v3.6.x/.github/workflows/hopper-build-and-test.yml)

## 常见问题

#### 问：安装后运行程序报错：找不到 GLIBC 或 GLIBCXX 版本

答：检查您环境中 libc.so.6 和 libstdc++.so.6.0.30 支持的 GLIBC / GLIBCXX 版本：

```shell
strings /lib/x86_64-linux-gnu/libc.so.6 |grep GLIBC
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 | grep GLIBCXX
```

如果所需的 GLIBC / GLIBCXX 版本已受支持，您也可以尝试：

```shell
export LD_PRELOAD="/lib/x86_64-linux-gnu/libc.so.6"  # 如果找不到 GLIBC
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30"  # 如果找不到 GLIBCXX
export LD_PRELOAD="/lib/x86_64-linux-gnu/libc.so.6 \
    /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30"  # 如果 GLIBC 和 GLIBCXX 都找不到
```
