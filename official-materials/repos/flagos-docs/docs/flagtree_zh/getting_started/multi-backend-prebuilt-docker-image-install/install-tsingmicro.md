[[英文版](./install_tsingmicro.md)|中文版]

## 💫 清微智能（Tsingmicro）[tsingmicro](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/tsingmicro/)

- 基于 Triton 3.3，x64
- 适用于 TX81

### 1. 构建与运行环境

#### 1.1 使用预装镜像（TX81）

如果使用此预装镜像，则无需执行后续步骤 1.x。
如果网络连接可用，也无需执行后续步骤 1.x，因为构建过程中会自动获取依赖项。

```shell
# 方案 A：docker pull（105GB）
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-tsingmicro-py310-torch2.7.0-ubuntu20.04:202603
docker pull ${IMAGE}
# 方案 B：docker load（44GB）
IMAGE=flagtree-tsingmicro-py310-torch2.7.0-ubuntu20.04:202603
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-tsingmicro-py310-torch2.7.0-ubuntu20.04.202603.tar.gz
docker load -i flagtree-tsingmicro-py310-torch2.7.0-ubuntu20.04.202603.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --network=host --ipc=host --privileged \
    --security-opt seccomp=unconfined \
    -v /dev:/dev -v /lib/modules:/lib/modules -v /sys:/sys \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖项

```shell
mkdir -p ~/.flagtree/tsingmicro; cd ~/.flagtree/tsingmicro
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.10-x64_v0.4.0.tar.gz
tar zxvf tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.10-x64_v0.4.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tx8_depends_dev_20260309_173649_v0.5.0.tar.gz
tar zxvf tx8_depends_dev_20260309_173649_v0.5.0.tar.gz
```

#### 1.3 手动下载 Triton 依赖项

Triton 依赖项已在预装镜像中下载并安装完毕。
如果不需要从源码构建 FlagTree 或 Triton，则无需下载 Triton 依赖项。

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# 适用于 Triton 3.3（x64）
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.3.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.3.x-linux-x64.tar.gz
```

执行上述脚本后，原有的 ~/.triton 目录将被重命名，并创建一个新的 ~/.triton 目录用于存放预下载的包。
请注意，脚本执行过程中会提示手动确认。

### 2. 安装命令

#### 2.1 免源码安装

```shell
# 注意：请先安装 PyTorch，再执行以下命令
python3 -m pip uninstall -y triton  # 重复执行该命令直到完全卸载
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.10 -m pip install flagtree===0.5.0+tsingmicro3.3 $RES
```

预装镜像中已安装 `flagtree`，可通过以下命令检查：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

构建前，需要执行 `source ~/env_setup.sh`。该脚本内容如下：

```bash
export TX8_DEPS_ROOT=~/.flagtree/tsingmicro/tx8_deps
export LLVM_SYSPATH=~/.flagtree/tsingmicro/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.10-x64
export LLVM_BINARY_DIR=${LLVM_SYSPATH}/bin
export PYTHONPATH=${LLVM_SYSPATH}/python_packages/mlir_core:$PYTHONPATH
export LD_LIBRARY_PATH=$TX8_DEPS_ROOT/lib:$LD_LIBRARY_PATH
```

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
git checkout -b triton_v3.3.x origin/triton_v3.3.x
export FLAGTREE_BACKEND=tsingmicro
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试与验证

测试前，需要执行 `source ~/env_setup.sh`。该脚本内容如上所示。

参考 [Tsingmicro 后端测试](https://github.com/flagos-ai/FlagTree/blob/triton_v3.3.x/.github/workflows/tsingmicro-build-and-test.yml)
