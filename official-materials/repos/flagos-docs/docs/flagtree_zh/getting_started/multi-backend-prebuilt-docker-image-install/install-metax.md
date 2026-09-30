[[英文版](./install_metax.md)|中文版]

## 💫 沐曦股份（MetaX）[metax](/third_party/metax/)

- 基于 Triton 3.0，x64
- 适用于 C550

### 1. 构建与运行环境

#### 1.1 使用预装镜像（C550）

如果使用此预装镜像，则无需执行后续步骤 1.x。
如果网络连接可用，也无需执行后续步骤 1.x，因为构建过程中会自动获取依赖项。

```shell
# 方案 A：docker pull（28.1GB）
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-metax-py312-torch2.8.0-vllm0.15.0-metax3.5.3.x-ubuntu22.04:202604-0.5.1
docker pull ${IMAGE}
# 方案 B：docker load（8.1GB）
IMAGE=flagtree-metax-py312-torch2.8.0-vllm0.15.0-metax3.5.3.x-ubuntu22.04:202604-0.5.1
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-metax-py312-torch2.8.0-vllm0.15.0-metax3.5.3.x-ubuntu22.04.202604-0.5.1.tar.gz
docker load -i flagtree-metax-py312-torch2.8.0-vllm0.15.0-metax3.5.3.x-ubuntu22.04.202604-0.5.1.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --net=host --uts=host --ipc=host --privileged=true \
    --group-add video \
    --shm-size 100gb --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device=/dev/dri --device=/dev/mxcd \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖项

```shell
mkdir -p ~/.flagtree/metax; cd ~/.flagtree/metax
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/metaxTritonPlugin-cpython3.12-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-linux-x86_64_v0.5.0.tar.gz
tar zxvf metaxTritonPlugin-cpython3.12-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-linux-x86_64_v0.5.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/ext_maca_mathlib_bc_v0.5.0.tar.gz
tar zxvf ext_maca_mathlib_bc_v0.5.0.tar.gz
# 注意：请联系沐曦供应商获取 maca-llvm-metax20250708.521-x86_64.tar.xz
tar xvf maca-llvm-metax20250708.521-x86_64.tar.xz
```

#### 1.3 手动下载 Triton 依赖项

Triton 依赖项已在预装镜像中下载并安装完毕。
如果不需要从源码构建 FlagTree 或 Triton，则无需下载 Triton 依赖项。

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# 适用于 Triton 3.1（x64）
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.1.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.1.x-linux-x64.tar.gz
```

执行上述脚本后，原有的 ~/.triton 目录将被重命名，并创建一个新的 ~/.triton 目录用于存放预下载的包。
请注意，脚本执行过程中会提示手动确认。

### 2. 安装命令

#### 2.1 免源码安装

```shell
# 注意：请先安装 PyTorch，再执行以下命令
python3 -m pip uninstall -y triton  # 重复执行该命令直到完全卸载
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.12 -m pip install flagtree===0.5.1+metax3.0 $RES
```

预装镜像中已安装 `flagtree`，可通过以下命令检查：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
export FLAGTREE_BACKEND=metax
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试与验证

参考 [MetaX 后端测试](/.github/workflows/metax-build-and-test.yml)
