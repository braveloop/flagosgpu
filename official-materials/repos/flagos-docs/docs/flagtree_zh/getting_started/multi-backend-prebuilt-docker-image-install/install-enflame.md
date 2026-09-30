[[英文版](./install_enflame.md)|中文版]

## 💫 燧原（Enflame）[enflame](https://github.com/flagos-ai/FlagTree/tree/triton_v3.6.x/third_party/enflame/)（Triton 3.6）

- 基于 Triton 3.6，x64
- 适用于 GCU300、GCU400（L300/L600）

### 1. 构建与运行环境

#### 1.1 使用镜像（Triton 3.6，GCU300/GCU400）

如果网络连接可用，则无需执行后续步骤 1.x，因为构建过程中会自动获取依赖项。

```shell
# 方案 A：docker pull（18.9GB）
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-enflame3.6-py312-torch2.10.0-ubuntu24.04:202605-1.9.7-base
docker pull ${IMAGE}
# 方案 B：docker load（4.0GB）
IMAGE=flagtree-enflame3.6-py312-torch2.10.0-ubuntu24.04:202605-1.9.7-base
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-enflame3.6-py312-torch2.10.0-ubuntu24.04.202605-1.9.7-base.tar.gz
docker load -i flagtree-enflame3.6-py312-torch2.10.0-ubuntu24.04.202605-1.9.7-base.tar.gz
```

```shell
cat /sys/module/enflame/version
    # 如果版本 < 1.9.10，请终止使用 GCU 的进程，并在宿主机上执行以下命令：
    # wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/TopsRider_Triton_gcu-3.6.0_1.0.20260521.cc.1.9.10_deb_amd64.run  # 3.7GB
    # bash TopsRider_Triton_gcu-3.6.0_1.0.20260521.cc.1.9.10_deb_amd64.run --driver -y
efsmi
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --privileged \
    -v /etc/localtime:/etc/localtime:ro \
    -v /home:/home \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖项

```shell
mkdir -p ~/.flagtree/enflame; cd ~/.flagtree/enflame
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm23-fc83c68-gcc9-x64_v0.4.0.tar.gz
tar zxvf enflame-llvm23-fc83c68-gcc9-x64_v0.4.0.tar.gz
```

#### 1.3 手动下载 Triton 依赖项

Triton 依赖项已在镜像中下载并安装完毕。
如果不需要从源码构建 FlagTree 或 Triton，则无需下载 Triton 依赖项。

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# 适用于 Triton 3.6（x64）
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.6.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.6.x-linux-x64.tar.gz
```

执行上述脚本后，原有的 ~/.triton 目录将被重命名，并创建一个新的 ~/.triton 目录用于存放预下载的包。
请注意，脚本执行过程中会提示手动确认。

### 2. 安装命令

#### 2.1 免源码安装

```shell
# 注意：请先安装 PyTorch，再执行以下命令
python3 -m pip uninstall -y triton --break-system-packages  # 重复执行该命令直到完全卸载
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.12 -m pip install flagtree===0.6.0rc1+enflame3.6 --break-system-packages $RES
```

安装 `flagtree` 后，可通过以下命令检查：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
cd ${YOUR_CODE_DIR}/FlagTree
git checkout -b triton_v3.6.x origin/triton_v3.6.x
export FLAGTREE_BACKEND=enflame
MAX_JOBS=8 python3 -m pip install . --no-build-isolation -v --break-system-packages
```

### 3. 测试与验证

参考 [Enflame 3.6 后端测试](https://github.com/flagos-ai/FlagTree/blob/triton_v3.6.x/.github/workflows/enflame-gcu400-build-and-test.yml)

---

## 💫 燧原（Enflame）[enflame](https://github.com/flagos-ai/FlagTree/tree/triton_v3.5.x/third_party/enflame/)（Triton 3.5）

- 基于 Triton 3.5，x64
- 适用于 GCU300、GCU400（L300/L600）

### 1. 构建与运行环境

#### 1.1 使用镜像（Triton 3.5，GCU300/GCU400）

如果网络连接可用，则无需执行后续步骤 1.x，因为构建过程中会自动获取依赖项。

```shell
# 方案 A：docker pull（13.3GB）
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-enflame3.5-py312-torch2.9.1-ubuntu24.04:202603
docker pull ${IMAGE}
# 方案 B：docker load（2.8GB）
IMAGE=flagtree-enflame3.5-py312-torch2.9.1-ubuntu24.04:202603
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-enflame3.5-py312-torch2.9.1-ubuntu24.04.202603.tar.gz
docker load -i flagtree-enflame3.5-py312-torch2.9.1-ubuntu24.04.202603.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --privileged \
    -v /etc/localtime:/etc/localtime:ro \
    -v /home:/home \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker cp ${CONTAINER}:/enflame enflame    # 将在当前目录创建 ./enflame 目录
bash enflame/driver/enflame-x86_64-gcc-1.7.2.14-20260302150833.run
efsmi
docker stop ${CONTAINER}
docker start ${CONTAINER}
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖项

```shell
mkdir -p ~/.flagtree/enflame; cd ~/.flagtree/enflame
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm22-189e06b-gcc9-x64_v0.4.0.tar.gz
tar zxvf enflame-llvm22-189e06b-gcc9-x64_v0.4.0.tar.gz
```

#### 1.3 手动下载 Triton 依赖项

Triton 依赖项已在镜像中下载并安装完毕。
如果不需要从源码构建 FlagTree 或 Triton，则无需下载 Triton 依赖项。

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# 适用于 Triton 3.5（x64）
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.5.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.5.x-linux-x64.tar.gz
```

执行上述脚本后，原有的 ~/.triton 目录将被重命名，并创建一个新的 ~/.triton 目录用于存放预下载的包。
请注意，脚本执行过程中会提示手动确认。

### 2. 安装命令

#### 2.1 免源码安装

```shell
# 注意：请先安装 PyTorch，再执行以下命令
python3 -m pip uninstall -y triton --break-system-packages  # 重复执行该命令直到完全卸载
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.12 -m pip install flagtree===0.5.0+enflame3.5 --break-system-packages $RES
```

安装 `flagtree` 后，可通过以下命令检查：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
cd ${YOUR_CODE_DIR}/FlagTree
git checkout -b triton_v3.5.x origin/triton_v3.5.x
export FLAGTREE_BACKEND=enflame
MAX_JOBS=8 python3 -m pip install . --no-build-isolation -v --break-system-packages
```

### 3. 测试与验证

参考 [Enflame 3.5 后端测试](https://github.com/flagos-ai/FlagTree/blob/triton_v3.5.x/.github/workflows/enflame-gcu400-build-and-test.yml)

---

## 💫 燧原（Enflame）[enflame](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/enflame/)（Triton 3.3）

- 基于 Triton 3.3，x64
- 适用于 GCU300

### 1. 构建与运行环境

#### 1.1 使用镜像（Triton 3.3，GCU300）

如果网络连接可用，则无需执行后续步骤 1.x，因为构建过程中会自动获取依赖项。

```shell
# 方案 A：docker pull（12.5GB）
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-enflame3.3-py310-torch2.7.0-ubuntu22.04:202603
docker pull ${IMAGE}
# 方案 B：docker load（5.7GB）
IMAGE=flagtree-enflame3.3-py310-torch2.7.0-ubuntu22.04:202603
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-hcu-py310-torch2.9.0-ubuntu22.04.202603.tar.gz
docker load -i flagtree-hcu-py310-torch2.9.0-ubuntu22.04.202603.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --privileged \
    -v /etc/localtime:/etc/localtime:ro \
    -v /home:/home \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker cp ${CONTAINER}:/enflame enflame    # 将在当前目录创建 ./enflame 目录
bash enflame/driver/enflame-x86_64-gcc-1.6.3.12-20251115104629.run
efsmi
docker stop ${CONTAINER}
docker start ${CONTAINER}
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖项

```shell
mkdir -p ~/.flagtree/enflame; cd ~/.flagtree/enflame
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
tar zxvf enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
```

#### 1.3 手动下载 Triton 依赖项

Triton 依赖项已在镜像中下载并安装完毕。
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
python3.10 -m pip install flagtree===0.4.0+enflame3.3 $RES
```

安装 `flagtree` 后，可通过以下命令检查：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
git checkout -b triton_v3.3.x origin/triton_v3.3.x
export FLAGTREE_BACKEND=enflame
MAX_JOBS=8 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试与验证

参考 [Enflame 3.3 后端测试](https://github.com/flagos-ai/FlagTree/blob/triton_v3.3.x/.github/workflows/enflame-gcu300-3.3-build-and-test.yml)
