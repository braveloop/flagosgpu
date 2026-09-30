[[英文版](./install_ascend.md)|中文版]

## 💫 华为昇腾（Huawei Ascend）[ascend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.2.x/third_party/ascend)

- 基于 Triton 3.2，aarch64
- 适用于 910B、910C

### 1. 构建与运行环境

#### 1.1 使用预装镜像（910C）

此预装镜像是基于 [Dockerfile-ubuntu22.04-python3.11-ascend](/dockerfiles/Dockerfile-ubuntu22.04-python3.11-ascend) 执行后续步骤 1.x 并安装 FlagTree 后创建的。
如果使用此预装镜像，则无需为 910C 执行后续步骤 1.x，对于 910B 只需执行步骤 1.2。
如果网络连接可用，也无需执行后续步骤 1.x，因为构建过程中会自动获取依赖项。

```shell
# 方案 A：docker pull（26.2GB）
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-ascend-910c-py311-torch2.6.0-cann8.5.0-ubuntu22.04-aarch64:202603
docker pull ${IMAGE}
# 方案 B：docker load（8.8GB）
IMAGE=flagtree-ascend-910c-py311-torch2.6.0-cann8.5.0-ubuntu22.04-aarch64:202603
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-ascend-910c-py311-torch2.6.0-cann8.5.0-ubuntu22.04-aarch64.202603.tar.gz
docker load -i flagtree-ascend-910c-py311-torch2.6.0-cann8.5.0-ubuntu22.04-aarch64.202603.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit -u 0 --user=root \
    --network=host --pid=host --ipc=host --privileged \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 安装 CANN

- 在 https://www.hiascend.com/developer/download/community/result?module=cann 注册账号，并下载对应平台的 `cann-toolkit` 和 `cann-ops`

```shell
# cann-toolkit
chmod +x Ascend-cann-toolkit_8.5.0_linux-aarch64.run
./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install
# cann-ops（适用于 910B / A2）
chmod +x Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
# cann-ops（适用于 910C / A3）
chmod +x Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
```

#### 1.3 手动下载 FlagTree 依赖项

```shell
mkdir -p ~/.flagtree/ascend; cd ~/.flagtree/ascend
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz
tar zxvf llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz
```

```shell
cd ${YOUR_CODE_DIR}/FlagTree/third_party
git clone https://github.com/flagos-ai/flir.git
cd flir
git checkout -b triton_v3.3.x origin/triton_v3.3.x  # 适用于 flagtree triton_v3.2.x triton_v3.3.x
```

#### 1.4 手动下载 Triton 依赖项

Triton 依赖项已在预装镜像中下载并安装完毕。
如果不需要从源码构建 FlagTree 或 Triton，则无需下载 Triton 依赖项。

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# 适用于 Triton 3.2（aarch64）
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.2.x-linux-aarch64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.2.x-linux-aarch64.tar.gz
```

执行上述脚本后，原有的 ~/.triton 目录将被重命名，并创建一个新的 ~/.triton 目录用于存放预下载的包。
请注意，脚本执行过程中会提示手动确认。

### 2. 安装命令

#### 2.1 免源码安装

```shell
# 注意：请先安装 PyTorch，再执行以下命令
python3 -m pip uninstall -y triton  # 重复执行该命令直到完全卸载
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.11 -m pip install flagtree===0.6.0rc1+ascend3.2 $RES
```

预装镜像中已安装 `flagtree`，可通过以下命令检查：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
git checkout -b triton_v3.2.x origin/triton_v3.2.x
export FLAGTREE_BACKEND=ascend
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试与验证

参考 [Ascend 后端测试](https://github.com/flagos-ai/FlagTree/blob/triton_v3.2.x/.github/workflows/ascend-build-and-test.yml)
