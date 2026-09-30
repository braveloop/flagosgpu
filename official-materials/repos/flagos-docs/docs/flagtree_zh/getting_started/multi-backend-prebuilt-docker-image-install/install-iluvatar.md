[[英文版](./install_iluvatar.md)|中文版]

## 💫 天数智芯（ILUVATAR）[iluvatar](https://github.com/flagos-ai/FlagTree/tree/main/third_party/iluvatar/)

- 基于 Triton 3.1，x64
- 适用于 MR-V100、BI-V150

### 1. 构建与运行环境

#### 1.1 使用镜像（BI-V150）

如果网络连接可用，则无需执行后续步骤 1.x，因为构建过程中会自动获取依赖项。

```shell
modinfo iluvatar | grep "description"  # f65d8ac7
# 方案 A：docker pull（17.9GB）
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-iluvatar-py312-torch2.7.1-4.4.0release_f65d8ac7-ubuntu24.04:202604-base
docker pull ${IMAGE}
# 方案 B：docker load（5.1GB）
IMAGE=flagtree-iluvatar-py312-torch2.7.1-4.4.0release_f65d8ac7-ubuntu24.04:202604-base
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-iluvatar-py312-torch2.7.1-4.4.0release_f65d8ac7-ubuntu24.04.202604-base.tar.gz
docker load -i flagtree-iluvatar-py312-torch2.7.1-4.4.0release_f65d8ac7-ubuntu24.04.202604-base.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --net=host --pid=host --privileged \
    --cap-add=ALL \
    --security-opt seccomp=unconfined \
    -v /lib/modules:/lib/modules -v /dev:/dev  \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data1:/data1 -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE}
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖项

```shell
mkdir -p ~/.flagtree/iluvatar; cd ~/.flagtree/iluvatar
# llvm
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatar-llvm18-x86_64_v0.5.0.tar.gz
tar zxvf iluvatar-llvm18-x86_64_v0.5.0.tar.gz

# iluvatarTritonPlugin
ABI=$(echo | g++ -dM -E -x c++ - | awk '/__GXX_ABI_VERSION/{print $3}')
case "${ABI}" in  # 适用于镜像中的 python3.12
  1018) PLUGIN_TGZ=iluvatarTritonPlugin-cpython3.12-glibc2.39-glibcxx3.4.33-cxxabi1.3.15-ubuntu-x86_64_v0.5.0.tar.gz ;;
  *) echo "不支持的 __GXX_ABI_VERSION=${ABI}"; exit 1 ;;
esac
case "${ABI}" in  # 适用于 python3.10，不适用于此镜像
  1013) PLUGIN_TGZ=iluvatarTritonPlugin-cpython3.10-glibc2.17-glibcxx3.4.19-cxxabi1.3.12-linux-x86_64_v0.5.0.tar.gz ;;
  1016) PLUGIN_TGZ=iluvatarTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x86_64_v0.5.0.tar.gz ;;
  1018) PLUGIN_TGZ=iluvatarTritonPlugin-cpython3.10-glibc2.39-glibcxx3.4.33-cxxabi1.3.15-ubuntu-x86_64_v0.5.0.tar.gz ;;
  *) echo "不支持的 __GXX_ABI_VERSION=${ABI}"; exit 1 ;;
esac
wget "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/${PLUGIN_TGZ}"
tar zxvf "${PLUGIN_TGZ}"
```

#### 1.3 手动下载 Triton 依赖项

Triton 依赖项已在镜像中下载并安装完毕。
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
python3.12 -m pip install flagtree===0.5.1+iluvatar3.1 $RES
```

安装 `flagtree` 后，可通过以下命令检查：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
export FLAGTREE_BACKEND=iluvatar
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试与验证

参考 [Iluvatar 后端测试](/.github/workflows/iluvatar-build-and-test.yml)
