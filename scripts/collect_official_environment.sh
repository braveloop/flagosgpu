#!/usr/bin/env bash

# 仅在赛事分配或组委会书面认可的 BI-V150/C500 环境运行。
# 本脚本只读取环境并归档证据；不会安装依赖、拉取代码、启动服务或运行工作负载。

set -u
set -o pipefail

usage() {
  echo "用法: $0 --platform BI-V150|C500 --output ABS_PATH" >&2
}

die() {
  echo "错误: $*" >&2
  exit 2
}

platform=""
output_dir=""
while (($# > 0)); do
  case "$1" in
    --platform)
      (($# >= 2)) || die "--platform 缺少参数"
      platform="$2"
      shift 2
      ;;
    --output)
      (($# >= 2)) || die "--output 缺少参数"
      output_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "未知参数: $1"
      ;;
  esac
done

case "$platform" in
  BI-V150|C500) ;;
  *) die "--platform 只能是 BI-V150 或 C500" ;;
esac

[[ "$output_dir" = /* ]] || die "--output 必须是绝对路径"
[[ ! -e "$output_dir" ]] || die "输出路径已存在，拒绝覆盖: $output_dir"
[[ ! -L "$output_dir" ]] || die "输出路径是符号链接，拒绝使用: $output_dir"
case "$output_dir" in
  /home/brave/flagos|/home/brave/flagos/*)
    die "拒绝把官方环境证据写入本机项目路径"
    ;;
esac

current_dir="$(pwd -P)"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
script_root="$(cd -- "$script_dir/.." && pwd -P)"
case "$current_dir" in
  /home/brave/flagos|/home/brave/flagos/*)
    die "检测到本机项目工作目录；本脚本只能在官方平台运行"
    ;;
esac
case "$script_root" in
  /home/brave/flagos|/home/brave/flagos/*)
    die "检测到本机脚本路径；请把脚本复制到官方平台后再运行"
    ;;
esac

umask 077
mkdir -p -- "$output_dir/stdout" "$output_dir/stderr" \
  || die "无法创建输出目录: $output_dir"
manifest="$output_dir/manifest.tsv"
printf 'label\texit_code\tstdout\tstderr\tcommand\n' >"$manifest"

record_missing() {
  local label="$1"
  local reason="$2"
  printf 'MISSING: %s\n' "$reason" >"$output_dir/stdout/$label.txt"
  : >"$output_dir/stderr/$label.txt"
  printf '%s\tNA\tstdout/%s.txt\tstderr/%s.txt\tMISSING: %s\n' \
    "$label" "$label" "$label" "$reason" >>"$manifest"
}

run_capture() {
  local label="$1"
  shift
  local stdout_file="$output_dir/stdout/$label.txt"
  local stderr_file="$output_dir/stderr/$label.txt"
  local rendered=""
  local arg
  for arg in "$@"; do
    printf -v rendered '%s%q ' "$rendered" "$arg"
  done
  "$@" >"$stdout_file" 2>"$stderr_file"
  local rc=$?
  printf '%s\t%s\tstdout/%s.txt\tstderr/%s.txt\t%s\n' \
    "$label" "$rc" "$label" "$label" "${rendered% }" >>"$manifest"
  return 0
}

{
  printf 'collection_time_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'declared_platform=%s\n' "$platform"
  printf 'collector=%s\n' "${BASH_SOURCE[0]}"
  printf 'working_directory=%s\n' "$current_dir"
  printf '%s\n' 'notice=--platform 仅是操作者声明，不会自动证明实际 GPU 型号；设备输出和服务日志仍需人工核对。'
  printf '%s\n' 'backend_status=UNKNOWN；实际 Attention/小算子 backend 必须从正式 server log 确认。'
  printf '%s\n' 'privacy=仅采集白名单环境变量，不转储全量 env；配置文件只记录路径与 SHA-256，提交前仍须人工检查主机名和路径。'
} >"$output_dir/collection-info.txt"

run_capture time-utc date -u +%Y-%m-%dT%H:%M:%SZ
run_capture host-uname uname -a
run_capture host-name hostname
if command -v lscpu >/dev/null 2>&1; then
  run_capture host-cpu lscpu
else
  record_missing host-cpu "lscpu command"
fi
if command -v free >/dev/null 2>&1; then
run_capture host-memory free -b
else
  record_missing host-memory "free command"
fi
run_capture os-release sh -c 'test -r /etc/os-release && sed -n "1,160p" /etc/os-release || echo MISSING:/etc/os-release'
run_capture container-cgroup sh -c 'test -r /proc/1/cgroup && sed -n "1,200p" /proc/1/cgroup || echo MISSING:/proc/1/cgroup'

if [[ "$platform" == "BI-V150" ]]; then
  if command -v corex-smi >/dev/null 2>&1; then
    run_capture device-corex-smi corex-smi
  else
    record_missing device-corex-smi "corex-smi command（FlagOS BI-V150 官方资料使用的设备查询命令）"
  fi
else
  found_metax_tool=0
  for tool in mx-smi metax-smi maca-smi; do
    if command -v "$tool" >/dev/null 2>&1; then
      found_metax_tool=1
      run_capture "device-$tool" "$tool"
    else
      record_missing "device-$tool" "$tool command"
    fi
  done
  if ((found_metax_tool == 0)); then
    printf '%s\n' 'UNKNOWN: 未发现候选 MetaX 只读设备命令；请按当前官方镜像文档补充设备证据。' \
      >"$output_dir/device-tool-status.txt"
  fi
fi

environment_file="$output_dir/environment-whitelist.tsv"
printf 'name\tvalue_or_path\tsha256\n' >"$environment_file"

record_env_value() {
  local name="$1"
  if [[ -v "$name" ]]; then
    printf '%s\t%s\tNA\n' "$name" "${!name}" >>"$environment_file"
  else
    printf '%s\tUNSET\tNA\n' "$name" >>"$environment_file"
  fi
}

record_env_path() {
  local name="$1"
  if [[ ! -v "$name" ]]; then
    printf '%s\tUNSET\tNA\n' "$name" >>"$environment_file"
    return
  fi
  local path="${!name}"
  if [[ -f "$path" ]]; then
    printf '%s\t%s\t%s\n' "$name" "$path" "$(sha256sum -- "$path" | awk '{print $1}')" \
      >>"$environment_file"
  else
    printf '%s\t%s\tMISSING\n' "$name" "$path" >>"$environment_file"
  fi
}

for name in \
  VLLM_PLUGINS USE_C_EXTENSION USE_FLAGGEMS \
  VLLM_FL_PLATFORM VLLM_FL_PREFER VLLM_FL_PREFER_ENABLED VLLM_FL_STRICT \
  VLLM_FL_DENY_VENDORS VLLM_FL_ALLOW_VENDORS VLLM_FL_PER_OP \
  VLLM_FL_USE_FLAGGEMS_ATTN VLLM_FL_PLUGIN_MODULES VLLM_FL_OP_FAST_PATH \
  VLLM_FL_FLAGOS_WHITELIST VLLM_FL_FLAGOS_BLACKLIST \
  VLLM_FL_FLAGOS_BLACKLIST_APPEND VLLM_FL_OOT_ENABLED \
  VLLM_FL_OOT_WHITELIST VLLM_FL_OOT_BLACKLIST; do
  record_env_value "$name"
done
for name in VLLM_FL_CONFIG VLLM_FL_OP_CONFIG FLAGGEMS_ENABLE_OPLIST_PATH; do
  record_env_path "$name"
done

find_repo() {
  local name="$1"
  local candidate
  for candidate in "/workspace/$name" "$current_dir/$name" "$script_root/$name"; do
    if [[ -e "$candidate/.git" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

for repo_name in vllm-plugin-FL FlagGems vllm; do
  repo_path="$(find_repo "$repo_name" 2>/dev/null || true)"
  label="$(printf '%s' "$repo_name" | tr '[:upper:]' '[:lower:]')"
  if [[ -n "$repo_path" ]]; then
    printf '%s\n' "$repo_path" >"$output_dir/$label-path.txt"
    run_capture "$label-head" git -C "$repo_path" rev-parse HEAD
    run_capture "$label-branch" git -C "$repo_path" branch --show-current
    run_capture "$label-status" git -C "$repo_path" status --short --branch
  else
    record_missing "$label-head" "$repo_name Git worktree"
    record_missing "$label-branch" "$repo_name Git worktree"
    record_missing "$label-status" "$repo_name Git worktree"
  fi
done

plugin_repo="$(find_repo vllm-plugin-FL 2>/dev/null || true)"
gems_repo="$(find_repo FlagGems 2>/dev/null || true)"
vllm_repo="$(find_repo vllm 2>/dev/null || true)"

print_key_hash() {
  local repo_label="$1"
  local repo_root="$2"
  local relative_path="$3"
  if [[ -z "$repo_root" || ! -d "$repo_root" ]]; then
    printf 'MISSING\t%s:%s\n' "$repo_label" "$relative_path"
    return
  fi
  local file="$repo_root/$relative_path"
  if [[ -f "$file" ]]; then
    printf '%s\t%s\n' "$(sha256sum -- "$file" | awk '{print $1}')" "$file"
  else
    printf 'MISSING\t%s:%s\n' "$repo_label" "$relative_path"
  fi
}

{
  printf 'sha256\tpath\n'
  print_key_hash vllm-plugin-FL "$plugin_repo" benchmarks/benchmark_throughput_serve.py
  print_key_hash vllm-plugin-FL "$plugin_repo" vllm_fl/platform.py
  print_key_hash vllm-plugin-FL "$plugin_repo" vllm_fl/dispatch/config/iluvatar.yaml
  print_key_hash vllm-plugin-FL "$plugin_repo" vllm_fl/dispatch/config/metax.yaml
  print_key_hash vllm-plugin-FL "$plugin_repo" vllm_fl/dispatch/discovery.py
  print_key_hash vllm-plugin-FL "$plugin_repo" vllm_fl/dispatch/policy.py
  print_key_hash vllm-plugin-FL "$plugin_repo" vllm_fl/dispatch/backends/flaggems/flaggems.py
  print_key_hash vllm-plugin-FL "$plugin_repo" vllm_fl/dispatch/backends/flaggems/register_ops.py
  print_key_hash vllm-plugin-FL "$plugin_repo" vllm_fl/dispatch/backends/flaggems/impl/attention.py
  print_key_hash vllm-plugin-FL "$plugin_repo" vllm_fl/dispatch/backends/flaggems/impl/custom_attention.py
  print_key_hash vllm-plugin-FL "$plugin_repo" vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py
  print_key_hash vllm-plugin-FL "$plugin_repo" vllm_fl/dispatch/backends/vendor/iluvatar/register_ops.py
  print_key_hash vllm-plugin-FL "$plugin_repo" vllm_fl/dispatch/backends/vendor/metax/metax.py
  print_key_hash vllm-plugin-FL "$plugin_repo" vllm_fl/dispatch/backends/vendor/metax/register_ops.py
  print_key_hash vllm-plugin-FL "$plugin_repo" vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py
  print_key_hash FlagGems "$gems_repo" src/flag_gems/config.py
  print_key_hash FlagGems "$gems_repo" src/flag_gems/ops/attention.py
  print_key_hash FlagGems "$gems_repo" src/flag_gems/ops/flash_api.py
  print_key_hash FlagGems "$gems_repo" src/flag_gems/ops/flash_kernel.py
  print_key_hash FlagGems "$gems_repo" src/flag_gems/fused/reshape_and_cache_flash.py
  print_key_hash vllm "$vllm_repo" vllm/model_executor/models/registry.py
  print_key_hash vllm "$vllm_repo" vllm/model_executor/models/llama.py
  print_key_hash vllm "$vllm_repo" vllm/model_executor/layers/attention/attention.py
} >"$output_dir/key-file-sha256.tsv"

model_config="/workspace/MiniCPM5-2B/config.json"
if [[ -f "$model_config" ]]; then
  sha256sum -- "$model_config" >"$output_dir/model-config.sha256"
else
  printf 'MISSING  %s\n' "$model_config" >"$output_dir/model-config.sha256"
fi

if command -v python3 >/dev/null 2>&1; then
  run_capture python-version python3 --version
  run_capture python-packages python3 -c 'import importlib.metadata as m, importlib.util as u; exec("def ver(n):\n try:\n  return m.version(n)\n except m.PackageNotFoundError:\n  return MISSING\n"); MISSING="MISSING"; pairs=(("torch","torch"),("triton","triton"),("vllm","vllm"),("vllm-plugin-fl","vllm_fl"),("flag_gems","flag_gems")); [print(f"{dist}\tversion={ver(dist)}\tresolvable_origin={(spec.origin if (spec := u.find_spec(module)) else MISSING)}") for dist,module in pairs]'
  run_capture flaggems-runtime-flags python3 -c 'import flag_gems.config as c; print(f"config_file={c.__file__}"); print(f"has_c_extension={c.has_c_extension}"); print(f"use_c_extension={c.use_c_extension}")'
else
  record_missing python-version "python3 command"
  record_missing python-packages "python3 command"
  record_missing flaggems-runtime-flags "python3 command"
fi

# find_spec 只记录当前解释器可解析的候选 origin，不证明服务进程实际导入该文件；
# metadata 探针若因包缺失而失败，stderr 与退出码会原样保留，不尝试安装。
if ! (
  cd -- "$output_dir" || exit 1
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum
) >"$output_dir/SHA256SUMS"; then
  die "无法生成输出文件校验清单"
fi

echo "环境证据已写入: $output_dir"
echo "提醒: 声明平台不等于设备型号验证；实际 backend 仍须结合 server log。"
