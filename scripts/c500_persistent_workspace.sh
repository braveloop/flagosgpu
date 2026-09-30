#!/usr/bin/env bash
set -Eeuo pipefail

# Re-enter through a login shell so the image's MACA/PATH setup is applied.
if [[ "${FLAGOS_C500_LOGIN_READY:-0}" != "1" ]]; then
  export FLAGOS_C500_LOGIN_READY=1
  exec /bin/bash --login "$(readlink -f -- "$0")" "$@"
fi

PERSIST_ROOT=/data/flagos
PLUGIN_SRC="$PERSIST_ROOT/src/vllm-plugin-FL"
FLAGGEMS_SRC="$PERSIST_ROOT/src/FlagGems"
MODEL_SRC="$PERSIST_ROOT/models/MiniCPM5-2B"
DATASET_SRC="$PERSIST_ROOT/datasets/math_500"

PLUGIN_WORK=/workspace/vllm-plugin-FL
FLAGGEMS_WORK=/workspace/FlagGems
MODEL_WORK=/workspace/MiniCPM5-2B
DATASET_WORK=/workspace/evalscope-datasets/math_500

PLUGIN_COMMIT=13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0
FLAGGEMS_COMMIT=a7620cc191a0b42e040194622c5758b22a7a25dc
DEV_BRANCH=bravegpuwinner/s2-dev

errors=0

usage() {
  cat <<'EOF'
Usage: c500_persistent_workspace.sh check|prepare|serve

  check    Inspect persistent sources, workspace assets, MACA login setup, and
           the package paths Python would import. Makes no changes.
  prepare  Create only missing /workspace symlinks after validating /data.
           It never copies, removes, renames, or replaces an existing path.
  serve    Re-run check, reject an existing server, then exec the official C500
           serve command. It does not install dependencies or run in background.
EOF
}

ok() { printf 'OK: %s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
fail() {
  printf 'ERROR: %s\n' "$*" >&2
  errors=$((errors + 1))
}
die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

verify_repo() {
  local label=$1 path=$2 expected_commit=$3 expected_branch=${4:-}
  local actual_commit actual_branch dirty

  if [[ ! -d "$path/.git" ]]; then
    fail "$label repository is missing at $path"
    return 1
  fi
  actual_commit=$(git -C "$path" rev-parse HEAD 2>/dev/null) || {
    fail "$label HEAD cannot be read at $path"
    return 1
  }
  if ! git -C "$path" merge-base --is-ancestor "$expected_commit" "$actual_commit" 2>/dev/null; then
    fail "$label HEAD $actual_commit does not descend from required base $expected_commit"
  fi

  if [[ -n "$expected_branch" ]]; then
    actual_branch=$(git -C "$path" symbolic-ref --quiet --short HEAD 2>/dev/null || true)
    [[ "$actual_branch" == "$expected_branch" ]] ||
      fail "$label branch is ${actual_branch:-DETACHED}; expected $expected_branch"
  fi

  dirty=$(git -C "$path" status --porcelain 2>/dev/null) || {
    fail "$label status cannot be read at $path"
    return 1
  }
  [[ -z "$dirty" ]] || fail "$label repository has local changes at $path"
  ok "$label source: $path @ $actual_commit"
}

verify_model() {
  local path=$1 shard
  [[ -d "$path" ]] || return 1
  [[ -s "$path/config.json" && -s "$path/tokenizer.json" ]] || return 1
  shard=$(find "$path" -maxdepth 1 -type f -name '*.safetensors' -size +0c -print -quit)
  [[ -n "$shard" ]]
}

verify_dataset() {
  local path=$1
  [[ -d "$path" && -s "$path/test.jsonl" && -s "$path/eval.yaml" ]]
}

report_binding() {
  local label=$1 workspace_path=$2 persistent_path=$3 resolved
  if [[ ! -e "$workspace_path" && ! -L "$workspace_path" ]]; then
    fail "$label workspace path is missing: $workspace_path"
    return
  fi
  resolved=$(readlink -f -- "$workspace_path" 2>/dev/null || true)
  if [[ "$resolved" == "$persistent_path" ]]; then
    ok "$label workspace path resolves to persistent storage: $workspace_path -> $resolved"
  else
    warn "$label workspace path is not bound to $persistent_path (resolved: ${resolved:-BROKEN})"
  fi
}

check_import_paths() {
  python3 - "$PERSIST_ROOT" <<'PY'
import importlib.util
import os
import sys

persistent_root = os.path.realpath(sys.argv[1])
missing = False
for name in ("vllm", "vllm_fl", "flag_gems"):
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"ERROR: Python package is missing: {name}", file=sys.stderr)
        missing = True
        continue
    origin = spec.origin or "namespace"
    locations = list(spec.submodule_search_locations or ())
    print(f"INFO: Python import path: {name} origin={origin} locations={locations}")
    candidates = [origin, *locations]
    if not any(
        item != "namespace"
        and os.path.commonpath((persistent_root, os.path.realpath(item))) == persistent_root
        for item in candidates
    ):
        print(
            f"WARN: {name} is not imported from {persistent_root}; the persistent "
            "development source must not be claimed active",
            file=sys.stderr,
        )
sys.exit(1 if missing else 0)
PY
}

run_check() {
  errors=0

  [[ -d /data && -d /root ]] || fail '/data and /root are not both available'
  verify_repo vllm-plugin-FL "$PLUGIN_SRC" "$PLUGIN_COMMIT" "$DEV_BRANCH" || true
  verify_repo FlagGems "$FLAGGEMS_SRC" "$FLAGGEMS_COMMIT" "$DEV_BRANCH" || true

  if verify_model "$MODEL_WORK"; then
    ok "model contents: $MODEL_WORK"
  else
    fail "model contents are incomplete at $MODEL_WORK"
  fi
  if verify_dataset "$DATASET_WORK"; then
    ok "dataset contents: $DATASET_WORK"
  else
    fail "dataset contents are incomplete at $DATASET_WORK"
  fi

  report_binding vllm-plugin-FL "$PLUGIN_WORK" "$PLUGIN_SRC"
  report_binding FlagGems "$FLAGGEMS_WORK" "$FLAGGEMS_SRC"
  report_binding model "$MODEL_WORK" "$MODEL_SRC"
  report_binding dataset "$DATASET_WORK" "$DATASET_SRC"

  if verify_model "$MODEL_SRC"; then
    ok "persistent model is ready: $MODEL_SRC"
  else
    warn "persistent model is not ready: $MODEL_SRC"
  fi
  if verify_dataset "$DATASET_SRC"; then
    ok "persistent dataset is ready: $DATASET_SRC"
  else
    warn "persistent dataset is not ready: $DATASET_SRC"
  fi

  command -v python3 >/dev/null 2>&1 || fail 'python3 is not on login-shell PATH'
  command -v vllm >/dev/null 2>&1 || fail 'vllm is not on login-shell PATH'
  [[ -n "${MACA_PATH:-}" && -d "$MACA_PATH" ]] ||
    fail 'MACA_PATH is not set to an existing directory by the login shell'
  [[ ":${LD_LIBRARY_PATH:-}:" == *:/opt/maca/lib:* ]] ||
    fail 'LD_LIBRARY_PATH does not contain /opt/maca/lib'

  if command -v python3 >/dev/null 2>&1; then
    check_import_paths || fail 'one or more required Python packages are missing'
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    warn "CUDA_VISIBLE_DEVICES is currently set to '${CUDA_VISIBLE_DEVICES}'; serve mode unsets it"
  else
    ok 'CUDA_VISIBLE_DEVICES is unset'
  fi

  if ((errors != 0)); then
    printf 'CHECK FAILED: %d prerequisite error(s)\n' "$errors" >&2
    return 1
  fi
  printf 'CHECK PASSED\n'
}

link_if_missing() {
  local label=$1 source=$2 destination=$3
  if [[ -e "$destination" || -L "$destination" ]]; then
    warn "$label unchanged because destination already exists: $destination"
    return 0
  fi
  ln -s -- "$source" "$destination"
  ok "$label linked: $destination -> $source"
}

run_prepare() {
  errors=0
  verify_repo vllm-plugin-FL "$PLUGIN_SRC" "$PLUGIN_COMMIT" "$DEV_BRANCH" || true
  verify_repo FlagGems "$FLAGGEMS_SRC" "$FLAGGEMS_COMMIT" "$DEV_BRANCH" || true
  verify_model "$MODEL_SRC" || fail "refusing to link incomplete model: $MODEL_SRC"
  verify_dataset "$DATASET_SRC" || fail "refusing to link incomplete dataset: $DATASET_SRC"
  ((errors == 0)) || die 'persistent sources/assets did not pass validation; nothing was linked'

  mkdir -p -- /workspace/evalscope-datasets
  link_if_missing vllm-plugin-FL "$PLUGIN_SRC" "$PLUGIN_WORK"
  link_if_missing FlagGems "$FLAGGEMS_SRC" "$FLAGGEMS_WORK"
  link_if_missing model "$MODEL_SRC" "$MODEL_WORK"
  link_if_missing dataset "$DATASET_SRC" "$DATASET_WORK"
  run_check
}

run_serve() {
  run_check || die 'serve prerequisites did not pass'
  if pgrep -af '[v]llm serve' >/dev/null 2>&1; then
    pgrep -af '[v]llm serve' >&2 || true
    die 'a vllm serve process already exists; refusing to start another'
  fi

  unset CUDA_VISIBLE_DEVICES
  export VLLM_PLUGINS=fl
  exec vllm serve /workspace/MiniCPM5-2B \
    --port 9031 \
    --served-model-name minicpm \
    --gpu-memory-utilization 0.85 \
    --max-model-len 131072
}

[[ $# -eq 1 ]] || {
  usage >&2
  exit 2
}

case "$1" in
  check) run_check ;;
  prepare) run_prepare ;;
  serve) run_serve ;;
  *)
    usage >&2
    exit 2
    ;;
esac
