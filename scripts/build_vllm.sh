#!/usr/bin/env bash
# Install vLLM for the host, with MAX_JOBS-capped compilation and wheel caching (AGENTS.md).
#
# This is the heavy, GPU-host step (a from-source vLLM/flash-attn build can take a long time
# and needs a CUDA toolchain). Model WEIGHTS are cached separately by `llb prep-models`; this
# caches the BUILT WHEELS under $DATA_DIR/wheels/vllm_<key>/ so a rebuild is reused.
#
# Usage:  bash scripts/build_vllm.sh            # install vllm (prebuilt wheel if available)
#         VLLM_SPEC='vllm==0.6.3' bash scripts/build_vllm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/shared/common.sh
. "$SCRIPT_DIR/shared/common.sh"
llb_load_env
PY="$(llb_python)"

# Cap build parallelism via the canonical helper (single source of truth, AGENTS.md).
MAX_JOBS="$(max_jobs)"
export MAX_JOBS
echo "[build-vllm] MAX_JOBS=$MAX_JOBS (capped per AGENTS.md)"

# ABI key from the installed torch + CUDA so cached wheels are reused only when compatible.
KEY="$("$PY" - <<'PYEOF' 2>/dev/null || echo notorch
try:
    import torch
    cuda = (torch.version.cuda or "cpu").replace(".", "")
    print(f"torch{torch.__version__.split('+')[0]}_cu{cuda}")
except Exception:
    print("notorch")
PYEOF
)"
WHEEL_CACHE="$DATA_DIR/wheels/vllm_${KEY}"
mkdir -p "$WHEEL_CACHE"
echo "[build-vllm] wheel cache: $WHEEL_CACHE"

VLLM_SPEC="${VLLM_SPEC:-vllm}"

# Build the wheel into the cache (MAX_JOBS-capped), then install preferring the cache. uv
# venvs ship without pip, so bootstrap it first for `pip wheel`.
"$PY" -m ensurepip --upgrade >/dev/null 2>&1 || uv pip install --python "$PY" pip
"$PY" -m pip wheel --wheel-dir "$WHEEL_CACHE" "$VLLM_SPEC"
uv pip install --python "$PY" --find-links "$WHEEL_CACHE" "$VLLM_SPEC"

echo "[build-vllm] done. Verify: $PY -c 'import vllm; print(vllm.__version__)'"
echo "[build-vllm] then serve a model: llb run-eval --backend vllm --model <hf-repo-id>"
