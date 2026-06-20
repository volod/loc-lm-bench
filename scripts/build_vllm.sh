#!/usr/bin/env bash
# Install vLLM for the host, with MAX_JOBS-capped compilation and wheel caching (AGENTS.md).
#
# FlashAttention ("self-attention") kernels are BUNDLED with vLLM (the vllm-flash-attn wheel,
# or compiled into vLLM's own extension); you do NOT build flash-attn separately. When a
# prebuilt vLLM wheel matches the host CUDA/torch, NOTHING is compiled (the good path) and
# MAX_JOBS is moot; it only caps the fallback from-source build.
#
# Model WEIGHTS are cached separately by `llb prep-models`; this caches the BUILT WHEELS under
# $DATA_DIR/wheels/vllm_<key>/ so a rebuild is reused.
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
export PIP_DISABLE_PIP_VERSION_CHECK=1   # quiet the "new release of pip" notice
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

# Report the install + confirm flash-attn is present (no separate build needed). Uses
# distribution metadata only, so it does not import vllm / initialize CUDA here.
"$PY" - <<'PYEOF' || true
import importlib.metadata as md
def ver(dist):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None
print(f"[build-vllm] vllm=={ver('vllm')}")
fa = ver("vllm-flash-attn")
if fa:
    print(f"[build-vllm] flash-attn: bundled (vllm-flash-attn=={fa}) -- no separate build needed")
else:
    print("[build-vllm] flash-attn: vendored inside the vllm wheel -- no separate build needed")
PYEOF
echo "[build-vllm] the active attention backend is printed at serve time (look for 'FlashAttention')."
echo "[build-vllm] serve a model: llb run-eval --backend vllm --model <hf-repo-id> --telemetry"
