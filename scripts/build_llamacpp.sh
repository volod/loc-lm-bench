#!/usr/bin/env bash
# Build llama.cpp's `llama-server` from source with CUDA, so the M4.5 LlamaCppLauncher has a
# real hardware-matched binary (the launcher invokes `llama-server` as a subprocess; the binary
# itself is a separate build, like vLLM -- see AGENTS.md "Heavy compilation").
#
# Mirrors build_vllm.sh: sources common.sh for project-root/.env + the canonical MAX_JOBS cap,
# keeps the source checkout clean, and writes only build outputs under $DATA_DIR. ASCII only.
#
# Usage:
#   scripts/build_llamacpp.sh            # build (clone/update + cmake + ninja)
#   LLAMACPP_REF=b6500 scripts/build_llamacpp.sh   # pin a llama.cpp tag/commit
# The resulting binary is printed at the end; add its dir to PATH so `llama-server` resolves.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/shared/common.sh
. "$SCRIPT_DIR/shared/common.sh"
llb_load_env

# Ensure the C++/CUDA build tools (cmake via uv per AGENTS.md; ninja from the system or uv).
if ! command -v cmake >/dev/null 2>&1; then
  echo "[build-llamacpp] cmake not found; installing via uv"
  uv pip install cmake >/dev/null
fi
command -v ninja >/dev/null 2>&1 || uv pip install ninja >/dev/null

JOBS="$(max_jobs)"
SRC_DIR="$DATA_DIR/llb/llamacpp/src"
BUILD_DIR="$DATA_DIR/llb/llamacpp/build"
REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
REF="${LLAMACPP_REF:-}"

# RTX 4060 Ti is Ada Lovelace (compute capability 8.9). Override CUDA_ARCH for another GPU.
CUDA_ARCH="${CUDA_ARCH:-89}"
# CUDA 12.0's nvcc caps the host compiler at gcc-12; default to it when present so a gcc-13
# default toolchain does not break the build. Override CUDA_HOST_CXX to force a compiler.
CUDA_HOST_CXX="${CUDA_HOST_CXX:-$(command -v g++-12 || true)}"
# Prefer the newest local CUDA toolkit (the PATH nvcc may be an older minor).
CUDA_ROOT="${CUDA_ROOT:-$(ls -d /usr/local/cuda-12.* /usr/local/cuda 2>/dev/null | sort -V | tail -1)}"

echo "[build-llamacpp] jobs=$JOBS arch=sm_$CUDA_ARCH cuda_root=${CUDA_ROOT:-PATH} host_cxx=${CUDA_HOST_CXX:-default}"

if [ -d "$SRC_DIR/.git" ]; then
  git -C "$SRC_DIR" fetch --depth 1 origin "${REF:-HEAD}"
  git -C "$SRC_DIR" reset --hard FETCH_HEAD
else
  mkdir -p "$(dirname "$SRC_DIR")"
  if [ -n "$REF" ]; then
    git clone --depth 1 --branch "$REF" "$REPO" "$SRC_DIR"
  else
    git clone --depth 1 "$REPO" "$SRC_DIR"
  fi
fi
echo "[build-llamacpp] source @ $(git -C "$SRC_DIR" rev-parse --short HEAD) (clean checkout)"

CMAKE_ARGS=(
  -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
  -DLLAMA_CURL=ON
  -DLLAMA_BUILD_SERVER=ON
)
[ -n "$CUDA_ROOT" ] && CMAKE_ARGS+=("-DCUDAToolkit_ROOT=$CUDA_ROOT" "-DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc")
[ -n "$CUDA_HOST_CXX" ] && CMAKE_ARGS+=("-DCMAKE_CUDA_HOST_COMPILER=$CUDA_HOST_CXX")

cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --target llama-server --parallel "$JOBS"

BIN="$BUILD_DIR/bin/llama-server"
if [ ! -x "$BIN" ]; then
  echo "[build-llamacpp] ERROR: expected binary not found at $BIN" >&2
  exit 1
fi
echo "[build-llamacpp] [ok] built $BIN"
echo "[build-llamacpp] add to PATH:  export PATH=\"$BUILD_DIR/bin:\$PATH\""
