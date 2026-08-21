#!/usr/bin/env bash
# Create/refresh the LEGACY ENCODER virtualenv: the second scoring environment for the bake-offs.
#
# Four roster candidates ship their forward pass as repository code written against the
# transformers 4.x API (src/llb/rag/encoders/model_stack.py). The repo pins transformers 5.x for the
# shipped path, and on that stack two of them raise at load while two load and return numbers that
# do not reproduce their own model card -- so the rows are a PACKAGING hole in the ranking, not a
# quality result. This venv holds the `[encoders-legacy]` extra (transformers<5 beside the same
# sentence-transformers and torch) so those rows can be scored in a separate pass.
#
# It lives under $DATA_DIR, never beside .venv: it is a run artifact of the comparison lanes, and
# `rm -rf $DATA_DIR` must take it with everything else. The shipped .venv is never touched.
set -euo pipefail

# shellcheck source=scripts/shared/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/shared/common.sh"
llb_load_env

# Match the shipped venv's interpreter: the two passes are compared row by row, so the only
# declared difference between them is the transformers major.
LLB_LEGACY_PYTHON_VERSION="${LLB_LEGACY_PYTHON_VERSION:-3.13}"

llb_encoders_legacy_venv() {
  printf '%s' "$DATA_DIR/venvs/encoders-legacy"
}

llb_encoders_legacy_python() {
  printf '%s' "$(llb_encoders_legacy_venv)/bin/python"
}

llb_encoders_legacy_sync() {
  local venv python
  venv="$(llb_encoders_legacy_venv)"
  python="$(llb_encoders_legacy_python)"
  command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv not found -- install from https://docs.astral.sh/uv/" >&2
    return 1
  }
  # `uv sync` against the SHIPPED uv.lock, exactly as scripts/setup_venv.sh builds .venv, pointed
  # at a second environment. The conflict declaration in pyproject.toml is what lets one lock hold
  # both resolutions, so this pass runs the same pinned versions CI would, minus the transformers
  # major. A bare `uv pip install` here resolves outside the lock and has been observed to drop a
  # transitive pin the lock carries, which fails the CLI at import time rather than at install.
  UV_PROJECT_ENVIRONMENT="$venv" uv sync --extra encoders-legacy --python "$LLB_LEGACY_PYTHON_VERSION"
  echo "[encoders-legacy] ready: $python"
  "$python" -c 'import transformers; print("[encoders-legacy] transformers", transformers.__version__)'
}

llb_encoders_legacy_sync "$@"
