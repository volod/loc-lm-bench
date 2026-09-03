#!/usr/bin/env bash
# Venv lifecycle for `make venv`: decide reuse-or-rebuild BEFORE syncing, sync, then reinstall the
# hardware-matched stack a rebuild discarded. The decision itself lives in llb.build.venv_state.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/shared/common.sh
. "$SCRIPT_DIR/shared/common.sh"
llb_load_env  # also resolves UV_LINK_MODE for the sync below

export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Defaults match [tool.llb.toolchain]; Make already exports both for `make venv`.
if [ -z "${VENV:-}" ]; then
  VENV="$PROJECT_ROOT/$(PYTHONPATH="$PROJECT_ROOT/src" python3 -m llb.build.toolchain venv)"
fi
if [ -z "${PYTHON_VERSION:-}" ]; then
  PYTHON_VERSION="$(PYTHONPATH="$PROJECT_ROOT/src" python3 -m llb.build.toolchain python-version)"
fi
VENV_INSTALL_VLLM="${VENV_INSTALL_VLLM:-auto}"
RECREATE_VENV="${RECREATE_VENV:-}"
EXTRAS="${EXTRAS:-}"
UV_SYNC_ARGS="${UV_SYNC_ARGS:---inexact --python $PYTHON_VERSION}"
# llb.build.venv_state exits with this when a rebuild would discard a stack nobody asked to replace.
PLAN_REFUSED=3

# Reject an unusable VENV_INSTALL_VLLM before the sync rather than after it: the operator finds out
# while the venv is still whole.
llb_require_vllm_mode() {
  case "$VENV_INSTALL_VLLM" in
    0 | false | no | 1 | true | yes | auto) ;;
    *)
      echo "ERROR: VENV_INSTALL_VLLM must be auto, 1, or 0 (got $VENV_INSTALL_VLLM)" >&2
      exit 2
      ;;
  esac
}

# Print the reuse-or-rebuild account and export the plan. A refusal stops the target; any other
# planner failure leaves the target exactly as capable as it was before the check existed, because
# `make venv` is the command an operator runs to REPAIR a half-broken venv.
llb_venv_plan() {
  LLB_VENV_ACTION=create
  LLB_VENV_FORCE_VLLM=0
  if [ ! -x "$VENV/bin/python" ]; then
    echo "[venv] creating $VENV (py$PYTHON_VERSION)"
    return 0
  fi
  local plan status=0
  local -a plan_args=(--venv "$VENV" --root "$PROJECT_ROOT" --extras "$EXTRAS")
  if [ -n "$RECREATE_VENV" ]; then plan_args+=(--recreate); fi
  plan="$("$(llb_python)" -m llb.build.venv_state "${plan_args[@]}")" || status=$?
  if [ "$status" -eq "$PLAN_REFUSED" ]; then
    exit 1
  elif [ "$status" -ne 0 ]; then
    echo "[venv] staleness check unavailable (exit $status) -- letting uv decide" >&2
    LLB_VENV_ACTION=unchecked
    return 0
  fi
  eval "$plan"
}

llb_venv_sync() {
  local sync_args
  read -r -a sync_args <<<"$UV_SYNC_ARGS"
  if [ -n "$RECREATE_VENV" ] && [ -d "$VENV" ]; then
    echo "[venv] RECREATE_VENV set -- removing $VENV"
    rm -rf "$VENV"
  fi
  echo "[venv] uv link mode: ${UV_LINK_MODE:-default (cache + checkout share a device)}"
  UV_PROJECT_ENVIRONMENT="$VENV" uv sync "${sync_args[@]}"
  echo "[venv] ready: $VENV ($LLB_VENV_ACTION; extras: $EXTRAS; versions pinned by uv.lock)"
}

# vLLM/torch/flash-attn are hardware-matched and live outside uv.lock, so a rebuild leaves the venv
# holding the lock's torch with nothing to match it. That reinstall is not skippable.
llb_venv_install_vllm() {
  if [ "$LLB_VENV_FORCE_VLLM" = "1" ]; then
    echo "[venv] installing vLLM binary wheels (forced: the sync moved the stack under it," \
      "VENV_INSTALL_VLLM=$VENV_INSTALL_VLLM)"
    bash "$PROJECT_ROOT/scripts/build_vllm.sh"
    return 0
  fi
  case "$VENV_INSTALL_VLLM" in
    0 | false | no) echo "[venv] vLLM install skipped (VENV_INSTALL_VLLM=$VENV_INSTALL_VLLM)" ;;
    1 | true | yes)
      echo "[venv] installing vLLM binary wheels (forced)"
      bash "$PROJECT_ROOT/scripts/build_vllm.sh"
      ;;
    auto)
      if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo "[venv] CUDA host detected; installing vLLM binary wheels"
        bash "$PROJECT_ROOT/scripts/build_vllm.sh"
      else
        echo "[venv] vLLM install skipped (no CUDA GPU detected; set VENV_INSTALL_VLLM=1 to force)"
      fi
      ;;
  esac
}

llb_require_vllm_mode
llb_venv_plan
llb_venv_sync
llb_venv_install_vllm
