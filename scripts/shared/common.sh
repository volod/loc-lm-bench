#!/usr/bin/env bash
# Shared bootstrap for loc-lm-bench shell scripts. SOURCE this file; do not execute it.
#
# Provides the canonical project-root / .env / DATA_DIR resolution and the single source of
# truth for the build parallelism cap (AGENTS.md: "the helpers are the single source of
# truth -- do not inline the formula"). ASCII output only.

# Project root = two levels up from this file (scripts/shared/common.sh -> repo root).
llb_project_root() {
  (cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
}
PROJECT_ROOT="${PROJECT_ROOT:-$(llb_project_root)}"

# Load .env (if present) and resolve DATA_DIR against the project root (per AGENTS.md).
llb_load_env() {
  if [ -f "$PROJECT_ROOT/.env" ]; then set -a; . "$PROJECT_ROOT/.env"; set +a; fi
  DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/.data}"
  case "$DATA_DIR" in /*) ;; *) DATA_DIR="$PROJECT_ROOT/$DATA_DIR" ;; esac
  export DATA_DIR
  llb_export_tool_caches
  llb_export_uv_link_mode
}

# Point every tool cache at $DATA_DIR/cache/<tool>, so a redirected DATA_DIR takes the caches with
# it and `rm -rf $DATA_DIR` still clears every temporary artifact. Resolving them HERE rather than
# as static paths in pyproject.toml is what honors a custom DATA_DIR: a config file cannot read
# .env, so a literal default there would keep writing into the project root after an operator
# moved DATA_DIR elsewhere (AGENTS.md: never hardcode, resolve from the project root + .env).
# `make/config.mk` exports the same values for make-driven runs -- Make cannot source this file.
# A tool invoked with neither layer loaded falls back to its OWN default (.ruff_cache/,
# .mypy_cache/, .pytest_cache/), which is gitignored and never masquerades as $DATA_DIR.
llb_export_tool_caches() {
  export RUFF_CACHE_DIR="${RUFF_CACHE_DIR:-$DATA_DIR/cache/ruff}"
  export MYPY_CACHE_DIR="${MYPY_CACHE_DIR:-$DATA_DIR/cache/mypy}"
  export DEEPEVAL_CACHE_FOLDER="${DEEPEVAL_CACHE_FOLDER:-$DATA_DIR/cache/deepeval}"
  export DEEPEVAL_RESULTS_FOLDER="${DEEPEVAL_RESULTS_FOLDER:-$DATA_DIR/cache/deepeval/results}"
}

# Device id (st_dev) of PATH, resolved against the nearest existing ancestor -- the path may
# not exist yet (a not-yet-created .venv or uv cache dir). Empty when it cannot be determined.
llb_path_device() {
  local path="$1" parent
  while [ -n "$path" ] && [ ! -e "$path" ]; do
    parent="$(dirname "$path")"
    [ "$parent" = "$path" ] && break
    path="$parent"
  done
  [ -e "$path" ] || return 1
  stat -c '%d' "$path" 2>/dev/null || stat -f '%d' "$path" 2>/dev/null
}

# Pick uv's cache->venv link mode adaptively and export UV_LINK_MODE only when needed. uv
# tries clone/hardlink first and falls back to a full copy (printing a degraded-performance
# warning) when its shared cache and the target venv live on different filesystems.
#
# When UV_LINK_MODE is unset or "auto" we run the resolver: if the cache and this checkout share
# a device -- the common case, e.g. both on the home disk -- we leave UV_LINK_MODE unset and uv
# keeps its fast default; only when the devices differ (this checkout on a separate disk) do we
# force `copy` to skip the doomed hardlink and its warning. Any other explicit UV_LINK_MODE
# (copy|hardlink|clone|symlink, from the environment, .env, or the command line) is honored as-is.
llb_export_uv_link_mode() {
  local mode="${UV_LINK_MODE:-}"
  if [ -n "$mode" ] && [ "${mode,,}" != "auto" ]; then
    export UV_LINK_MODE
    return 0
  fi
  unset UV_LINK_MODE
  command -v uv >/dev/null 2>&1 || return 0
  local cache_dev root_dev
  cache_dev="$(llb_path_device "$(uv cache dir 2>/dev/null)")" || cache_dev=""
  root_dev="$(llb_path_device "$PROJECT_ROOT/.venv")" || root_dev=""
  if [ -n "$cache_dev" ] && [ -n "$root_dev" ] && [ "$cache_dev" != "$root_dev" ]; then
    export UV_LINK_MODE=copy
  fi
}

# Ensure a .env exists. On first creation, seed it from .env.example and print a friendly,
# colored setup notice (return 1) so the caller stops and lets the user fill in HF_TOKEN and
# review host defaults. This is a normal first-run step, NOT an error. Returns 0 silently when
# .env already exists. Reused by `make venv` and `make demo-eval` so the bootstrap-and-stop
# behavior lives in one place. Colors are emitted only to an interactive terminal (so piped
# output and logs stay clean); framing is ASCII per AGENTS.md.
llb_ensure_env() {
  if [ -f "$PROJECT_ROOT/.env" ]; then return 0; fi
  cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"

  local bold="" cyan="" green="" yellow="" dim="" reset=""
  if [ -t 1 ] && command -v tput >/dev/null 2>&1 && [ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]; then
    bold="$(tput bold)"; cyan="$(tput setaf 6)"; green="$(tput setaf 2)"
    yellow="$(tput setaf 3)"; dim="$(tput dim)"; reset="$(tput sgr0)"
  fi
  local rule="========================================================================"

  printf '\n%s%s%s\n' "$cyan$bold" "$rule" "$reset"
  printf '%s  loc-lm-bench setup -- first run created .env (this is expected)%s\n' "$cyan$bold" "$reset"
  printf '%s%s%s\n' "$cyan$bold" "$rule" "$reset"
  printf '%s[ok]%s seeded %s%s%s from .env.example\n\n' \
    "$green$bold" "$reset" "$bold" "$PROJECT_ROOT/.env" "$reset"
  printf '%sFinish setup before the pipeline can run:%s\n' "$bold" "$reset"
  printf '  %s1.%s set %sHF_TOKEN%s -- gated downloads (MamayLM / Gemma 4 / ingest / prep)\n' \
    "$yellow$bold" "$reset" "$bold" "$reset"
  printf '       %sget a read token at https://huggingface.co/settings/tokens%s\n' "$dim" "$reset"
  printf '  %s2.%s review host defaults for this machine:\n' "$yellow$bold" "$reset"
  printf '       %sDATA_DIR%s                     where indexes / runs / MLflow are written\n' "$bold" "$reset"
  printf '       %sOLLAMA_HOST%s / %sVLLM_HOST%s        candidate inference endpoints\n' "$bold" "$reset" "$bold" "$reset"
  printf '       %sJUDGE_MODEL%s / %sDEEPEVAL_JUDGE_BASE_URL%s   local judge (optional; off unless set)\n\n' \
    "$bold" "$reset" "$bold" "$reset"
  printf '  %sedit%s     %s%s%s\n' "$bold" "$reset" "$cyan" "$PROJECT_ROOT/.env" "$reset"
  printf '  %sthen run%s %s%smake demo-eval%s\n' "$bold" "$reset" "$green$bold" "" "$reset"
  printf '%s%s%s\n\n' "$cyan$bold" "$rule" "$reset"
  return 1
}

# Prefer the project venv python; fall back to system python3.
llb_python() {
  if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    printf '%s' "$PROJECT_ROOT/.venv/bin/python"
  else
    printf '%s' "python3"
  fi
}

# Canonical parallelism cap for from-source C++/CUDA builds (ninja/cmake/nvcc).
# Formula (AGENTS.md): MAX_JOBS = min(cpu_cores // 2, RAM_GiB // 14), floored at 1.
max_jobs() {
  local cores mem_kb mem_gib by_cpu by_ram n
  cores="$(nproc 2>/dev/null || echo 1)"
  mem_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  mem_gib=$(( mem_kb / 1024 / 1024 ))
  by_cpu=$(( cores / 2 ))
  by_ram=$(( mem_gib / 14 ))
  n=$by_cpu
  [ "$by_ram" -lt "$n" ] && n=$by_ram
  [ "$n" -lt 1 ] && n=1
  printf '%s' "$n"
}
