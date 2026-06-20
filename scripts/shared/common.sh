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
