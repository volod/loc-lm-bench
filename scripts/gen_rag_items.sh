#!/usr/bin/env bash
# gen_rag_items.sh -- thin entrypoint for the sample RAG gold-item generator.
#
# Data lives in samples/rag_items_uk.json; logic lives in src/llb/prep/gen_rag_items.py.
# Prefers the project .venv; falls back to system python3 (the generator is stdlib-only).
# Output goes under DATA_DIR/llb/ (regeneratable runtime data, gitignored). ASCII logs only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Honor .env / DATA_DIR; default to <project>/.data. Resolve relative DATA_DIR against
# the project root, not the current directory (per AGENTS.md).
if [ -f "$PROJECT_ROOT/.env" ]; then set -a; . "$PROJECT_ROOT/.env"; set +a; fi
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/.data}"
case "$DATA_DIR" in /*) ;; *) DATA_DIR="$PROJECT_ROOT/$DATA_DIR" ;; esac

# Prefer the project venv; fall back to system python3.
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  PY="$PROJECT_ROOT/.venv/bin/python"
else
  command -v python3 >/dev/null 2>&1 || { echo "[gen_rag_items] ERROR: python3 not found" >&2; exit 1; }
  PY="python3"
fi

# Make src/ importable whether or not the package is installed editable.
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PY" -m llb.prep.gen_rag_items \
  --spec "$PROJECT_ROOT/samples/rag_items_uk.json" \
  --out-dir "$DATA_DIR/llb"
