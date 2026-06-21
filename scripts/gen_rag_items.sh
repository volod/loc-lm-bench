#!/usr/bin/env bash
# gen_rag_items.sh -- thin entrypoint for the sample RAG gold-item generator.
#
# Data lives in samples/rag_items_uk.json; logic lives in src/llb/prep/gen_rag_items.py.
# Output goes under DATA_DIR/llb/ (regeneratable runtime data, gitignored). ASCII logs only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/shared/common.sh
. "$SCRIPT_DIR/shared/common.sh"   # PROJECT_ROOT, llb_load_env, llb_python
llb_load_env
PY="$(llb_python)"

# Make src/ importable whether or not the package is installed editable.
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PY" -m llb.prep.gen_rag_items \
  --spec "$PROJECT_ROOT/samples/rag_items_uk.json" \
  --out-dir "$DATA_DIR/llb"
