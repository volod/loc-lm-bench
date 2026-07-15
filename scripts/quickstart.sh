#!/usr/bin/env bash
# Quickstart orchestration for README goldset and PDF corpus flows.
#
# The orchestration lives in sourced fragments under scripts/quickstart/: helpers, model selection,
# PDF draft staging, serving config, the three tracks (A committed-goldset, B PDF corpus, C mixed
# corpus), and target dispatch. This entrypoint owns process setup, the QS_* configuration block,
# and the logging wrapper (main). The QS_* globals are consumed by the sourced fragments, so the
# file-wide SC2034 disable keeps shellcheck from flagging them as unused in this file.
# shellcheck disable=SC2034
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QS_DIR="$ROOT/scripts/quickstart"
# shellcheck source=scripts/shared/common.sh
source "$ROOT/scripts/shared/common.sh"
llb_load_env
cd "$PROJECT_ROOT"

# shellcheck source=scripts/quickstart/helpers.sh
source "$QS_DIR/helpers.sh"

QS_ROOT="$(resolve_path "${QUICKSTART_ROOT:-$DATA_DIR}")"
QS_LOG_DIR="$(resolve_path "${QUICKSTART_LOG_DIR:-$DATA_DIR/llb/logs/quickstart}")"
QS_UV_CACHE_DIR="$(resolve_path "${QUICKSTART_UV_CACHE_DIR:-$QS_ROOT/uv-cache}")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$QS_UV_CACHE_DIR}"
llb_export_uv_link_mode

QS_A_DATA="$(resolve_path "${QUICKSTART_A_DATA_DIR:-$QS_ROOT/quickstart-leaderboard}")"
QS_A_GOLDSET="$(resolve_path "${QUICKSTART_A_GOLDSET:-samples/goldsets/ua_squad_postedited_v1/goldset.jsonl}")"
QS_A_CORPUS="$(resolve_path "${QUICKSTART_A_CORPUS:-samples/goldsets/ua_squad_postedited_v1/corpus}")"
QS_A_SWEEP_ID="${QUICKSTART_A_SWEEP_ID:-qs-committed}"
QS_MODELS_MANIFEST="$(resolve_path "${MODELS_MANIFEST:-samples/configs/models_uk.yaml}")"
QS_SKIP_APT="${QUICKSTART_SKIP_APT:-1}"
QS_SETUP_VENV="${QUICKSTART_SETUP_VENV:-auto}"
QS_PREP_MODELS="${QUICKSTART_PREP_MODELS:-1}"
QS_PREP_SERVING_TARGETS="${QUICKSTART_PREP_SERVING_TARGETS:-1}"
QS_RUN_SWEEP="${QUICKSTART_RUN_SWEEP:-1}"
QS_RUN_PLATFORM_MATRIX="${QUICKSTART_RUN_PLATFORM_MATRIX:-1}"
QS_RUN_SECURITY="${QUICKSTART_RUN_SECURITY:-1}"
QS_RECOMMEND_MIN_CASES="${QUICKSTART_RECOMMEND_MIN_CASES:-1}"
QS_SWEEP_LIMIT="${QUICKSTART_SWEEP_LIMIT:-}"
QS_PROMPT_DIR="$(resolve_path "${QUICKSTART_PROMPT_DIR:-$QS_A_DATA/prompt-system/quickstart}")"
QS_PROMPT_ID="${QUICKSTART_PROMPT_ID:-}"
QS_GPU_GB="${QUICKSTART_GPU_GB:-}"
QS_RAG_K="${RAG_K:-10}"
QS_SPLIT="${SPLIT:-final}"
QS_HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
QS_SECURITY_MODEL="${QUICKSTART_SECURITY_MODEL:-hf.co/INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0-GGUF:Q4_K_M}"
QS_SECURITY_BACKEND="${QUICKSTART_SECURITY_BACKEND:-ollama}"
QS_SECURITY_CASES="$(resolve_path "${QUICKSTART_SECURITY_CASES:-samples/benchmarks/security_cases_uk.json}")"
QS_SECURITY_VERIFICATION_REF="$(resolve_path "${QUICKSTART_SECURITY_VERIFICATION_REF:-samples/verification/composite_samples/security/sample_manifest.json}")"

QS_PDF_SOURCE="$(resolve_path "${QUICKSTART_PDF_SOURCE:-$QS_ROOT/quickstart-pdf-corpus}")"
QS_PDF_MD="$(resolve_path "${QUICKSTART_PDF_MD:-$QS_ROOT/quickstart-pdf-corpus-md}")"
QS_PDF_RAG_DATA="$(resolve_path "${QUICKSTART_PDF_RAG_DATA:-$QS_ROOT/quickstart-pdf-corpus-rag}")"
QS_PDF_DRAFT_MD="$(resolve_path "${QUICKSTART_PDF_DRAFT_MD:-$QS_ROOT/quickstart-pdf-corpus-draft-md}")"
QS_PDF_DRAFT="$(resolve_path "${QUICKSTART_PDF_DRAFT:-$QS_ROOT/quickstart-pdf-corpus-draft}")"
QS_PDF_GRAPH_DATA="$(resolve_path "${QUICKSTART_PDF_GRAPH_DATA:-$QS_ROOT/quickstart-pdf-corpus-graph}")"
QS_PDF_LEADERBOARD_DATA="$(resolve_path "${QUICKSTART_PDF_LEADERBOARD_DATA:-$QS_ROOT/quickstart-pdf-corpus-leaderboard}")"
QS_PDF_MODEL_BENCH_DATA="$(resolve_path "${QUICKSTART_PDF_MODEL_BENCH_DATA:-$QS_A_DATA}")"
QS_PDF_ACCEPTED="$(resolve_path "${QUICKSTART_PDF_ACCEPTED:-$QS_PDF_DRAFT/accepted}")"
QS_PDF_DRAFT_DOCS="${QUICKSTART_PDF_DRAFT_DOCS:-all}"
QS_DRAFT_MODEL="${QUICKSTART_DRAFT_MODEL:-auto}"
QS_DRAFT_ENDPOINT="${QUICKSTART_DRAFT_ENDPOINT:-local}"
QS_DRAFT_BACKEND="${QUICKSTART_DRAFT_BACKEND:-ollama}"
QS_DRAFT_BASE_URL="${QUICKSTART_DRAFT_BASE_URL:-}"
QS_DRAFT_MAX_ITEMS="${QUICKSTART_DRAFT_MAX_ITEMS:-180}"
QS_DRAFT_VERIFY_N="${QUICKSTART_DRAFT_VERIFY_N:-40}"
QS_DRAFT_TIMEOUT="${QUICKSTART_DRAFT_TIMEOUT:-900}"
QS_DRAFT_MAX_TOKENS="${QUICKSTART_DRAFT_MAX_TOKENS:-4096}"
QS_DRAFT_TEMPERATURE="${QUICKSTART_DRAFT_TEMPERATURE:-0}"
QS_DRAFT_NUM_CTX="${QUICKSTART_DRAFT_NUM_CTX:-16384}"
QS_DRAFT_VLLM_PORT="${QUICKSTART_DRAFT_VLLM_PORT:-8000}"
QS_DRAFT_VLLM_GPU_MEMORY_UTILIZATION="${QUICKSTART_DRAFT_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
QS_DRAFT_VLLM_MAX_MODEL_LEN="${QUICKSTART_DRAFT_VLLM_MAX_MODEL_LEN:-}"
QS_DRAFT_VLLM_CPU_OFFLOAD_GB="${QUICKSTART_DRAFT_VLLM_CPU_OFFLOAD_GB:-}"
QS_DRAFT_VLLM_KV_OFFLOADING_SIZE_GB="${QUICKSTART_DRAFT_VLLM_KV_OFFLOADING_SIZE_GB:-}"
QS_DRAFT_VLLM_DTYPE="${QUICKSTART_DRAFT_VLLM_DTYPE:-auto}"
QS_DRAFT_VLLM_QUANTIZATION="${QUICKSTART_DRAFT_VLLM_QUANTIZATION:-}"
QS_DRAFT_VLLM_STARTUP_TIMEOUT="${QUICKSTART_DRAFT_VLLM_STARTUP_TIMEOUT:-600}"
QS_DRAFT_EXTRACT_MAX_CHARS="${QUICKSTART_DRAFT_EXTRACT_MAX_CHARS:-}"
QS_DRAFT_EXTRACT_CHUNK_OVERLAP="${QUICKSTART_DRAFT_EXTRACT_CHUNK_OVERLAP:-}"
QS_DRAFT_CONCURRENCY="${QUICKSTART_DRAFT_CONCURRENCY:-}"
QS_MODEL_SELECTION="${QUICKSTART_MODEL_SELECTION:-auto}"
QS_ASSUME_YES="${QUICKSTART_ASSUME_YES:-0}"
QS_PDF_MIN_CHARS="${QUICKSTART_PDF_MIN_CHARS:-500}"
QS_PDF_PARSER="${QUICKSTART_PDF_PARSER:-auto}"

QS_CORPUS_SRC="$(resolve_path "${QUICKSTART_CORPUS_SRC:-$QS_ROOT/quickstart-corpus}")"
QS_CORPUS_MD="$(resolve_path "${QUICKSTART_CORPUS_MD:-$QS_ROOT/quickstart-corpus-md}")"
QS_CORPUS_RAG_DATA="$(resolve_path "${QUICKSTART_CORPUS_RAG_DATA:-$QS_ROOT/quickstart-corpus-rag}")"
QS_CORPUS_DRAFT="$(resolve_path "${QUICKSTART_CORPUS_DRAFT:-$QS_ROOT/quickstart-corpus-draft}")"
QS_CORPUS_GRAPH_DATA="$(resolve_path "${QUICKSTART_CORPUS_GRAPH_DATA:-$QS_ROOT/quickstart-corpus-graph}")"
QS_CORPUS_MIN_CHARS="${QUICKSTART_CORPUS_MIN_CHARS:-500}"
QS_CORPUS_PARSER="${QUICKSTART_CORPUS_PARSER:-auto}"
QS_CORPUS_RESUME="$([ -n "${QUICKSTART_CORPUS_RESUME:-}" ] && resolve_path "$QUICKSTART_CORPUS_RESUME" || true)"

# shellcheck source=scripts/quickstart/model_select.sh
source "$QS_DIR/model_select.sh"
# shellcheck source=scripts/quickstart/pdf_draft.sh
source "$QS_DIR/pdf_draft.sh"
# shellcheck source=scripts/quickstart/serving.sh
source "$QS_DIR/serving.sh"
# shellcheck source=scripts/quickstart/track_a.sh
source "$QS_DIR/track_a.sh"
# shellcheck source=scripts/quickstart/track_b.sh
source "$QS_DIR/track_b.sh"
# shellcheck source=scripts/quickstart/track_c.sh
source "$QS_DIR/track_c.sh"
# shellcheck source=scripts/quickstart/dispatch.sh
source "$QS_DIR/dispatch.sh"

main() {
  local target="${1:-help}"
  if [ "${2:-}" = "--no-log" ]; then
    run_target "$target"
    return
  fi

  mkdir -p "$QS_LOG_DIR"
  local stamp log
  stamp="$(date +%Y%m%d-%H%M%S)"
  log="$QS_LOG_DIR/quickstart-$target-$stamp.log"
  printf '[quickstart] target=%s\n' "$target"
  printf '[quickstart] log=%s\n' "$(rel_path "$log")"
  set +e
  bash "$0" "$target" --no-log 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -eq 0 ]; then
    printf '[quickstart] OK target=%s log=%s\n' "$target" "$(rel_path "$log")"
  else
    printf '[quickstart] FAILED target=%s exit=%s log=%s\n' "$target" "$rc" "$(rel_path "$log")" >&2
  fi
  return "$rc"
}

main "$@"
