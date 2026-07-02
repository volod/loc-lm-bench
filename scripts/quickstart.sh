#!/usr/bin/env bash
# Quickstart orchestration for README goldset and PDF corpus flows.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/shared/common.sh
source "$ROOT/scripts/shared/common.sh"
llb_load_env
cd "$PROJECT_ROOT"

resolve_path() {
  local value="$1"
  case "$value" in
    /*) printf '%s' "$value" ;;
    *) printf '%s/%s' "$PROJECT_ROOT" "$value" ;;
  esac
}

rel_path() {
  local value="$1"
  case "$value" in
    "$PROJECT_ROOT"/*) printf '%s' "${value#"$PROJECT_ROOT"/}" ;;
    *) printf '%s' "$value" ;;
  esac
}

make_cmd() {
  make -C "$PROJECT_ROOT" --no-print-directory "$@"
}

make_with_data_dir() {
  local data_dir="$1"
  shift
  DATA_DIR="$data_dir" make_cmd "$@"
}

heading() {
  printf '\n### [%s] %s\n' "$1" "$2"
}

result() {
  printf '[result] %s\n' "$1"
}

is_yes_value() {
  case "${1,,}" in
    1|true|yes|y) return 0 ;;
    *) return 1 ;;
  esac
}

is_interactive() {
  [ -t 0 ] && [ -t 1 ]
}

prompt_yes_no() {
  local question="$1"
  local default="${2:-no}"
  local suffix answer
  if is_yes_value "$QS_ASSUME_YES"; then
    printf '[prompt] %s yes (QUICKSTART_ASSUME_YES=1)\n' "$question"
    return 0
  fi
  if ! is_interactive; then
    if [ "$default" = "yes" ]; then
      printf '[prompt] %s yes (non-interactive default)\n' "$question"
      return 0
    fi
    echo "ERROR: this step needs confirmation: $question" >&2
    echo "Set QUICKSTART_ASSUME_YES=1 or provide QUICKSTART_DRAFT_MODEL explicitly." >&2
    exit 2
  fi
  if [ "$default" = "yes" ]; then
    suffix="[Y/n]"
  else
    suffix="[y/N]"
  fi
  read -r -p "$question $suffix " answer
  answer="${answer:-$default}"
  is_yes_value "$answer"
}

prompt_value() {
  local question="$1"
  local answer
  if ! is_interactive; then
    echo "ERROR: cannot prompt in non-interactive mode: $question" >&2
    exit 2
  fi
  read -r -p "$question " answer
  printf '%s' "$answer"
}

quickstart_py() {
  test -x "$PROJECT_ROOT/.venv/bin/python" || {
    echo "ERROR: .venv missing -- run make venv first" >&2
    exit 1
  }
  "$PROJECT_ROOT/.venv/bin/python" -m llb.quickstart.model_choice "$@"
}

list_local_models() {
  if command -v ollama >/dev/null 2>&1; then
    ollama list | sed 's/^/[ollama] /'
  else
    result "ollama command not found; enter the local endpoint model id manually"
  fi
}

pdf_bench_json() {
  printf '%s/recommend/pdf_model_choice.json' "$QS_PDF_MODEL_BENCH_DATA"
}

pdf_bench_has_runs() {
  find "$QS_PDF_MODEL_BENCH_DATA/run-eval" -mindepth 2 -maxdepth 2 -name manifest.json -print -quit \
    2>/dev/null | grep -q .
}

write_pdf_bench_recommendation() {
  local json
  json="$(pdf_bench_json)"
  make_with_data_dir "$QS_PDF_MODEL_BENCH_DATA" recommend \
    RECOMMEND_MIN_CASES="$QS_RECOMMEND_MIN_CASES" \
    RECOMMEND_JSON_OUT="$json" \
    RECOMMEND_NO_CHART=1
  result "model recommendation JSON: $(rel_path "$json")"
}

run_pdf_model_benchmark() {
  heading "model" "benchmark local candidates for this host"
  result "benchmark data dir: $(rel_path "$QS_PDF_MODEL_BENCH_DATA")"
  if ! prompt_yes_no "The local model benchmark can take roughly 1-4 hours. Proceed?" "no"; then
    echo "ERROR: model benchmark was not approved" >&2
    echo "Provide QUICKSTART_DRAFT_MODEL=<local-model-id> or rerun with QUICKSTART_MODEL_SELECTION=choose." >&2
    exit 2
  fi
  track_a_setup
  track_a_rag
  track_a_models
  track_a_eval
}

select_model_from_benchmark_json() {
  local json choice count model
  json="$(pdf_bench_json)"
  quickstart_py table "$json"
  case "$QS_MODEL_SELECTION" in
    auto|benchmark)
      model="$(quickstart_py selection "$json" recommended_for_host)"
      if prompt_yes_no "Use recommended local drafter model '$model'?" "yes"; then
        QS_DRAFT_MODEL="$model"
        QS_DRAFT_ENDPOINT="local"
        return 0
      fi
      ;;
  esac
  count="$(quickstart_py count "$json")"
  choice="$(prompt_value "Select a model number from 1-$count, or enter a model id:")"
  case "$choice" in
    '' ) echo "ERROR: empty model choice" >&2; exit 2 ;;
    *[!0-9]* ) QS_DRAFT_MODEL="$choice" ;;
    * )
      QS_DRAFT_MODEL="$(quickstart_py candidate "$json" "$choice")"
      ;;
  esac
  QS_DRAFT_ENDPOINT="local"
}

select_manual_local_model() {
  list_local_models
  QS_DRAFT_MODEL="$(prompt_value "Enter local model id for ontology drafting:")"
  test -n "$QS_DRAFT_MODEL" || { echo "ERROR: empty model id" >&2; exit 2; }
  QS_DRAFT_ENDPOINT="local"
}

select_frontier_model() {
  result "frontier mode sends corpus text to the configured provider through litellm"
  result "set the matching provider API key in the environment before drafting"
  QS_DRAFT_MODEL="$(prompt_value "Enter litellm model id for frontier drafting:")"
  test -n "$QS_DRAFT_MODEL" || { echo "ERROR: empty frontier model id" >&2; exit 2; }
  QS_DRAFT_ENDPOINT="frontier"
}

select_pdf_draft_model() {
  if [ "$QS_DRAFT_MODEL" != "auto" ]; then
    result "draft model: $QS_DRAFT_MODEL (endpoint=$QS_DRAFT_ENDPOINT)"
    return 0
  fi
  if [ "$QS_DRAFT_ENDPOINT" = "frontier" ]; then
    select_frontier_model
    return 0
  fi

  case "$QS_MODEL_SELECTION" in
    auto)
      if pdf_bench_has_runs; then
        write_pdf_bench_recommendation
        select_model_from_benchmark_json
        return 0
      fi
      if ! is_interactive && ! is_yes_value "$QS_ASSUME_YES"; then
        echo "ERROR: no model benchmark found at $(rel_path "$QS_PDF_MODEL_BENCH_DATA/run-eval")" >&2
        echo "Run one of:" >&2
        echo "  make quickstart-goldset QUICKSTART_RUN_SECURITY=0" >&2
        echo "  QUICKSTART_DRAFT_MODEL=<local-model-id> make quickstart-pdf-corpus" >&2
        exit 2
      fi
      ;;
    benchmark)
      run_pdf_model_benchmark
      write_pdf_bench_recommendation
      select_model_from_benchmark_json
      return 0
      ;;
    choose)
      select_manual_local_model
      return 0
      ;;
    external|frontier)
      select_frontier_model
      return 0
      ;;
    *)
      echo "ERROR: QUICKSTART_MODEL_SELECTION must be auto, benchmark, choose, or frontier" >&2
      exit 2
      ;;
  esac

  printf '[prompt] No benchmark summary found. Choose model-selection mode:\n'
  printf '  1) run local benchmark now (recommended)\n'
  printf '  2) select a local model manually\n'
  printf '  3) use a frontier model through litellm\n'
  local choice
  choice="$(prompt_value "Enter 1, 2, or 3:")"
  case "$choice" in
    1|"")
      run_pdf_model_benchmark
      write_pdf_bench_recommendation
      select_model_from_benchmark_json
      ;;
    2) select_manual_local_model ;;
    3) select_frontier_model ;;
    *) echo "ERROR: unsupported model-selection choice: $choice" >&2; exit 2 ;;
  esac
}

pdf_draft_doc_ids() {
  if [ "$QS_PDF_DRAFT_DOCS" = "all" ]; then
    find "$QS_PDF_MD" -maxdepth 1 -type f -name '*.md' -printf '%f\n' \
      | sed 's/\.md$//' \
      | sort
  else
    printf '%s\n' $QS_PDF_DRAFT_DOCS
  fi
}

stage_pdf_draft_corpus() {
  if [ "$QS_PDF_DRAFT_MD" = "$QS_PDF_MD" ]; then
    echo "ERROR: QUICKSTART_PDF_DRAFT_MD must differ from QUICKSTART_PDF_MD" >&2
    exit 2
  fi
  rm -rf "$QS_PDF_DRAFT_MD"
  mkdir -p "$QS_PDF_DRAFT_MD"
  local doc n=0
  while IFS= read -r doc; do
    test -n "$doc" || continue
    test -f "$QS_PDF_MD/$doc.md" || { echo "ERROR: missing $QS_PDF_MD/$doc.md" >&2; exit 1; }
    test -f "$QS_PDF_MD/$doc.citations.json" || {
      echo "ERROR: missing $QS_PDF_MD/$doc.citations.json" >&2
      exit 1
    }
    cp -R "$QS_PDF_MD/$doc.md" "$QS_PDF_MD/$doc.citations.json" "$QS_PDF_DRAFT_MD/"
    n=$((n + 1))
    printf '[draft-corpus] staged %s\n' "$doc"
  done < <(pdf_draft_doc_ids)
  if [ "$n" -eq 0 ]; then
    echo "ERROR: no markdown documents found under $(rel_path "$QS_PDF_MD")" >&2
    exit 1
  fi
  result "draft input corpus: $(rel_path "$QS_PDF_DRAFT_MD") ($n docs)"
}

pdf_draft_stats() {
  local docs chars chunk windows calls tok_s
  docs="$(find "$QS_PDF_DRAFT_MD" -maxdepth 1 -type f -name '*.md' | wc -l | tr -d ' ')"
  chars="$(find "$QS_PDF_DRAFT_MD" -maxdepth 1 -type f -name '*.md' -print0 \
    | xargs -0 wc -c 2>/dev/null \
    | awk 'END {print $1 + 0}')"
  chunk="${QS_DRAFT_EXTRACT_MAX_CHARS:-12000}"
  windows=$(( (chars + chunk - 1) / chunk ))
  calls=$(( windows + QS_DRAFT_MAX_ITEMS ))
  tok_s=0
  if [ -f "$(pdf_bench_json)" ] && [ "$QS_DRAFT_ENDPOINT" = "local" ]; then
    tok_s="$(quickstart_py speed "$(pdf_bench_json)" "$QS_DRAFT_MODEL")"
  fi
  if [ "${tok_s%.*}" = "0" ]; then
    case "$QS_DRAFT_MODEL" in
      *12B*|*12b*) tok_s=24 ;;
      *27B*|*27b*|*31b*|*35b*) tok_s=8 ;;
      *24b*|*26b*) tok_s=12 ;;
      *) tok_s=16 ;;
    esac
  fi
  local seconds hours
  seconds="$(awk -v calls="$calls" -v tok="$tok_s" 'BEGIN {printf "%.0f", calls * 500 / tok * 2}')"
  hours="$(awk -v sec="$seconds" 'BEGIN {printf "%.1f", sec / 3600}')"
  printf '%s docs, %s chars, about %s extraction windows + %s draft calls, %s hours' \
    "$docs" "$chars" "$windows" "$QS_DRAFT_MAX_ITEMS" "$hours"
}

QS_ROOT="$(resolve_path "${QUICKSTART_ROOT:-$DATA_DIR}")"
QS_LOG_DIR="$(resolve_path "${QUICKSTART_LOG_DIR:-$DATA_DIR/llb/logs/quickstart}")"
QS_UV_CACHE_DIR="$(resolve_path "${QUICKSTART_UV_CACHE_DIR:-$QS_ROOT/uv-cache}")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$QS_UV_CACHE_DIR}"
llb_export_uv_link_mode

QS_A_DATA="$(resolve_path "${QUICKSTART_A_DATA_DIR:-$QS_ROOT/quickstart-leaderboard}")"
QS_A_GOLDSET="$(resolve_path "${QUICKSTART_A_GOLDSET:-samples/goldsets/ua_squad_postedited_v1/goldset.jsonl}")"
QS_A_CORPUS="$(resolve_path "${QUICKSTART_A_CORPUS:-samples/goldsets/ua_squad_postedited_v1/corpus}")"
QS_A_SWEEP_ID="${QUICKSTART_A_SWEEP_ID:-qs-committed}"
QS_MODELS_MANIFEST="$(resolve_path "${MODELS_MANIFEST:-samples/models_uk.yaml}")"
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
QS_SECURITY_CASES="$(resolve_path "${QUICKSTART_SECURITY_CASES:-samples/security_cases_uk.json}")"
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
QS_DRAFT_BASE_URL="${QUICKSTART_DRAFT_BASE_URL:-}"
QS_DRAFT_MAX_ITEMS="${QUICKSTART_DRAFT_MAX_ITEMS:-180}"
QS_DRAFT_VERIFY_N="${QUICKSTART_DRAFT_VERIFY_N:-40}"
QS_DRAFT_TIMEOUT="${QUICKSTART_DRAFT_TIMEOUT:-900}"
QS_DRAFT_MAX_TOKENS="${QUICKSTART_DRAFT_MAX_TOKENS:-4096}"
QS_DRAFT_TEMPERATURE="${QUICKSTART_DRAFT_TEMPERATURE:-0}"
QS_DRAFT_EXTRACT_MAX_CHARS="${QUICKSTART_DRAFT_EXTRACT_MAX_CHARS:-}"
QS_DRAFT_EXTRACT_CHUNK_OVERLAP="${QUICKSTART_DRAFT_EXTRACT_CHUNK_OVERLAP:-}"
QS_MODEL_SELECTION="${QUICKSTART_MODEL_SELECTION:-auto}"
QS_ASSUME_YES="${QUICKSTART_ASSUME_YES:-0}"
QS_PDF_MIN_CHARS="${QUICKSTART_PDF_MIN_CHARS:-500}"
QS_PDF_PARSER="${QUICKSTART_PDF_PARSER:-auto}"

summarize_serving_configs() {
  local tier_json
  tier_json="$(latest_serving_tier_json)"
  if [ -n "$tier_json" ]; then
    result "serving target index: $(rel_path "$tier_json")"
    grep -E '"target"|"backend"|"model"' "$tier_json" | sed 's/^/[serving] /'
  fi
}

latest_serving_tier_json() {
  local tier expected line
  tier="$QS_GPU_GB"
  if [ -z "$tier" ] && [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    line="$("$PROJECT_ROOT/.venv/bin/python" -m llb.main detect-gpu-vram 2>/dev/null || true)"
    case "$line" in
      gpu_tier=*)
        tier="${line#gpu_tier=}"
        tier="${tier%% *}"
        ;;
    esac
  fi
  if [ -n "$tier" ]; then
    expected="$QS_A_DATA/llb/serving/gpu-${tier}gb/tier.json"
    if [ -f "$expected" ]; then
      printf '%s\n' "$expected"
      return 0
    fi
  fi
  find "$QS_A_DATA/llb/serving" -maxdepth 2 -name tier.json -print 2>/dev/null | sort | tail -n 1 || true
}

ensure_goldset_venv() {
  case "$QS_SETUP_VENV" in
    0|false|no)
      test -x "$PROJECT_ROOT/.venv/bin/python" || {
        echo "ERROR: QUICKSTART_SETUP_VENV=$QS_SETUP_VENV but .venv is missing" >&2
        echo "Run make venv or rerun with QUICKSTART_SETUP_VENV=1." >&2
        exit 1
      }
      result "reusing existing .venv; setup disabled by QUICKSTART_SETUP_VENV=$QS_SETUP_VENV"
      ;;
    auto)
      if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
        result "reusing existing .venv; set QUICKSTART_SETUP_VENV=1 to refresh dependencies"
      else
        make_cmd venv SKIP_APT="$QS_SKIP_APT"
      fi
      ;;
    1|true|yes)
      make_cmd venv SKIP_APT="$QS_SKIP_APT"
      ;;
    *)
      echo "ERROR: QUICKSTART_SETUP_VENV must be auto, 1, or 0 (got $QS_SETUP_VENV)" >&2
      exit 2
      ;;
  esac
}

track_a_setup() {
  heading "1/3" "prepare environment"
  result "uv cache: $(rel_path "$UV_CACHE_DIR")"
  ensure_goldset_venv

  heading "2/3" "detect CUDA tier"
  make_with_data_dir "$QS_A_DATA" detect-gpu-vram

  heading "3/3" "generate serving configs"
  if [ -n "$QS_GPU_GB" ]; then
    make_with_data_dir "$QS_A_DATA" gen-serving-config GPU_GB="$QS_GPU_GB"
  else
    make_with_data_dir "$QS_A_DATA" gen-serving-config
  fi
  summarize_serving_configs
  result "goldset quickstart setup artifacts: $(rel_path "$QS_A_DATA")"
}

track_a_rag() {
  heading "1/2" "build committed-goldset FAISS index"
  HF_HUB_OFFLINE="$QS_HF_HUB_OFFLINE" make_with_data_dir "$QS_A_DATA" build-index \
    CORPUS="$QS_A_CORPUS"

  heading "2/2" "validate retrieval gate"
  HF_HUB_OFFLINE="$QS_HF_HUB_OFFLINE" make_with_data_dir "$QS_A_DATA" validate-retrieval \
    GOLDSET="$QS_A_GOLDSET" \
    RAG_K="$QS_RAG_K"
  result "RAG artifacts: $(rel_path "$QS_A_DATA/llb/rag")"
}

track_a_models() {
  heading "1/3" "list runnable model candidates for this host"
  make_with_data_dir "$QS_A_DATA" list-models MODELS_MANIFEST="$QS_MODELS_MANIFEST"

  heading "2/3" "prepare candidate model weights"
  if [ "$QS_PREP_MODELS" = "0" ]; then
    result "skipped prep-models because QUICKSTART_PREP_MODELS=0"
  else
    make_with_data_dir "$QS_A_DATA" prep-models MODELS_MANIFEST="$QS_MODELS_MANIFEST"
    result "model stores are managed by their backends; planner manifest: $(rel_path "$QS_MODELS_MANIFEST")"
  fi

  heading "3/3" "prepare generated CUDA-tier serving targets"
  if [ "$QS_PREP_SERVING_TARGETS" = "0" ]; then
    result "skipped prep-serving-targets because QUICKSTART_PREP_SERVING_TARGETS=0"
  else
    local tier_json
    tier_json="$(latest_serving_tier_json)"
    test -n "$tier_json" || {
      echo "ERROR: no generated serving tier.json found under $QS_A_DATA/llb/serving" >&2
      exit 1
    }
    make_with_data_dir "$QS_A_DATA" prep-serving-targets SERVING_TIER_JSON="$tier_json"
    result "serving target models prepared from: $(rel_path "$tier_json")"
  fi
}

track_a_eval() {
  heading "1/2" "run model-family sweep"
  if [ "$QS_RUN_SWEEP" = "0" ]; then
    result "skipped sweep because QUICKSTART_RUN_SWEEP=0"
  else
    make_with_data_dir "$QS_A_DATA" sweep \
      SWEEP_ID="$QS_A_SWEEP_ID" \
      MODELS_MANIFEST="$QS_MODELS_MANIFEST" \
      GOLDSET="$QS_A_GOLDSET" \
      SPLIT="$QS_SPLIT" \
      SWEEP_LIMIT="$QS_SWEEP_LIMIT"
    result "sweep cells: $(rel_path "$QS_A_DATA/sweep/$QS_A_SWEEP_ID/cells")"
  fi

  heading "2/3" "run inference-backend platform matrix"
  if [ "$QS_RUN_PLATFORM_MATRIX" = "0" ]; then
    result "skipped platform-matrix because QUICKSTART_RUN_PLATFORM_MATRIX=0"
  else
    HF_HUB_OFFLINE="$QS_HF_HUB_OFFLINE" make_with_data_dir "$QS_A_DATA" platform-matrix \
      PLATFORM_MATRIX_GOLDSET="$QS_A_GOLDSET"
    result "backend comparison runs: $(rel_path "$QS_A_DATA/run-eval")"
  fi

  heading "3/3" "summarize host-adaptive recommendation + comparison chart"
  make_with_data_dir "$QS_A_DATA" recommend RECOMMEND_MIN_CASES="$QS_RECOMMEND_MIN_CASES" || \
    result "recommend skipped (no comparable run bundles yet)"
  result "recommendation summary: $(rel_path "$QS_A_DATA/recommend/summary.md")"
  result "comparison chart: $(rel_path "$QS_A_DATA/recommend/comparison.png")"
}

track_a_security() {
  heading "1/1" "run model security tests as a separate benchmark tier"
  if [ "$QS_RUN_SECURITY" = "0" ]; then
    result "skipped security benchmark because QUICKSTART_RUN_SECURITY=0"
    return 0
  fi
  make_with_data_dir "$QS_A_DATA" bench-security \
    SECURITY_MODEL="$QS_SECURITY_MODEL" \
    SECURITY_BACKEND="$QS_SECURITY_BACKEND" \
    SECURITY_CASES="$QS_SECURITY_CASES" \
    SECURITY_VERIFICATION_REF="$QS_SECURITY_VERIFICATION_REF" \
    SECURITY_DATA_VERIFIED=1
  result "security benchmark artifacts: $(rel_path "$QS_A_DATA/security")"
}

track_a_prompt() {
  heading "1/3" "prepare prompt-system candidates"
  make_with_data_dir "$QS_A_DATA" prompt-system-prepare \
    PROMPT_SYSTEM_CORPUS="$QS_A_CORPUS" \
    PROMPT_SYSTEM_OUT_DIR="$QS_PROMPT_DIR"

  heading "2/3" "summarize prompt candidates"
  make_with_data_dir "$QS_A_DATA" prompt-system-review \
    PROMPT_SYSTEM_RUN_DIR="$QS_PROMPT_DIR" \
    PROMPT_SYSTEM_ACTION=summary

  heading "3/3" "pin, score, and compare when a prompt id is supplied"
  if [ -z "$QS_PROMPT_ID" ]; then
    result "review candidates, then rerun with QUICKSTART_PROMPT_ID=<id>"
    printf '[next] make quickstart-goldset-prompt QUICKSTART_PROMPT_ID=<id>\n'
    return 0
  fi
  make_with_data_dir "$QS_A_DATA" prompt-system-review \
    PROMPT_SYSTEM_RUN_DIR="$QS_PROMPT_DIR" \
    PROMPT_SYSTEM_ACTION=pin \
    PROMPT_SYSTEM_ID="$QS_PROMPT_ID"
  make_with_data_dir "$QS_A_DATA" run-eval \
    GOLDSET="$QS_A_GOLDSET" \
    PROMPT_SYSTEM_ID="$QS_PROMPT_ID" \
    PROMPT_PACKAGE="$QS_PROMPT_DIR"
  make_with_data_dir "$QS_A_DATA" prompt-system-compare
  result "prompt comparison artifacts: $(rel_path "$QS_A_DATA/run-eval")"
}

track_a_all() {
  track_a_setup
  track_a_rag
  track_a_models
  track_a_eval
  track_a_security
  track_a_prompt
  result "goldset quickstart leaderboard artifacts: $(rel_path "$QS_A_DATA")"
  printf '[next] make board DATA_DIR=%s\n' "$(rel_path "$QS_A_DATA")"
  printf '[next] make mlflow DATA_DIR=%s\n' "$(rel_path "$QS_A_DATA")"
}

track_b_convert() {
  heading "1/2" "prepare PDF/OCR environment"
  result "uv cache: $(rel_path "$UV_CACHE_DIR")"
  make_cmd venv SKIP_APT="$QS_SKIP_APT" EXTRAS=pdf-quality

  heading "2/2" "convert PDFs to markdown with citation sidecars"
  make_with_data_dir "$DATA_DIR" pdf-to-markdown \
    PDF_DIR="$QS_PDF_SOURCE" \
    PDF_OUT_DIR="$QS_PDF_MD" \
    PDF_MIN_CHARS="$QS_PDF_MIN_CHARS" \
    PDF_PARSER="$QS_PDF_PARSER"
  result "converted markdown corpus: $(rel_path "$QS_PDF_MD")"
}

track_b_index() {
  heading "1/1" "build full-corpus FAISS index"
  HF_HUB_OFFLINE="$QS_HF_HUB_OFFLINE" make_with_data_dir "$QS_PDF_RAG_DATA" build-index \
    CORPUS="$QS_PDF_MD"
  result "full PDF RAG store: $(rel_path "$QS_PDF_RAG_DATA/llb/rag")"
}

track_b_draft() {
  heading "1/4" "select draft model"
  select_pdf_draft_model
  result "draft model: $QS_DRAFT_MODEL (endpoint=$QS_DRAFT_ENDPOINT)"

  heading "2/4" "stage full draft corpus"
  stage_pdf_draft_corpus

  heading "3/4" "confirm full ontology and goldset draft"
  local stats
  stats="$(pdf_draft_stats)"
  result "estimated draft workload: $stats"
  result "draft outputs include goldset.jsonl, needle_items.jsonl, ontology.json, extraction.jsonl, pdf_ontology_report.json, prompt_dictionary_candidates.jsonl"
  if ! prompt_yes_no "The next draft step is expected to take about ${stats##*, }. Proceed?" "no"; then
    echo "ERROR: full PDF draft was not approved" >&2
    echo "Rerun with QUICKSTART_ASSUME_YES=1 or reduce QUICKSTART_DRAFT_MAX_ITEMS for a bounded probe." >&2
    exit 2
  fi

  heading "4/4" "draft unverified goldset and ontology"
  make_cmd prepare-goldset-draft \
    DRAFT_CORPUS="$QS_PDF_DRAFT_MD" \
    DRAFT_MODEL="$QS_DRAFT_MODEL" \
    DRAFT_ENDPOINT="$QS_DRAFT_ENDPOINT" \
    DRAFT_BASE_URL="$QS_DRAFT_BASE_URL" \
    DRAFT_MAX_ITEMS="$QS_DRAFT_MAX_ITEMS" \
    DRAFT_VERIFY_N="$QS_DRAFT_VERIFY_N" \
    DRAFT_MAX_TOKENS="$QS_DRAFT_MAX_TOKENS" \
    DRAFT_TEMPERATURE="$QS_DRAFT_TEMPERATURE" \
    DRAFT_EXTRACT_MAX_CHARS="$QS_DRAFT_EXTRACT_MAX_CHARS" \
    DRAFT_EXTRACT_CHUNK_OVERLAP="$QS_DRAFT_EXTRACT_CHUNK_OVERLAP" \
    DRAFT_NO_THINK=1 \
    DRAFT_OUT_DIR="$QS_PDF_DRAFT" \
    DRAFT_TIMEOUT="$QS_DRAFT_TIMEOUT"
  result "draft bundle: $(rel_path "$QS_PDF_DRAFT")"
}

track_b_graph() {
  heading "1/1" "build graph from draft ontology extraction"
  make_with_data_dir "$QS_PDF_GRAPH_DATA" build-graph BUNDLE="$QS_PDF_DRAFT"
  result "graph artifacts: $(rel_path "$QS_PDF_GRAPH_DATA/llb/graph")"
}

track_b_validate() {
  heading "1/2" "validate draft goldset structure"
  make_cmd validate-goldset GOLDSET="$QS_PDF_DRAFT/goldset.jsonl" CORPUS="$QS_PDF_DRAFT/corpus"

  heading "2/2" "validate draft retrieval against full PDF index"
  HF_HUB_OFFLINE="$QS_HF_HUB_OFFLINE" make_with_data_dir "$QS_PDF_RAG_DATA" validate-retrieval \
    GOLDSET="$QS_PDF_DRAFT/goldset.jsonl" \
    RAG_K="$QS_RAG_K"
  result "validation uses draft goldset: $(rel_path "$QS_PDF_DRAFT/goldset.jsonl")"
}

track_b_review() {
  heading "1/1" "review verification sample"
  make_cmd verify-review VERIFY_WS="$QS_PDF_DRAFT/verify_sample.csv"
  result "review worksheet: $(rel_path "$QS_PDF_DRAFT/verify_sample.csv")"
}

track_b_accept() {
  heading "1/1" "emit accepted ledger after human review"
  make_cmd verify-accept BUNDLE="$QS_PDF_DRAFT" VERIFY_WS="$QS_PDF_DRAFT/verify_sample.csv"
  result "accepted ledger: $(rel_path "$QS_PDF_ACCEPTED")"
}

track_b_after_accept() {
  test -f "$QS_PDF_ACCEPTED/goldset.jsonl" || {
    echo "ERROR: accepted ledger not found at $(rel_path "$QS_PDF_ACCEPTED/goldset.jsonl")" >&2
    echo "Run make quickstart-pdf-corpus-review and make quickstart-pdf-corpus-accept after human review." >&2
    exit 1
  }
  heading "1/3" "build accepted-ledger RAG index"
  HF_HUB_OFFLINE="$QS_HF_HUB_OFFLINE" make_with_data_dir "$QS_PDF_LEADERBOARD_DATA" build-index \
    CORPUS="$QS_PDF_ACCEPTED/corpus"

  heading "2/3" "validate accepted-ledger retrieval"
  HF_HUB_OFFLINE="$QS_HF_HUB_OFFLINE" make_with_data_dir "$QS_PDF_LEADERBOARD_DATA" validate-retrieval \
    GOLDSET="$QS_PDF_ACCEPTED/goldset.jsonl" \
    RAG_K="$QS_RAG_K"

  heading "3/3" "run accepted-ledger model sweep"
  make_with_data_dir "$QS_PDF_LEADERBOARD_DATA" sweep \
    SWEEP_ID=quickstart-pdf-corpus \
    MODELS_MANIFEST="$QS_MODELS_MANIFEST" \
    GOLDSET="$QS_PDF_ACCEPTED/goldset.jsonl" \
    SPLIT="$QS_SPLIT"
  result "accepted leaderboard artifacts: $(rel_path "$QS_PDF_LEADERBOARD_DATA")"
}

track_b_all() {
  track_b_convert
  track_b_index
  track_b_draft
  track_b_graph
  track_b_validate
  result "PDF corpus quickstart stopped before scoring because drafted rows are verified=false"
  printf '[next] make quickstart-pdf-corpus-review\n'
  printf '[next] make quickstart-pdf-corpus-accept\n'
  printf '[next] make quickstart-pdf-corpus-score\n'
}

usage() {
  cat <<'EOF'
Usage: scripts/quickstart.sh <target>

Targets:
  goldset                  committed-goldset setup + RAG + model prep + sweep + backend matrix + security + prompts
  goldset-setup            venv, GPU tier detection, serving config generation
  goldset-rag              build and validate committed-goldset RAG index
  goldset-models           list and prepare model candidates
  goldset-eval             sweep model candidates and run backend platform matrix
  goldset-security         run model security tests as a separate benchmark tier
  goldset-prompt           prepare prompt candidates; pin/eval when QUICKSTART_PROMPT_ID is set
  pdf-corpus               PDF corpus conversion + index + draft + graph + validation
  pdf-corpus-convert       PDF to markdown conversion
  pdf-corpus-index         build full PDF-corpus RAG index
  pdf-corpus-draft         select drafter, prepare full draft corpus, and draft unverified goldset
  pdf-corpus-graph         build graph artifacts from the draft bundle
  pdf-corpus-validate      validate draft structure and retrieval
  pdf-corpus-review        interactive human review of verify_sample.csv
  pdf-corpus-accept        emit accepted ledger after review
  pdf-corpus-score         run accepted corpus/goldset through goldset scoring
EOF
}

run_target() {
  local target="$1"
  case "$target" in
    goldset) track_a_all ;;
    goldset-setup) track_a_setup ;;
    goldset-rag) track_a_rag ;;
    goldset-models) track_a_models ;;
    goldset-eval) track_a_eval ;;
    goldset-security) track_a_security ;;
    goldset-prompt) track_a_prompt ;;
    pdf-corpus) track_b_all ;;
    pdf-corpus-convert) track_b_convert ;;
    pdf-corpus-index) track_b_index ;;
    pdf-corpus-draft) track_b_draft ;;
    pdf-corpus-graph) track_b_graph ;;
    pdf-corpus-validate) track_b_validate ;;
    pdf-corpus-review) track_b_review ;;
    pdf-corpus-accept) track_b_accept ;;
    pdf-corpus-score) track_b_after_accept ;;
    help|-h|--help|"") usage ;;
    *) echo "ERROR: unknown quickstart target: $target" >&2; usage >&2; exit 2 ;;
  esac
}

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
