# shellcheck shell=bash
# Track C: mixed txt/md/pdf corpus flow (reuses track B stages).
# use_corpus_paths repoints the QS_PDF_* globals (consumed by the sourced track B fragment) at the
# mixed-corpus dirs, so shellcheck cannot see their uses in this file.
# shellcheck disable=SC2034

# --- mixed txt/md/pdf corpus track ------------------------------------------------------------
# Generalizes the PDF track to any mixed corpus via `ingest-corpus`. The index/graph/validate
# stages are identical, so they are reused by pointing the PDF-track paths at the corpus dirs
# (use_corpus_paths). Drafting runs directly over the converted corpus -- passthrough .md/.txt
# have no citation sidecars, so no per-doc staging step is needed.

use_corpus_paths() {
  QS_PDF_MD="$QS_CORPUS_MD"
  QS_PDF_RAG_DATA="$QS_CORPUS_RAG_DATA"
  QS_PDF_DRAFT="$QS_CORPUS_DRAFT"
  QS_PDF_DRAFT_MD="$QS_CORPUS_MD"
  QS_PDF_GRAPH_DATA="$QS_CORPUS_GRAPH_DATA"
}

track_c_convert() {
  heading "1/2" "prepare ingest environment (PDF/OCR extras for mixed corpora)"
  result "uv cache: $(rel_path "$UV_CACHE_DIR")"
  make_cmd venv SKIP_APT="$QS_SKIP_APT" EXTRAS=pdf-quality

  heading "2/2" "ingest mixed txt/md/pdf corpus"
  make_with_data_dir "$DATA_DIR" ingest-corpus \
    CORPUS_ROOT="$QS_CORPUS_SRC" \
    CORPUS_OUT_DIR="$QS_CORPUS_MD" \
    CORPUS_MIN_CHARS="$QS_CORPUS_MIN_CHARS" \
    CORPUS_PARSER="$QS_CORPUS_PARSER"
  result "converted corpus: $(rel_path "$QS_CORPUS_MD")"
}

track_c_draft() {
  heading "1/3" "select draft model"
  select_pdf_draft_model
  result "draft model: $QS_DRAFT_MODEL (endpoint=$QS_DRAFT_ENDPOINT backend=$QS_DRAFT_BACKEND)"

  heading "2/3" "confirm full ontology and goldset draft"
  local stats draft_egress_consent
  stats="$(pdf_draft_stats)"
  result "estimated draft workload: $stats"
  if [ -n "$QS_CORPUS_RESUME" ]; then
    result "resuming interrupted bundle: $(rel_path "$QS_CORPUS_RESUME")"
  fi
  if ! prompt_yes_no \
    "The next draft step is expected to take about ${stats##*, }. Proceed?" \
    "no" \
    "Rerun with QUICKSTART_ASSUME_YES=1 make quickstart-corpus, or reduce QUICKSTART_DRAFT_MAX_ITEMS for a bounded probe." \
  ; then
    echo "ERROR: full corpus draft was not approved" >&2
    echo "Rerun with QUICKSTART_ASSUME_YES=1 or reduce QUICKSTART_DRAFT_MAX_ITEMS for a bounded probe." >&2
    exit 2
  fi

  draft_egress_consent=0
  if [ "$QS_DRAFT_ENDPOINT" = "frontier" ]; then
    if ! prompt_yes_no \
      "Send corpus '$QS_CORPUS_MD' to Litellm destination '$QS_DRAFT_MODEL' (max calls: $QUICKSTART_DRAFT_MAX_CALLS)?" \
      "no" \
      "Set QUICKSTART_ASSUME_YES=1 only after approving this corpus egress and provider spend." \
    ; then
      echo "ERROR: frontier corpus egress was not approved" >&2
      exit 2
    fi
    draft_egress_consent=1
  fi

  heading "3/3" "draft unverified goldset and ontology"
  make_cmd prepare-goldset-draft \
    DRAFT_CORPUS="$QS_CORPUS_MD" \
    DRAFT_MODEL="$QS_DRAFT_MODEL" \
    DRAFT_ENDPOINT="$QS_DRAFT_ENDPOINT" \
    DRAFT_EGRESS_CONSENT="$draft_egress_consent" \
    DRAFT_MAX_USD="$QUICKSTART_DRAFT_MAX_USD" \
    DRAFT_MAX_CALLS="$QUICKSTART_DRAFT_MAX_CALLS" \
    DRAFT_BACKEND="$QS_DRAFT_BACKEND" \
    DRAFT_BASE_URL="$QS_DRAFT_BASE_URL" \
    DRAFT_MAX_ITEMS="$QS_DRAFT_MAX_ITEMS" \
    DRAFT_VERIFY_N="$QS_DRAFT_VERIFY_N" \
    DRAFT_MAX_TOKENS="$QS_DRAFT_MAX_TOKENS" \
    DRAFT_TEMPERATURE="$QS_DRAFT_TEMPERATURE" \
    DRAFT_EXTRACT_MAX_CHARS="$QS_DRAFT_EXTRACT_MAX_CHARS" \
    DRAFT_EXTRACT_CHUNK_OVERLAP="$QS_DRAFT_EXTRACT_CHUNK_OVERLAP" \
    DRAFT_CONCURRENCY="$QS_DRAFT_CONCURRENCY" \
    DRAFT_NO_THINK=1 \
    DRAFT_NUM_CTX="$QS_DRAFT_NUM_CTX" \
    DRAFT_VLLM_PORT="$QS_DRAFT_VLLM_PORT" \
    DRAFT_VLLM_GPU_MEMORY_UTILIZATION="$QS_DRAFT_VLLM_GPU_MEMORY_UTILIZATION" \
    DRAFT_VLLM_MAX_MODEL_LEN="$QS_DRAFT_VLLM_MAX_MODEL_LEN" \
    DRAFT_VLLM_CPU_OFFLOAD_GB="$QS_DRAFT_VLLM_CPU_OFFLOAD_GB" \
    DRAFT_VLLM_KV_OFFLOADING_SIZE_GB="$QS_DRAFT_VLLM_KV_OFFLOADING_SIZE_GB" \
    DRAFT_VLLM_DTYPE="$QS_DRAFT_VLLM_DTYPE" \
    DRAFT_VLLM_QUANTIZATION="$QS_DRAFT_VLLM_QUANTIZATION" \
    DRAFT_VLLM_STARTUP_TIMEOUT="$QS_DRAFT_VLLM_STARTUP_TIMEOUT" \
    DRAFT_RETRIEVAL_INDEX_DIR="$QS_CORPUS_RAG_DATA/llb/rag" \
    DRAFT_RETRIEVAL_K="$QS_RAG_K" \
    DRAFT_REQUIRE_PASSED_GATES=1 \
    DRAFT_OUT_DIR="$QS_CORPUS_DRAFT" \
    DRAFT_RESUME="$QS_CORPUS_RESUME" \
    DRAFT_TIMEOUT="$QS_DRAFT_TIMEOUT"
  result "draft bundle: $(rel_path "$QS_CORPUS_DRAFT")"
}

track_c_all() {
  use_corpus_paths
  track_c_convert
  track_b_index
  track_c_draft
  track_b_graph
  track_b_validate
  result "mixed-corpus quickstart stopped before scoring because drafted rows are verified=false"
  printf '[next] an interrupted draft resumes with QUICKSTART_CORPUS_RESUME=%s make quickstart-corpus-draft\n' "$(rel_path "$QS_CORPUS_DRAFT")"
  printf '[next] make quickstart-pdf-corpus-review QUICKSTART_PDF_DRAFT=%s\n' "$(rel_path "$QS_CORPUS_DRAFT")"
}
