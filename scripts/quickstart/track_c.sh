# shellcheck shell=bash
# Track C: mixed txt/md/pdf corpus flow (reuses track B stages).
# Sourced by scripts/quickstart.sh, which loads the siblings below first; this fragment never
# sources them itself, so it declares them for the llb_* call check.
# llb-requires: helpers.sh
# llb-requires: model_select.sh
# llb-requires: pdf_draft.sh
# llb-requires: track_b.sh
#
# llb_use_corpus_paths repoints the QS_PDF_* globals (consumed by the sourced track B fragment) at
# the mixed-corpus dirs, so shellcheck cannot see their uses in this file.
# shellcheck disable=SC2034

# --- mixed txt/md/pdf corpus track ------------------------------------------------------------
# Generalizes the PDF track to any mixed corpus via `ingest-corpus`. The index/graph/validate
# stages are identical, so they are reused by pointing the PDF-track paths at the corpus dirs
# (llb_use_corpus_paths). Drafting runs directly over the converted corpus -- passthrough .md/.txt
# have no citation sidecars, so no per-doc staging step is needed.

llb_use_corpus_paths() {
  QS_PDF_MD="$QS_CORPUS_MD"
  QS_PDF_RAG_DATA="$QS_CORPUS_RAG_DATA"
  QS_PDF_DRAFT="$QS_CORPUS_DRAFT"
  QS_PDF_DRAFT_MD="$QS_CORPUS_MD"
  QS_PDF_GRAPH_DATA="$QS_CORPUS_GRAPH_DATA"
}

llb_track_c_convert() {
  llb_heading "1/2" "prepare ingest environment (PDF/OCR extras for mixed corpora)"
  llb_result "uv cache: $(llb_rel_path "$UV_CACHE_DIR")"
  llb_make_cmd venv SKIP_APT="$QS_SKIP_APT" EXTRAS=pdf-quality

  llb_heading "2/2" "ingest mixed txt/md/pdf corpus"
  llb_make_with_data_dir "$DATA_DIR" ingest-corpus \
    CORPUS_ROOT="$QS_CORPUS_SRC" \
    CORPUS_OUT_DIR="$QS_CORPUS_MD" \
    CORPUS_MIN_CHARS="$QS_CORPUS_MIN_CHARS" \
    CORPUS_PARSER="$QS_CORPUS_PARSER"
  llb_result "converted corpus: $(llb_rel_path "$QS_CORPUS_MD")"
}

llb_track_c_draft() {
  llb_heading "1/3" "select draft model"
  llb_select_pdf_draft_model
  llb_result "draft model: $QS_DRAFT_MODEL (endpoint=$QS_DRAFT_ENDPOINT backend=$QS_DRAFT_BACKEND)"

  llb_heading "2/3" "confirm full ontology and goldset draft"
  local stats draft_egress_consent
  stats="$(llb_pdf_draft_stats)"
  llb_result "estimated draft workload: $stats"
  if [ -n "$QS_CORPUS_RESUME" ]; then
    llb_result "resuming interrupted bundle: $(llb_rel_path "$QS_CORPUS_RESUME")"
  fi
  if ! llb_prompt_yes_no \
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
    if ! llb_prompt_yes_no \
      "Send corpus '$QS_CORPUS_MD' to Litellm destination '$QS_DRAFT_MODEL' (max calls: $QUICKSTART_DRAFT_MAX_CALLS)?" \
      "no" \
      "Set QUICKSTART_ASSUME_YES=1 only after approving this corpus egress and provider spend." \
    ; then
      echo "ERROR: frontier corpus egress was not approved" >&2
      exit 2
    fi
    draft_egress_consent=1
  fi

  llb_heading "3/3" "draft unverified goldset and ontology"
  llb_make_cmd prepare-goldset-draft \
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
    DRAFT_VERIFY_DERIVE=1 \
    DRAFT_VERIFY_CONFIDENCE="$QS_DRAFT_VERIFY_CONFIDENCE" \
    DRAFT_VERIFY_PRECISION="$QS_DRAFT_VERIFY_PRECISION" \
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
  llb_result "draft bundle: $(llb_rel_path "$QS_CORPUS_DRAFT")"
}

llb_track_c_all() {
  llb_use_corpus_paths
  llb_track_c_convert
  llb_track_b_index
  llb_track_c_draft
  llb_track_b_graph
  llb_track_b_validate
  llb_result "mixed-corpus quickstart stopped before scoring because drafted rows are verified=false"
  printf '[next] an interrupted draft resumes with QUICKSTART_CORPUS_RESUME=%s make quickstart-corpus-draft\n' "$(llb_rel_path "$QS_CORPUS_DRAFT")"
  printf '[next] make quickstart-pdf-corpus-review QUICKSTART_PDF_DRAFT=%s\n' "$(llb_rel_path "$QS_CORPUS_DRAFT")"
}
