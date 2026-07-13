# shellcheck shell=bash
# Track B: PDF corpus conversion -> draft -> validation flow.

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
  result "draft model: $QS_DRAFT_MODEL (endpoint=$QS_DRAFT_ENDPOINT backend=$QS_DRAFT_BACKEND)"

  heading "2/4" "stage full draft corpus"
  stage_pdf_draft_corpus

  heading "3/4" "confirm full ontology and goldset draft"
  local stats
  stats="$(pdf_draft_stats)"
  result "estimated draft workload: $stats"
  result "draft outputs include goldset.jsonl, needle_items.jsonl, ontology.json, extraction.jsonl, pdf_ontology_report.json, prompt_dictionary_candidates.jsonl"
  if ! prompt_yes_no \
    "The next draft step is expected to take about ${stats##*, }. Proceed?" \
    "no" \
    "Rerun with QUICKSTART_ASSUME_YES=1 make quickstart-pdf-corpus, or reduce QUICKSTART_DRAFT_MAX_ITEMS for a bounded probe." \
  ; then
    echo "ERROR: full PDF draft was not approved" >&2
    echo "Rerun with QUICKSTART_ASSUME_YES=1 or reduce QUICKSTART_DRAFT_MAX_ITEMS for a bounded probe." >&2
    exit 2
  fi

  heading "4/4" "draft unverified goldset and ontology"
  make_cmd prepare-goldset-draft \
    DRAFT_CORPUS="$QS_PDF_DRAFT_MD" \
    DRAFT_MODEL="$QS_DRAFT_MODEL" \
    DRAFT_ENDPOINT="$QS_DRAFT_ENDPOINT" \
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
    DRAFT_RETRIEVAL_INDEX_DIR="$QS_PDF_RAG_DATA/llb/rag" \
    DRAFT_RETRIEVAL_K="$QS_RAG_K" \
    DRAFT_REQUIRE_PASSED_GATES=1 \
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
