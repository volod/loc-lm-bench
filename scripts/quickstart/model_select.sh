# shellcheck shell=bash
# GPU-tier / benchmark / interactive drafter-model selection.

host_gemma4_field() {
  local field="$1"
  local -a args
  args=(host-gemma4)
  if [ -n "$QS_DRAFT_NUM_CTX" ]; then
    args+=(--min-context-tokens "$QS_DRAFT_NUM_CTX")
  fi
  if [ -n "$QS_GPU_GB" ]; then
    args+=(--gpu-gb "$QS_GPU_GB")
  fi
  args+=("$field")
  quickstart_py "${args[@]}"
}

resolve_quickstart_gpu_tier() {
  local line tier
  if [ -n "$QS_GPU_GB" ]; then
    return 0
  fi
  line="$("$PROJECT_ROOT/.venv/bin/python" -m llb.main detect-gpu-vram 2>/dev/null || true)"
  case "$line" in
    gpu_tier=*"(no GPU detected)"*)
      return 0
      ;;
    gpu_tier=*)
      tier="${line#gpu_tier=}"
      QS_GPU_GB="${tier%% *}"
      ;;
  esac
}

list_local_models() {
  if command -v ollama >/dev/null 2>&1; then
    ollama list | sed 's/^/[ollama] /'
  else
    result "ollama command not found; enter the local endpoint model id manually"
  fi
}

draft_backend_args() {
  printf '%s\n' ollama
  if [ -x "$PROJECT_ROOT/.venv/bin/vllm" ] || command -v vllm >/dev/null 2>&1; then
    printf '%s\n' vllm
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
  if ! prompt_yes_no \
    "The local model benchmark can take roughly 1-4 hours. Proceed?" \
    "no" \
    "Set QUICKSTART_ASSUME_YES=1 to run the benchmark unattended, or set QUICKSTART_MODEL_SELECTION=auto|choose." \
  ; then
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
  local json choice count model backend
  local -a drafter_backends
  json="$(pdf_bench_json)"
  quickstart_py table "$json"
  case "$QS_MODEL_SELECTION" in
    auto|benchmark)
      # drafter (not necessarily recommended_for_host): restrict the pick to local backends this
      # host can serve for the PDF draft endpoint.
      mapfile -t drafter_backends < <(draft_backend_args)
      model="$(quickstart_py drafter "$json" "${drafter_backends[@]}")"
      backend="$(quickstart_py drafter-backend "$json" "${drafter_backends[@]}")"
      if prompt_yes_no "Use recommended local drafter model '$model' (backend=$backend)?" "yes"; then
        QS_DRAFT_MODEL="$model"
        QS_DRAFT_ENDPOINT="local"
        QS_DRAFT_BACKEND="$backend"
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
      QS_DRAFT_BACKEND="$(quickstart_py candidate-backend "$json" "$choice")"
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

select_host_gemma4_model() {
  local target tier util max_len cpu_offload_gb kv_offloading_size_gb
  resolve_quickstart_gpu_tier
  target="$(host_gemma4_field target)"
  tier="$(host_gemma4_field tier-gb)"
  QS_DRAFT_MODEL="$(host_gemma4_field model)"
  QS_DRAFT_BACKEND="$(host_gemma4_field backend)"
  QS_DRAFT_ENDPOINT="local"
  result "host Gemma 4 target: $target (tier=${tier}gb)"
  if [ "$QS_DRAFT_BACKEND" = "vllm" ]; then
    util="$(host_gemma4_field gpu-memory-utilization)"
    max_len="$(host_gemma4_field max-model-len)"
    cpu_offload_gb="$(host_gemma4_field cpu-offload-gb)"
    kv_offloading_size_gb="$(host_gemma4_field kv-offloading-size-gb)"
    if [ -n "$util" ]; then
      QS_DRAFT_VLLM_GPU_MEMORY_UTILIZATION="$util"
    fi
    if [ -n "$max_len" ] && [ -z "$QS_DRAFT_VLLM_MAX_MODEL_LEN" ]; then
      QS_DRAFT_VLLM_MAX_MODEL_LEN="$max_len"
    fi
    if [ -n "$cpu_offload_gb" ] && [ -z "$QS_DRAFT_VLLM_CPU_OFFLOAD_GB" ]; then
      QS_DRAFT_VLLM_CPU_OFFLOAD_GB="$cpu_offload_gb"
    fi
    if [ -n "$kv_offloading_size_gb" ] && [ -z "$QS_DRAFT_VLLM_KV_OFFLOADING_SIZE_GB" ]; then
      QS_DRAFT_VLLM_KV_OFFLOADING_SIZE_GB="$kv_offloading_size_gb"
    fi
    result "host Gemma 4 vLLM settings: max_model_len=$QS_DRAFT_VLLM_MAX_MODEL_LEN gpu_memory_utilization=$QS_DRAFT_VLLM_GPU_MEMORY_UTILIZATION cpu_offload_gb=${QS_DRAFT_VLLM_CPU_OFFLOAD_GB:-0} kv_offloading_size_gb=${QS_DRAFT_VLLM_KV_OFFLOADING_SIZE_GB:-0}"
  fi
}

select_pdf_draft_model() {
  if [ "$QS_DRAFT_MODEL" != "auto" ]; then
    result "draft model: $QS_DRAFT_MODEL (endpoint=$QS_DRAFT_ENDPOINT backend=$QS_DRAFT_BACKEND)"
    return 0
  fi
  if [ "$QS_DRAFT_ENDPOINT" = "frontier" ]; then
    select_frontier_model
    return 0
  fi

  case "$QS_MODEL_SELECTION" in
    auto)
      select_host_gemma4_model
      return 0
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
}
