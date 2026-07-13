# shellcheck shell=bash
# Track A: committed-goldset leaderboard flow.

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

