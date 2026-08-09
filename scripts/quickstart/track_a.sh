# shellcheck shell=bash
# Track A: committed-goldset leaderboard flow.
# Sourced by scripts/quickstart.sh, which loads the siblings below first; this fragment never
# sources them itself, so it declares them for the llb_* call check.
# llb-requires: helpers.sh
# llb-requires: serving.sh

llb_track_a_setup() {
  llb_heading "1/3" "prepare environment"
  llb_result "uv cache: $(llb_rel_path "$UV_CACHE_DIR")"
  llb_ensure_goldset_venv

  llb_heading "2/3" "detect CUDA tier"
  llb_make_with_data_dir "$QS_A_DATA" detect-gpu-vram

  llb_heading "3/3" "generate serving configs"
  if [ -n "$QS_GPU_GB" ]; then
    llb_make_with_data_dir "$QS_A_DATA" gen-serving-config GPU_GB="$QS_GPU_GB"
  else
    llb_make_with_data_dir "$QS_A_DATA" gen-serving-config
  fi
  llb_summarize_serving_configs
  llb_result "goldset quickstart setup artifacts: $(llb_rel_path "$QS_A_DATA")"
}

llb_track_a_rag() {
  llb_heading "1/2" "build committed-goldset FAISS index"
  HF_HUB_OFFLINE="$QS_HF_HUB_OFFLINE" llb_make_with_data_dir "$QS_A_DATA" build-index \
    CORPUS="$QS_A_CORPUS"

  llb_heading "2/2" "validate retrieval gate"
  HF_HUB_OFFLINE="$QS_HF_HUB_OFFLINE" llb_make_with_data_dir "$QS_A_DATA" validate-retrieval \
    GOLDSET="$QS_A_GOLDSET" \
    RAG_K="$QS_RAG_K"
  llb_result "RAG artifacts: $(llb_rel_path "$QS_A_DATA/llb/rag")"
}

llb_track_a_models() {
  llb_heading "1/3" "list runnable model candidates for this host"
  llb_make_with_data_dir "$QS_A_DATA" list-models MODELS_MANIFEST="$QS_MODELS_MANIFEST"

  llb_heading "2/3" "prepare candidate model weights"
  if [ "$QS_PREP_MODELS" = "0" ]; then
    llb_result "skipped prep-models because QUICKSTART_PREP_MODELS=0"
  else
    llb_make_with_data_dir "$QS_A_DATA" prep-models MODELS_MANIFEST="$QS_MODELS_MANIFEST"
    llb_result "model stores are managed by their backends; planner manifest: $(llb_rel_path "$QS_MODELS_MANIFEST")"
  fi

  llb_heading "3/3" "prepare generated CUDA-tier serving targets"
  if [ "$QS_PREP_SERVING_TARGETS" = "0" ]; then
    llb_result "skipped prep-serving-targets because QUICKSTART_PREP_SERVING_TARGETS=0"
  else
    local tier_json
    tier_json="$(llb_latest_serving_tier_json)"
    test -n "$tier_json" || {
      echo "ERROR: no generated serving tier.json found under $QS_A_DATA/llb/serving" >&2
      exit 1
    }
    llb_make_with_data_dir "$QS_A_DATA" prep-serving-targets SERVING_TIER_JSON="$tier_json"
    llb_result "serving target models prepared from: $(llb_rel_path "$tier_json")"
  fi
}

llb_track_a_eval() {
  llb_heading "1/2" "run model-family sweep"
  if [ "$QS_RUN_SWEEP" = "0" ]; then
    llb_result "skipped sweep because QUICKSTART_RUN_SWEEP=0"
  else
    llb_make_with_data_dir "$QS_A_DATA" sweep \
      SWEEP_ID="$QS_A_SWEEP_ID" \
      MODELS_MANIFEST="$QS_MODELS_MANIFEST" \
      GOLDSET="$QS_A_GOLDSET" \
      SPLIT="$QS_SPLIT" \
      SWEEP_LIMIT="$QS_SWEEP_LIMIT"
    llb_result "sweep cells: $(llb_rel_path "$QS_A_DATA/sweep/$QS_A_SWEEP_ID/cells")"
  fi

  llb_heading "2/3" "run inference-backend platform matrix"
  if [ "$QS_RUN_PLATFORM_MATRIX" = "0" ]; then
    llb_result "skipped platform-matrix because QUICKSTART_RUN_PLATFORM_MATRIX=0"
  else
    HF_HUB_OFFLINE="$QS_HF_HUB_OFFLINE" llb_make_with_data_dir "$QS_A_DATA" platform-matrix \
      PLATFORM_MATRIX_GOLDSET="$QS_A_GOLDSET"
    llb_result "backend comparison runs: $(llb_rel_path "$QS_A_DATA/run-eval")"
  fi

  llb_heading "3/3" "summarize host-adaptive recommendation + comparison chart"
  llb_make_with_data_dir "$QS_A_DATA" recommend RECOMMEND_MIN_CASES="$QS_RECOMMEND_MIN_CASES" || \
    llb_result "recommend skipped (no comparable run bundles yet)"
  llb_result "recommendation summary: $(llb_rel_path "$QS_A_DATA/recommend/summary.md")"
  llb_result "comparison chart: $(llb_rel_path "$QS_A_DATA/recommend/comparison.png")"
}

llb_track_a_security() {
  llb_heading "1/1" "run model security tests as a separate benchmark tier"
  if [ "$QS_RUN_SECURITY" = "0" ]; then
    llb_result "skipped security benchmark because QUICKSTART_RUN_SECURITY=0"
    return 0
  fi
  llb_make_with_data_dir "$QS_A_DATA" bench-security \
    SECURITY_MODEL="$QS_SECURITY_MODEL" \
    SECURITY_BACKEND="$QS_SECURITY_BACKEND" \
    SECURITY_CASES="$QS_SECURITY_CASES" \
    SECURITY_VERIFICATION_REF="$QS_SECURITY_VERIFICATION_REF" \
    SECURITY_DATA_VERIFIED=1
  llb_result "security benchmark artifacts: $(llb_rel_path "$QS_A_DATA/security")"
}

llb_track_a_prompt() {
  llb_heading "1/3" "prepare prompt-system candidates"
  llb_make_with_data_dir "$QS_A_DATA" prompt-system-prepare \
    PROMPT_SYSTEM_CORPUS="$QS_A_CORPUS" \
    PROMPT_SYSTEM_OUT_DIR="$QS_PROMPT_DIR"

  llb_heading "2/3" "summarize prompt candidates"
  llb_make_with_data_dir "$QS_A_DATA" prompt-system-review \
    PROMPT_SYSTEM_RUN_DIR="$QS_PROMPT_DIR" \
    PROMPT_SYSTEM_ACTION=summary

  llb_heading "3/3" "pin, score, and compare when a prompt id is supplied"
  if [ -z "$QS_PROMPT_ID" ]; then
    llb_result "review candidates, then rerun with QUICKSTART_PROMPT_ID=<id>"
    printf '[next] make quickstart-goldset-prompt QUICKSTART_PROMPT_ID=<id>\n'
    return 0
  fi
  llb_make_with_data_dir "$QS_A_DATA" prompt-system-review \
    PROMPT_SYSTEM_RUN_DIR="$QS_PROMPT_DIR" \
    PROMPT_SYSTEM_ACTION=pin \
    PROMPT_SYSTEM_ID="$QS_PROMPT_ID"
  llb_make_with_data_dir "$QS_A_DATA" run-eval \
    GOLDSET="$QS_A_GOLDSET" \
    PROMPT_SYSTEM_ID="$QS_PROMPT_ID" \
    PROMPT_PACKAGE="$QS_PROMPT_DIR"
  llb_make_with_data_dir "$QS_A_DATA" prompt-system-compare
  llb_result "prompt comparison artifacts: $(llb_rel_path "$QS_A_DATA/run-eval")"
}

llb_track_a_all() {
  llb_track_a_setup
  llb_track_a_rag
  llb_track_a_models
  llb_track_a_eval
  llb_track_a_security
  llb_track_a_prompt
  llb_result "goldset quickstart leaderboard artifacts: $(llb_rel_path "$QS_A_DATA")"
  printf '[next] make board DATA_DIR=%s\n' "$(llb_rel_path "$QS_A_DATA")"
  printf '[next] make mlflow DATA_DIR=%s\n' "$(llb_rel_path "$QS_A_DATA")"
}

