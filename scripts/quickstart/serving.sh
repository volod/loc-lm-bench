# shellcheck shell=bash
# Serving-config summaries and the goldset venv bootstrap.

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

