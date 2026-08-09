# shellcheck shell=bash
# Path, make, prompt, and logging helpers shared by every quickstart track.

llb_resolve_path() {
  local value="$1"
  case "$value" in
    /*) printf '%s' "$value" ;;
    *) printf '%s/%s' "$PROJECT_ROOT" "$value" ;;
  esac
}

llb_rel_path() {
  local value="$1"
  case "$value" in
    "$PROJECT_ROOT"/*) printf '%s' "${value#"$PROJECT_ROOT"/}" ;;
    *) printf '%s' "$value" ;;
  esac
}

llb_make_cmd() {
  make -C "$PROJECT_ROOT" --no-print-directory "$@"
}

llb_make_with_data_dir() {
  local data_dir="$1"
  shift
  DATA_DIR="$data_dir" llb_make_cmd "$@"
}

llb_heading() {
  printf '\n### [%s] %s\n' "$1" "$2"
}

llb_result() {
  printf '[llb_result] %s\n' "$1"
}

llb_is_yes_value() {
  case "${1,,}" in
    1|true|yes|y) return 0 ;;
    *) return 1 ;;
  esac
}

llb_is_interactive() {
  [ -t 0 ] && [ -t 1 ]
}

llb_prompt_yes_no() {
  local question="$1"
  local default="${2:-no}"
  local hint="${3:-Set QUICKSTART_ASSUME_YES=1 to approve this non-interactive confirmation.}"
  local suffix answer
  if llb_is_yes_value "$QS_ASSUME_YES"; then
    printf '[prompt] %s yes (QUICKSTART_ASSUME_YES=1)\n' "$question"
    return 0
  fi
  if ! llb_is_interactive; then
    if [ "$default" = "yes" ]; then
      printf '[prompt] %s yes (non-interactive default)\n' "$question"
      return 0
    fi
    echo "ERROR: this step needs confirmation: $question" >&2
    echo "$hint" >&2
    exit 2
  fi
  if [ "$default" = "yes" ]; then
    suffix="[Y/n]"
  else
    suffix="[y/N]"
  fi
  read -r -p "$question $suffix " answer
  answer="${answer:-$default}"
  llb_is_yes_value "$answer"
}

llb_prompt_value() {
  local question="$1"
  local answer
  if ! llb_is_interactive; then
    echo "ERROR: cannot prompt in non-interactive mode: $question" >&2
    exit 2
  fi
  read -r -p "$question " answer
  printf '%s' "$answer"
}

llb_quickstart_py() {
  test -x "$PROJECT_ROOT/.venv/bin/python" || {
    echo "ERROR: .venv missing -- run make venv first" >&2
    exit 1
  }
  "$PROJECT_ROOT/.venv/bin/python" -m llb.quickstart.model_choice "$@"
}

