#!/usr/bin/env bash
# Install Debian/Ubuntu packages listed under scripts/apt/*.packages.
#
# Usage:
#   scripts/install_apt_deps.sh production
#   scripts/install_apt_deps.sh dev
#   scripts/install_apt_deps.sh all
#
# Environment:
#   SKIP_APT=1     Skip apt operations (useful in CI/containers without apt).
#   APT_DRY_RUN=1  Print packages that would be installed without calling apt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=shared/common.sh
. "$SCRIPT_DIR/shared/common.sh"

APT_DIR="$PROJECT_ROOT/scripts/apt"
VALID_PROFILES=(production dev all)

usage() {
  cat <<EOF
Usage: $(basename "$0") <production|dev|all>

Install OS packages from scripts/apt/*.packages (Debian/Ubuntu + sudo).
Set SKIP_APT=1 to skip. Set APT_DRY_RUN=1 to list missing packages only.
EOF
}

read_packages() {
  local list_file="$1"
  local line pkg
  if [ ! -f "$list_file" ]; then
    echo "ERROR: package list not found: $list_file" >&2
    exit 1
  fi
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    line="$(printf '%s' "$line" | tr -d '[:space:]')"
    [ -n "$line" ] || continue
    printf '%s\n' "$line"
  done <"$list_file"
}

collect_missing() {
  local list_file="$1"
  local pkg
  while IFS= read -r pkg; do
    if dpkg -s "$pkg" >/dev/null 2>&1; then
      printf '[apt] already installed: %s\n' "$pkg" >&2
    else
      printf '%s\n' "$pkg"
    fi
  done < <(read_packages "$list_file")
}

llb_dpkg_broken_count() {
  dpkg -l 2>/dev/null | awk '$1 ~ /^(iF|iU|iH)$/ {count++} END {print count+0}'
}

llb_warn_dpkg_state() {
  local broken
  broken="$(llb_dpkg_broken_count)"
  if [ "$broken" -gt 0 ]; then
    printf '[apt] WARN: dpkg reports %s unconfigured/broken package(s) on this host\n' "$broken" >&2
    printf '[apt] WARN: apt may fail while configuring unrelated kernel/NVIDIA packages\n' >&2
    printf '[apt] WARN: fix with: sudo dpkg --configure -a (see docs/guides/dev-setup.md)\n' >&2
  fi
}

llb_pkg_installed() {
  dpkg -s "$1" 2>/dev/null | awk -F': ' '/^Status:/ {print $2}' | grep -q 'install ok installed'
}

llb_verify_installed() {
  local profile="$1"
  shift
  local pkg missing=0
  for pkg in "$@"; do
    if llb_pkg_installed "$pkg"; then
      printf '[apt] verified: %s\n' "$pkg"
    else
      printf '[apt] ERROR: %s is not installed after apt run\n' "$pkg" >&2
      missing=1
    fi
  done
  return "$missing"
}

install_profile() {
  local profile="$1"
  local list_file="$APT_DIR/${profile}.packages"
  local -a missing=()
  local -a requested=()
  local pkg

  while IFS= read -r pkg; do
    [ -n "$pkg" ] || continue
    requested+=("$pkg")
    missing+=("$pkg")
  done < <(collect_missing "$list_file")

  if [ "${#missing[@]}" -eq 0 ]; then
    printf '[apt] profile=%s: nothing to install\n' "$profile"
    return 0
  fi

  printf '[apt] profile=%s: missing %s\n' "$profile" "${missing[*]}"
  if [ "${APT_DRY_RUN:-0}" = "1" ]; then
    return 0
  fi

  llb_warn_dpkg_state

  local apt_rc=0
  if [ "$(id -u)" -eq 0 ]; then
    apt-get install -y --no-install-recommends --no-upgrade "${missing[@]}" || apt_rc=$?
  else
    if ! command -v sudo >/dev/null 2>&1; then
      echo "ERROR: need root or sudo to install apt packages: ${missing[*]}" >&2
      exit 1
    fi
    sudo apt-get install -y --no-install-recommends --no-upgrade "${missing[@]}" || apt_rc=$?
  fi

  if llb_verify_installed "$profile" "${requested[@]}"; then
    if [ "$apt_rc" -ne 0 ]; then
      printf '[apt] WARN: apt exited %s but requested profile=%s packages are installed\n' \
        "$apt_rc" "$profile" >&2
      printf '[apt] WARN: repair unrelated dpkg issues when convenient: sudo dpkg --configure -a\n' >&2
    fi
    return 0
  fi

  printf '[apt] ERROR: apt exited %s and one or more profile=%s packages are still missing\n' \
    "$apt_rc" "$profile" >&2
  return 1
}

profile="${1:-}"
if [ -z "$profile" ]; then
  usage >&2
  exit 1
fi

case "$profile" in
  production | dev | all) ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "ERROR: unknown profile: $profile (expected: ${VALID_PROFILES[*]})" >&2
    exit 1
    ;;
esac

if [ "${SKIP_APT:-0}" = "1" ]; then
  printf '[apt] SKIP_APT=1 -- skipping profile=%s\n' "$profile"
  exit 0
fi

if ! command -v apt-get >/dev/null 2>&1; then
  printf '[apt] apt-get not found -- skipping profile=%s on this host\n' "$profile"
  exit 0
fi

if [ "$profile" = "all" ]; then
  install_profile production
  install_profile dev
else
  install_profile "$profile"
fi
