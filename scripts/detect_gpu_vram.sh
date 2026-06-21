#!/usr/bin/env bash
# Print supported GPU VRAM tier (12 / 16 / 24 / 32 GiB) for this host.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/shared/common.sh
source "$ROOT/scripts/shared/common.sh"
llb_load_env
exec "$(llb_python)" -m llb.main detect-gpu-vram "$@"
