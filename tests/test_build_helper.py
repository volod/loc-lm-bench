"""The canonical MAX_JOBS shell helper (AGENTS.md single source of truth)."""

import shutil
import subprocess
from pathlib import Path

import pytest

COMMON_SH = Path(__file__).resolve().parents[1] / "scripts" / "shared" / "common.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_max_jobs_is_a_positive_int():
    out = subprocess.run(
        ["bash", "-c", f". {COMMON_SH}; max_jobs"],
        capture_output=True, text=True, check=True,
    )
    value = out.stdout.strip()
    assert value.isdigit() and int(value) >= 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_common_sh_exposes_helpers():
    out = subprocess.run(
        ["bash", "-c", f". {COMMON_SH}; type max_jobs llb_load_env llb_python >/dev/null && echo ok"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "ok"
