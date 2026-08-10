"""The maintainability-index scan is deliberately narrower than a repository sweep.

MI combines complexity and module volume. The latter already has a soft project rule, so a
C-grade MI row remains an informational code-quality finding rather than a CI failure. These tests
pin both parts of that decision: only ``src`` and ``tests`` are scanned, and printed findings do
not turn the report non-zero.
"""

import os
import subprocess
from pathlib import Path

from llb.core.paths import PROJECT_ROOT

_DRIVER = f"""
set -euo pipefail
LLB_REPORT_PREFIX=test-maintainability
. "{PROJECT_ROOT}/scripts/shared/common.sh"
. "{PROJECT_ROOT}/scripts/shared/complexity.sh"
llb_maintainability_report
printf 'report_status=0\\n'
"""


def _fake_radon(tmp_path: Path) -> tuple[Path, Path]:
    log = tmp_path / "radon.args"
    binary = tmp_path / "radon"
    binary.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >"{log}"\nprintf "%s\\n" "$LLB_TEST_OUTPUT"\n',
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary, log


def _run_report(tmp_path: Path, output: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    radon, log = _fake_radon(tmp_path)
    env = {
        **os.environ,
        "DATA_DIR": str(tmp_path / "data"),
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "LLB_MI_MIN_GRADE": "C",
        "LLB_RADON": str(radon),
        "LLB_TEST_OUTPUT": output,
    }
    result = subprocess.run(
        ["bash", "-c", _DRIVER],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    return result, log


def test_maintainability_report_scans_only_source_and_tests(tmp_path):
    result, log = _run_report(tmp_path, "")

    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8") == "mi src tests -s -n C -x C\n"


def test_maintainability_finding_is_informational(tmp_path):
    finding = "tests/example.py - C (9.99)"
    result, _ = _run_report(tmp_path, finding)

    assert result.returncode == 0, result.stderr
    assert "report_status=0" in result.stdout
    assert "maintainability index grade C only (src tests only; informational)" in result.stdout
    assert finding in result.stdout
