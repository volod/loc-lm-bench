"""Which shellcheck runs decides what a green shell-lint run MEANS.

The gate resolves exactly one binary -- the pinned `shellcheck-py` wheel at `$LLB_SHELLCHECK` --
and deliberately does NOT fall back to a `shellcheck` on `PATH`. A distro package is releases
behind the pin and misses checks the pinned one applies, so a fallback let the same commit pass
locally and fail in CI. These tests pin that: a PATH binary is never invoked, a missing wheel is a
failure that names the path it looked at, and the `LLB_SHELLCHECK_OPTIONAL` escape hatch still
turns the failure into a skip whose ok line refuses to claim the tree is clean.
"""

import subprocess
from pathlib import Path

from llb.core.paths import PROJECT_ROOT

# Sources the gate the way both entrypoints do, runs the resolution step alone, and prints its
# status plus the ok line the entrypoints would print on success.
_DRIVER = """
set -uo pipefail
LLB_REPORT_PREFIX=test-shell-lint
. "%(root)s/scripts/shared/common.sh"
llb_load_env
. "%(root)s/scripts/shared/shell_lint.sh"
llb_shellcheck_step
printf 'step_status=%%s\\n' "$?"
llb_shell_lint_ok_line
""" % {"root": PROJECT_ROOT}


def _fake_shellcheck(bin_dir: Path, log: Path) -> Path:
    """A shellcheck that finds nothing and records that it was asked to look."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary = bin_dir / "shellcheck"
    binary.write_text(
        f'#!/usr/bin/env bash\nprintf "invoked\\n" >>"{log}"\nexit 0\n',
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary


def _run_step(tmp_path: Path, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": f"{tmp_path / 'path-bin'}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "DATA_DIR": str(tmp_path / "data"),
        **env_overrides,
    }
    return subprocess.run(
        ["bash", "-c", _DRIVER],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )


def _fresh_checkout(root: Path) -> Path:
    """The tracked `*.sh` and nothing else, in a git tree -- a CI checkout, minus the dev box.

    `git init` with no commit is enough: `llb_lintable_shell_scripts` lists new-but-not-ignored
    files, so the copies show up the way a script does before its first commit.
    """
    for relative in _tracked_shell_scripts():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PROJECT_ROOT / relative).read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


def _tracked_shell_scripts() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "*.sh"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return listed.stdout.split()


def test_the_source_directive_scan_is_clean_in_a_checkout_that_has_no_env_file(tmp_path):
    """One verdict per commit, not one per host.

    `shellcheck -x` FOLLOWS a sourced path where it exists and reports SC1091 where it does not, so
    a script that sources an untracked runtime file (`.env`) is silent on a dev box that has one and
    a finding in a fresh checkout -- the split verdict the pinned binary exists to prevent, arriving
    through the tree instead of through the linter. Every such site is annotated
    `# shellcheck source=/dev/null` with its reason, and this is what checks that from the side the
    dev box cannot see.
    """
    fresh = _fresh_checkout(tmp_path / "checkout")
    assert not (fresh / ".env").exists()
    driver = (
        f'set -uo pipefail\n. "{fresh}/scripts/shared/common.sh"\nllb_load_env\n'
        f'. "{fresh}/scripts/shared/shell_lint.sh"\nllb_shellcheck_sources_scan\n'
    )

    result = subprocess.run(
        ["bash", "-c", driver],
        capture_output=True,
        text=True,
        cwd=fresh,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "DATA_DIR": str(tmp_path / "data"),
            "LLB_SHELLCHECK": str(PROJECT_ROOT / ".venv" / "bin" / "shellcheck"),
        },
        check=False,
    )

    assert result.stdout == "", result.stdout


def test_a_shellcheck_on_path_is_never_used(tmp_path):
    """The whole point: a distro binary must not be able to answer for the pinned one."""
    path_log = tmp_path / "path.log"
    _fake_shellcheck(tmp_path / "path-bin", path_log)
    absent = tmp_path / "venv" / "bin" / "shellcheck"

    result = _run_step(tmp_path, LLB_SHELLCHECK=str(absent))

    assert "step_status=1" in result.stdout
    assert not path_log.exists(), "the PATH shellcheck was invoked -- the fallback is back"


def test_a_missing_wheel_names_the_path_it_looked_at(tmp_path):
    """On a host that HAS `shellcheck` on PATH, a bare 'MISSING' reads as a lie."""
    _fake_shellcheck(tmp_path / "path-bin", tmp_path / "path.log")
    absent = tmp_path / "venv" / "bin" / "shellcheck"

    result = _run_step(tmp_path, LLB_SHELLCHECK=str(absent))

    assert f"MISSING at {absent}" in result.stdout
    assert "make venv EXTRAS=dev" in result.stdout


def test_the_resolved_binary_is_the_one_both_scans_run(tmp_path):
    """Both the lint scan and the source-directive scan go through `$LLB_SHELLCHECK`."""
    wheel_log = tmp_path / "wheel.log"
    wheel = _fake_shellcheck(tmp_path / "venv-bin", wheel_log)
    path_log = tmp_path / "path.log"
    _fake_shellcheck(tmp_path / "path-bin", path_log)

    result = _run_step(tmp_path, LLB_SHELLCHECK=str(wheel))

    assert "step_status=0" in result.stdout
    assert wheel_log.read_text(encoding="utf-8").count("invoked") >= 2
    assert not path_log.exists()


def test_the_optional_escape_hatch_still_refuses_to_claim_a_clean_tree(tmp_path):
    """A skip is allowed; a skip that reports the tree as linted is not."""
    absent = tmp_path / "venv" / "bin" / "shellcheck"

    result = _run_step(tmp_path, LLB_SHELLCHECK=str(absent), LLB_SHELLCHECK_OPTIONAL="1")

    assert "step_status=0" in result.stdout
    assert "shellcheck NOT run" in result.stdout


def test_the_shipped_dev_apt_profile_no_longer_carries_shellcheck():
    """The apt row existed only to feed the dropped fallback."""
    packages = (PROJECT_ROOT / "scripts" / "apt" / "dev.packages").read_text(encoding="utf-8")
    listed = [
        stripped for line in packages.splitlines() if (stripped := line.split("#", 1)[0].strip())
    ]

    assert listed == []
