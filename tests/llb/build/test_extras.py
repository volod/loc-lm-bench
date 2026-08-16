"""Installing an optional extra must not re-resolve the venv off `uv.lock` (llb.build.extras).

`uv sync` pins the venv to the same versions GitHub CI installs; a follow-up
`uv pip install -e ".[review]"` re-resolves the whole requirement set and takes the newest version
each specifier admits, so it walks packages back off the lock -- `ruff` and `mypy` among them,
which is a local `make ci` running a different linter and type checker than the workflow. These
cover the constrained install path, the off-lock report that names what to re-install, and the
end-to-end uv invocation the make target actually issues.
"""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from llb.build import extras, lock_guard, lock_reader
from llb.core import env
from llb.core.paths import PROJECT_ROOT

INSTALL_EXTRAS_SH = PROJECT_ROOT / "scripts" / "install_extras.sh"

LOCK_FIXTURE = """
version = 1

[[package]]
name = "ruff"
version = "0.15.18"

[[package]]
name = "textual"
version = "8.2.7"

[[package]]
name = "numpy"
version = "2.4.6"
"""

PYPROJECT_FIXTURE = """
[project]
name = "llb"
dependencies = ["numpy>=2.0"]

[project.optional-dependencies]
dev = ["ruff>=0.6,<0.17", "textual>=0.60"]
review = ["textual>=0.60"]
finetune = ["trl>=0.9"]
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / lock_reader.LOCK_FILENAME).write_text(LOCK_FIXTURE, encoding="utf-8")
    (tmp_path / lock_reader.PYPROJECT_FILENAME).write_text(PYPROJECT_FIXTURE, encoding="utf-8")
    return tmp_path


@pytest.fixture
def groups(project: Path) -> dict[str, set[str]]:
    return lock_reader.declared_groups(project / lock_reader.PYPROJECT_FILENAME)


def test_declared_groups_splits_base_from_every_extra(groups):
    assert groups == {
        lock_reader.BASE_GROUP: {"numpy"},
        "dev": {"ruff", "textual"},
        "review": {"textual"},
        "finetune": {"trl"},
    }
    assert extras.extra_names(groups) == {"dev", "review", "finetune"}


def test_parse_groups_drops_blanks_and_duplicates():
    assert extras.parse_groups(" review , pdf-quality ,, review ") == ("review", "pdf-quality")
    assert extras.parse_groups("") == ()


def test_project_requirement_builds_the_editable_spec():
    assert extras.project_requirement(("dev", "review")) == ".[dev,review]"
    assert extras.project_requirement(()) == "."  # no extras still installs the project itself


def test_unknown_groups_catches_a_typo(groups):
    # Without this, `.[reveiw]` resolves as the bare project and quietly installs nothing extra.
    assert extras.unknown_groups(("dev", "reveiw"), groups) == ("reveiw",)
    assert extras.unknown_groups(("dev", "review"), groups) == ()
    # The base list is not an installable extra name.
    assert extras.unknown_groups((lock_reader.BASE_GROUP,), groups) == (lock_reader.BASE_GROUP,)


def test_install_args_run_uv_under_the_lock_constraint(tmp_path):
    constraint = tmp_path / "uv-lock-constraints.txt"

    command = extras.install_args(Path("/venv/bin/python"), ("dev", "review"), constraint)

    assert command[:5] == ["uv", "pip", "install", "--python", "/venv/bin/python"]
    assert command[-2:] == ["-e", ".[dev,review]"]
    assert "--constraint" in command and str(constraint) in command


def test_install_args_without_a_constraint_are_a_plain_install():
    command = extras.install_args(Path("/venv/bin/python"), ("dev",), None)

    assert "--constraint" not in command


def test_declaring_groups_names_every_extra_that_owns_the_package(groups):
    assert extras.declaring_groups("textual", groups) == ("dev", "review")
    assert extras.declaring_groups("trl", groups) == ("finetune",)
    assert extras.declaring_groups("numpy", groups) == ()  # base-only, no extra to re-install


def test_remedy_command_covers_every_drifted_package_in_one_install(groups):
    drifts = [
        lock_guard.Drift("ruff", "0.15.20", ("0.15.18",)),
        lock_guard.Drift("trl", "1.8.0", ("1.7.1",)),
    ]

    assert extras.remedy_command(drifts, groups) == "make install-extras EXTRAS=dev,finetune"
    assert extras.remedy_command([], groups) is None


def test_report_lines_name_the_drift_its_group_and_the_fix(groups):
    drifts = [lock_guard.Drift("ruff", "0.15.20", ("0.15.18",))]

    lines = extras.report_lines(drifts, groups)

    assert "1 declared package(s) sit off uv.lock" in lines[0]
    assert "ruff 0.15.20 is off the lock (pinned: 0.15.18) -- declared by dev" in lines[1]
    assert lines[-1] == "put them back with: make install-extras EXTRAS=dev"


def test_report_lines_say_so_when_the_venv_matches(groups):
    assert extras.report_lines([], groups) == ["every declared package matches uv.lock"]


def test_find_off_lock_reads_only_declared_packages(project, monkeypatch):
    monkeypatch.setattr(
        lock_reader,
        "installed_versions",
        lambda names: {"ruff": "0.15.20", "numpy": "2.4.6", "pytest": "9.9.9"},
    )

    drifts, groups = extras.find_off_lock(project)

    # pytest is neither declared nor locked here, so it is none of this report's business.
    assert drifts == [lock_guard.Drift("ruff", "0.15.20", ("0.15.18",))]
    assert "dev" in groups


def test_check_exit_code_follows_the_guard_mode(project, monkeypatch):
    monkeypatch.setattr(lock_reader, "installed_versions", lambda names: {"ruff": "0.15.20"})

    assert extras.check(project, mode=lock_guard.GUARD_REFUSE) == 1
    assert extras.check(project, mode=lock_guard.GUARD_REPORT) == 0

    monkeypatch.setattr(lock_reader, "installed_versions", lambda names: {"ruff": "0.15.18"})
    assert extras.check(project, mode=lock_guard.GUARD_REFUSE) == 0


def test_install_refuses_an_unknown_extra_before_touching_uv(project, monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("uv must not run for an extra pyproject.toml does not declare")

    monkeypatch.setattr(subprocess, "run", _fail)

    with pytest.raises(ValueError, match="no such extra"):
        extras.install(("reveiw",), root=project)


def test_main_reports_a_mistyped_extra_without_a_traceback(caplog):
    """A typo is a usage mistake -- the operator needs the available names, not a stack trace."""
    with caplog.at_level("ERROR"):
        assert extras.main(["--extras", "reveiw"]) == 2

    assert "no such extra" in caplog.text and "review" in caplog.text


def test_install_constrains_the_resolution_and_re_checks(project, monkeypatch):
    monkeypatch.delenv(env.EXTRAS_LOCK_GUARD, raising=False)
    monkeypatch.setattr(lock_reader, "installed_versions", lambda names: {"ruff": "0.15.18"})
    seen: dict[str, object] = {}

    def _capture(command, check=False):
        seen["command"] = command
        index = command.index("--constraint")
        seen["constraint"] = Path(command[index + 1]).read_text(encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", _capture)

    assert extras.install(("dev",), root=project) == 0

    assert seen["command"][-2:] == ["-e", ".[dev]"]
    # The substance: the resolution is held to versions uv.lock carries, so the extra cannot
    # upgrade ruff past the pin `make ci` runs.
    assert "ruff==0.15.18" in str(seen["constraint"])


def test_install_refuses_a_drift_it_caused(project, monkeypatch):
    """The guard fires on the install's OWN damage, which is what a bare `uv pip install` does."""
    monkeypatch.delenv(env.EXTRAS_LOCK_GUARD, raising=False)
    versions = iter([{"ruff": "0.15.18"}, {"ruff": "0.16.3"}])
    monkeypatch.setattr(lock_reader, "installed_versions", lambda names: next(versions))
    monkeypatch.setattr(
        subprocess, "run", lambda command, check=False: subprocess.CompletedProcess(command, 0)
    )

    with pytest.raises(RuntimeError, match="ruff 0.16.3"):
        extras.install(("dev",), root=project)


def test_install_names_leftover_drift_without_failing_on_it(project, monkeypatch, caplog):
    """Drift another extra left behind is a real finding, but not this install's to fail on.

    Failing here would make `make install-extras EXTRAS=pdf-quality` break on a stale `trl` that
    has nothing to do with it -- the same split the vLLM guard makes. The install prints the one
    command that clears the leftover; `make lock-drift` is what turns it into a non-zero exit.
    """
    monkeypatch.delenv(env.EXTRAS_LOCK_GUARD, raising=False)
    monkeypatch.setattr(lock_reader, "installed_versions", lambda names: {"textual": "9.0.0"})
    monkeypatch.setattr(
        subprocess, "run", lambda command, check=False: subprocess.CompletedProcess(command, 0)
    )

    with caplog.at_level("WARNING"):
        assert extras.install(("finetune",), root=project) == 0

    assert "textual 9.0.0 is off the lock" in caplog.text
    assert "make install-extras EXTRAS=dev,review" in caplog.text


def test_guard_mode_reads_the_extras_variable(monkeypatch):
    monkeypatch.delenv(env.EXTRAS_LOCK_GUARD, raising=False)
    assert lock_guard.guard_mode(lock_guard.EXTRAS_GUARD) == lock_guard.GUARD_REFUSE

    monkeypatch.setenv(env.EXTRAS_LOCK_GUARD, "report")
    assert lock_guard.guard_mode(lock_guard.EXTRAS_GUARD) == lock_guard.GUARD_REPORT
    # The vLLM guard keeps its own variable, so relaxing one never relaxes the other.
    monkeypatch.delenv(env.VLLM_LOCK_GUARD, raising=False)
    assert lock_guard.guard_mode(lock_guard.VLLM_GUARD) == lock_guard.GUARD_REFUSE


def test_conflict_hint_names_the_calling_install_and_its_escape_hatch():
    hint = lock_guard.conflict_hint(lock_guard.EXTRAS_GUARD)

    assert "this extras install" in hint
    assert env.EXTRAS_LOCK_GUARD in hint
    assert "uv lock --upgrade-package" in hint


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_make_target_entrypoint_installs_through_a_constraint_file(tmp_path):
    """End to end through `scripts/install_extras.sh` with a fake uv: the call the target issues.

    The constraint file lives in a temp dir the installer deletes on exit, so the fake copies it
    out while it still exists -- that content is the whole point of the target.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_log = tmp_path / "uv.log"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$UV_LOG"
for arg in "$@"; do
  if [ -n "${copy_next:-}" ]; then cp "$arg" "$UV_CONSTRAINT_COPY"; copy_next=; fi
  if [ "$arg" = "--constraint" ]; then copy_next=1; fi
done
if [ "$1 $2" = "cache dir" ]; then echo /shared/uv-cache; fi
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    fake_python = venv_bin / "python"
    fake_python.write_text(
        f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} "$@"\n', encoding="utf-8"
    )
    fake_python.chmod(0o755)

    subprocess.run(
        ["bash", str(INSTALL_EXTRAS_SH), "--extras", "review"],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PROJECT_ROOT": str(tmp_path),
            "UV_LOG": str(uv_log),
            "UV_CONSTRAINT_COPY": str(tmp_path / "constraint.txt"),
            # The post-install re-check reads THIS interpreter, whose declared packages are on the
            # repo lock; `report` keeps a developer's own drift from failing the assertion below.
            env.EXTRAS_LOCK_GUARD: lock_guard.GUARD_REPORT,
        },
        check=True,
        capture_output=True,
        text=True,
    )

    calls = uv_log.read_text(encoding="utf-8")
    assert "pip install --python" in calls
    assert "-e .[review]" in calls
    constraint = (tmp_path / "constraint.txt").read_text(encoding="utf-8").split()
    assert any(line.startswith("ruff==") for line in constraint), constraint


def test_the_installed_extras_match_the_repo_lock():
    """The acceptance assertion: nothing this venv declares sits off `uv.lock`.

    A drifted declared package is exactly the split-verdict failure this module exists to prevent
    -- `make ci` runs the ruff, mypy, and pymarkdownlnt the lock names, and GitHub CI syncs the
    same lock, so a locally-upgraded one gives the same commit two verdicts. The remedy is printed
    by `make lock-drift`; `LLB_EXTRAS_LOCK_GUARD=report` downgrades this to a skip for a developer
    who is deliberately testing an upgrade.
    """
    drifts, groups = extras.find_off_lock(PROJECT_ROOT)
    if not drifts:
        return
    detail = "; ".join(extras.report_lines(drifts, groups))
    if lock_guard.guard_mode(lock_guard.EXTRAS_GUARD) != lock_guard.GUARD_REFUSE:
        pytest.skip(detail)
    pytest.fail(detail)
