"""The vLLM install must not move a package `uv.lock` pins (llb.build.lock_guard).

vLLM's requirements are mostly unpinned and overlap packages this project declares, so an
unconstrained install can upgrade one of them past the lock -- and the failure lands in `make ci`
as a type error in our own source. These cover the constraint the install now runs under and the
post-install check that names a drift at install time instead.
"""

import subprocess
from pathlib import Path

import pytest

from llb.build import lock_guard, lock_reader
from llb.core import env
from llb.core.paths import PROJECT_ROOT

LOCK_FIXTURE = """
version = 1

[[package]]
name = "mcp"
version = "1.26.0"

[[package]]
name = "mcp"
version = "1.28.1"

[[package]]
name = "NumPy"
version = "2.4.6"

[[package]]
name = "typing_extensions"
version = "4.16.0"

[[package]]
name = "llb"
source = { virtual = "." }
"""

PYPROJECT_FIXTURE = """
[project]
name = "llb"
dependencies = ["numpy>=2.0", "httpx>=0.27", "langchain-text-splitters==1.1.2"]

[project.optional-dependencies]
mcp = ["mcp>=1.2,<2", "anyio>=4.0"]
pdf-quality = ["docling[rapidocr]>=2.0"]
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / lock_reader.LOCK_FILENAME).write_text(LOCK_FIXTURE, encoding="utf-8")
    (tmp_path / lock_reader.PYPROJECT_FILENAME).write_text(PYPROJECT_FIXTURE, encoding="utf-8")
    return tmp_path


def test_lock_index_keeps_every_locked_version_per_name(project):
    locked = lock_reader.lock_index(project / lock_reader.LOCK_FILENAME)
    # Conflicting extras pin `mcp` twice; "matches the lock" must mean either version.
    assert locked["mcp"] == {"1.26.0", "1.28.1"}
    assert locked["numpy"] == {"2.4.6"}  # name canonicalized from "NumPy"
    assert locked["typing-extensions"] == {"4.16.0"}
    assert "llb" not in locked  # the virtual project entry carries no version


def test_sorted_versions_orders_by_release_segment():
    # Plain string order would put 1.9.0 after 1.28.1 and break the bounded-range constraint.
    assert lock_reader.sorted_versions(["1.28.1", "1.9.0", "1.26.0"]) == [
        "1.9.0",
        "1.26.0",
        "1.28.1",
    ]
    assert lock_reader.sorted_versions(["2.0.0", "2.0.0rc1"]) == ["2.0.0rc1", "2.0.0"]


def test_declared_names_covers_base_and_every_extra(project):
    declared = lock_reader.declared_names(project / lock_reader.PYPROJECT_FILENAME)
    assert declared == {
        "numpy",
        "httpx",
        "langchain-text-splitters",
        "mcp",
        "anyio",
        "docling",  # the [rapidocr] extra marker is stripped from the name
    }


def test_plan_constraints_prefers_the_installed_locked_version(project):
    locked = lock_reader.lock_index(project / lock_reader.LOCK_FILENAME)
    declared = lock_reader.declared_names(project / lock_reader.PYPROJECT_FILENAME)

    plan = lock_guard.plan_constraints(locked, declared, {"mcp": "1.28.1", "numpy": "2.4.6"})

    assert plan == {"mcp": "==1.28.1", "numpy": "==2.4.6"}


def test_plan_constraints_bounds_a_package_the_environment_lacks(project):
    """The case that actually bites: `mcp` is in no `make venv` extra, so the vLLM install is what
    first pulls it in -- unconstrained, that resolves to the newest major on PyPI."""
    locked = lock_reader.lock_index(project / lock_reader.LOCK_FILENAME)
    declared = lock_reader.declared_names(project / lock_reader.PYPROJECT_FILENAME)

    plan = lock_guard.plan_constraints(locked, declared, {})

    # One locked version pins exactly; the conflicting-extra fork is bounded to what the lock
    # spans, which excludes 2.x without guessing which fork this environment is.
    assert plan == {"numpy": "==2.4.6", "mcp": ">=1.26.0,<=1.28.1"}


def test_plan_constraints_ignores_an_already_drifted_install(project):
    locked = lock_reader.lock_index(project / lock_reader.LOCK_FILENAME)
    declared = lock_reader.declared_names(project / lock_reader.PYPROJECT_FILENAME)

    plan = lock_guard.plan_constraints(locked, declared, {"mcp": "2.0.0"})

    # A drifted version is never re-pinned as if it were the lock -- it falls back to the lock.
    assert plan["mcp"] == ">=1.26.0,<=1.28.1"


def test_find_drift_names_the_off_lock_package(project):
    locked = lock_reader.lock_index(project / lock_reader.LOCK_FILENAME)
    declared = lock_reader.declared_names(project / lock_reader.PYPROJECT_FILENAME)

    drifts = lock_guard.find_drift(
        locked, declared, {"mcp": "2.0.0", "numpy": "2.4.6", "typing-extensions": "9.9.9"}
    )

    # typing-extensions is locked but NOT declared here, so it is none of this guard's business.
    assert drifts == [lock_guard.Drift("mcp", "2.0.0", ("1.26.0", "1.28.1"))]
    assert "mcp 2.0.0" in lock_guard.describe(drifts[0])


def test_snapshot_reads_only_declared_packages(project, monkeypatch):
    seen: dict[str, set[str]] = {}

    def _fake(names):
        seen["names"] = set(names)
        return {"mcp": "1.28.1"}

    monkeypatch.setattr(lock_reader, "installed_versions", _fake)

    assert lock_guard.snapshot(lock_guard.GUARD_REFUSE, root=project) == {"mcp": "1.28.1"}
    assert "docling" in seen["names"] and "typing-extensions" not in seen["names"]
    assert lock_guard.snapshot(lock_guard.GUARD_OFF, root=project) == {}


def test_lock_constraint_writes_uv_pins(project):
    before = {"mcp": "1.28.1"}

    with lock_guard.lock_constraint(lock_guard.GUARD_REFUSE, before, root=project) as constraint:
        assert constraint is not None
        lines = sorted(constraint.read_text(encoding="utf-8").split())
        assert lines == ["mcp==1.28.1", "numpy==2.4.6"]  # installed + locked pins exactly
        assert lock_guard.constraint_args(constraint) == ["--constraint", str(constraint)]

    assert not constraint.exists()  # the scratch dir is temporary, never left in the tree


def test_lock_constraint_explains_a_uv_resolution_failure(project, caplog):
    """The one failure the constraint introduces -- vLLM wanting a version the lock lacks -- must
    say so, or it reads as an unrelated uv error."""
    with pytest.raises(subprocess.CalledProcessError):
        with lock_guard.lock_constraint(lock_guard.GUARD_REFUSE, {}, root=project):
            raise subprocess.CalledProcessError(1, ["uv", "pip", "install"])

    hint = lock_guard.conflict_hint(lock_guard.VLLM_GUARD)
    assert hint in caplog.text
    assert "uv lock --upgrade-package" in hint and env.VLLM_LOCK_GUARD in hint


def test_lock_constraint_is_skipped_when_the_guard_is_off(project):
    with lock_guard.lock_constraint(lock_guard.GUARD_OFF, {}, root=project) as constraint:
        assert constraint is None
    assert lock_guard.constraint_args(None) == []


def test_enforce_refuses_a_drift_this_install_caused(project, monkeypatch):
    monkeypatch.setattr(lock_reader, "installed_versions", lambda names: {"mcp": "2.0.0"})

    with pytest.raises(RuntimeError) as excinfo:
        lock_guard.enforce(lock_guard.GUARD_REFUSE, {"mcp": "1.28.1"}, root=project)

    message = str(excinfo.value)
    assert "mcp 2.0.0" in message and "1.26.0, 1.28.1" in message
    assert env.VLLM_LOCK_GUARD in message  # the message names its own escape hatch


def test_enforce_does_not_blame_the_install_for_preexisting_drift(project, monkeypatch):
    """A hand-installed extra (`.[review]`, `.[pdf-quality]`) may already sit off the lock.

    That is a real finding but not this install's doing, so it is reported rather than failed --
    otherwise `make venv` breaks on a host whose drift has nothing to do with vLLM.
    """
    monkeypatch.setattr(lock_reader, "installed_versions", lambda names: {"mcp": "2.0.0"})

    report = lock_guard.enforce(lock_guard.GUARD_REFUSE, {"mcp": "2.0.0"}, root=project)

    assert report.caused == ()
    assert report.preexisting == (lock_guard.Drift("mcp", "2.0.0", ("1.26.0", "1.28.1")),)


def test_enforce_reports_without_failing_and_off_skips(project, monkeypatch):
    monkeypatch.setattr(lock_reader, "installed_versions", lambda names: {"mcp": "2.0.0"})

    report = lock_guard.enforce(lock_guard.GUARD_REPORT, {"mcp": "1.28.1"}, root=project)
    assert report.caused == (lock_guard.Drift("mcp", "2.0.0", ("1.26.0", "1.28.1")),)

    assert lock_guard.enforce(lock_guard.GUARD_OFF, {}, root=project) == lock_guard.DriftReport()


def test_enforce_passes_a_synced_environment(project, monkeypatch):
    monkeypatch.setattr(lock_reader, "installed_versions", lambda names: {"mcp": "1.26.0"})

    assert lock_guard.enforce(lock_guard.GUARD_REFUSE, {}, root=project) == lock_guard.DriftReport()


def test_guard_mode_defaults_to_refuse_and_rejects_junk(monkeypatch):
    monkeypatch.delenv(env.VLLM_LOCK_GUARD, raising=False)
    assert lock_guard.guard_mode() == lock_guard.GUARD_REFUSE

    monkeypatch.setenv(env.VLLM_LOCK_GUARD, "REPORT")
    assert lock_guard.guard_mode() == lock_guard.GUARD_REPORT

    monkeypatch.setenv(env.VLLM_LOCK_GUARD, "maybe")
    with pytest.raises(ValueError, match=env.VLLM_LOCK_GUARD):
        lock_guard.guard_mode()


def test_the_installed_mcp_matches_the_repo_lock():
    """The acceptance assertion: the `mcp` this environment carries is a version `uv.lock` pins.

    `mcp` is the package the unconstrained vLLM install actually moved (vLLM requires it with no
    specifier), and mypy reads it directly through `src/llb/bench/mcp_server.py`. Asserting it here
    names the dependency resolution; without this, the same drift surfaces as a type error in our
    own source. An environment without `mcp` -- the lean GitHub CI install -- has nothing to check.
    """
    locked = lock_reader.lock_index(PROJECT_ROOT / lock_reader.LOCK_FILENAME)
    installed = lock_reader.installed_versions(["mcp"])
    if not installed:
        pytest.skip("mcp is an opt-in extra and is not installed here")

    drifts = lock_guard.find_drift(locked, ["mcp"], installed)

    assert not drifts, "; ".join(lock_guard.describe(drift) for drift in drifts)
