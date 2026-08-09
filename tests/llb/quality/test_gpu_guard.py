"""The no-GPU guard has to catch device work without punishing a test that merely imports torch.

`llb.quality.gpu_guard` reads two effects out of an already-imported module table: a CUDA context
that appeared during a test, and `flashinfer` newly imported. These tests pin the three properties
that decide whether the guard is usable -- it fires on device work, it stays silent for a plain
`import torch` and on a host with no torch at all, and a declared marker or `LLB_GPU_GUARD` takes it
out of the way. The live suite is the fourth assertion: `tests/conftest.py` runs it over every
unmarked test in this file too.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from llb.core import env
from llb.quality import gpu_guard

_CONFTEST = Path(__file__).resolve().parents[2] / "conftest.py"


def _suite_conftest() -> ModuleType:
    """The suite's root conftest, loaded under its own name so its wiring is drivable here."""
    spec = importlib.util.spec_from_file_location("llb_suite_conftest", _CONFTEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _torch(initialized: bool) -> SimpleNamespace:
    """A torch stand-in shaped like the one attribute the guard reads."""
    return SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: initialized))


def _modules(*, torch: object | None = None, flashinfer: bool = False) -> dict[str, object]:
    table: dict[str, object] = {}
    if torch is not None:
        table["torch"] = torch
    if flashinfer:
        table["flashinfer"] = SimpleNamespace()
    return table


def test_no_torch_on_the_host_reads_as_a_clean_state():
    """GitHub CI has no torch, so the guard must no-op rather than import one to ask."""
    state = gpu_guard.device_state(_modules())
    assert state == gpu_guard.DeviceState(cuda_initialized=False, imported=())


def test_importing_torch_without_touching_the_device_is_not_a_finding():
    before = gpu_guard.device_state(_modules())
    after = gpu_guard.device_state(_modules(torch=_torch(initialized=False)))
    assert gpu_guard.device_findings(before, after) == ()


def test_a_fake_torch_fixture_does_not_read_as_a_live_context():
    """A test's fake `torch` shadows the real one; a stub with no `cuda` must read as clean."""
    after = gpu_guard.device_state(_modules(torch=SimpleNamespace()))
    assert after.cuda_initialized is False


def test_a_cuda_context_opened_during_the_test_is_a_finding():
    before = gpu_guard.device_state(_modules(torch=_torch(initialized=False)))
    after = gpu_guard.device_state(_modules(torch=_torch(initialized=True)))
    findings = gpu_guard.device_findings(before, after)
    assert len(findings) == 1
    assert "CUDA context" in findings[0]


def test_a_context_that_predates_the_test_is_not_charged_to_it():
    """Attribution is first-offender: once a context exists, later tests inherit it innocent."""
    live = gpu_guard.device_state(_modules(torch=_torch(initialized=True)))
    assert gpu_guard.device_findings(live, live) == ()


def test_importing_flashinfer_is_a_finding_on_its_own():
    """The JIT build is the cost, and it happens without any torch context in this process."""
    before = gpu_guard.device_state(_modules())
    after = gpu_guard.device_state(_modules(flashinfer=True))
    findings = gpu_guard.device_findings(before, after)
    assert len(findings) == 1
    assert "flashinfer" in findings[0]


@pytest.mark.parametrize("marker", gpu_guard.EXEMPT_MARKERS)
def test_a_declared_marker_takes_the_guard_out_of_the_way(marker: str):
    assert gpu_guard.exempting_marker([marker, "parametrize"]) == marker
    assert gpu_guard.DeviceGuard.start([marker], environ={}, modules=_modules()) is None


def test_an_undeclared_test_is_guarded_and_its_verdict_names_the_test_and_the_way_out():
    guard = gpu_guard.DeviceGuard.start(
        ["parametrize"], environ={}, modules=_modules(torch=_torch(initialized=False))
    )
    assert guard is not None
    outcome = guard.verdict("tests/x.py::test_y", modules=_modules(torch=_torch(initialized=True)))
    assert outcome is not None
    mode, message = outcome
    assert mode == gpu_guard.MODE_REFUSE
    assert "tests/x.py::test_y" in message
    assert "gpu_env" in message


def test_a_clean_test_gets_no_verdict():
    modules = _modules(torch=_torch(initialized=False))
    guard = gpu_guard.DeviceGuard.start(["parametrize"], environ={}, modules=modules)
    assert guard is not None
    assert guard.verdict("tests/x.py::test_y", modules=modules) is None


def test_report_mode_keeps_the_finding_and_drops_the_refusal():
    environ = {env.GPU_GUARD: "report"}
    guard = gpu_guard.DeviceGuard.start(["parametrize"], environ=environ, modules=_modules())
    assert guard is not None
    outcome = guard.verdict("tests/x.py::test_y", modules=_modules(flashinfer=True))
    assert outcome is not None
    assert outcome[0] == gpu_guard.MODE_REPORT


def test_off_disables_the_guard_entirely():
    environ = {env.GPU_GUARD: "off"}
    assert gpu_guard.DeviceGuard.start(["parametrize"], environ=environ, modules=_modules()) is None


def test_the_default_mode_is_a_refusal():
    assert gpu_guard.guard_mode({}) == gpu_guard.MODE_REFUSE
    assert gpu_guard.guard_mode({env.GPU_GUARD: ""}) == gpu_guard.MODE_REFUSE


def test_an_unknown_mode_is_refused_rather_than_read_as_off():
    """A typo'd knob must not silently disable the check it was aimed at."""
    with pytest.raises(ValueError, match="LLB_GPU_GUARD"):
        gpu_guard.guard_mode({env.GPU_GUARD: "yes"})


# The three below drive the SUITE's own fixture body against the real `sys.modules` and the real
# environment, so they cover the wiring a fake module table cannot: the guard reads the live process
# and reports through pytest. Each pins the mode rather than inheriting the operator's, or a run
# under `LLB_GPU_GUARD=off` would assert the guard fires while it is switched off. `monkeypatch`
# undoes the fake import before the autouse guard wrapping THESE tests takes its own after-snapshot
# -- the root conftest fixture is set up first, so it is torn down last.
def _reach_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace())


def test_the_suite_wiring_fails_an_unmarked_test_that_reaches_the_device(monkeypatch):
    monkeypatch.setenv(env.GPU_GUARD, gpu_guard.MODE_REFUSE)
    steps = _suite_conftest().device_guard_steps("tests/x.py::test_y", [])
    next(steps)
    _reach_the_device(monkeypatch)
    with pytest.raises(pytest.fail.Exception, match=r"\[gpu-guard\] tests/x.py::test_y"):
        next(steps, None)


def test_the_suite_wiring_leaves_a_declared_test_alone(monkeypatch):
    monkeypatch.setenv(env.GPU_GUARD, gpu_guard.MODE_REFUSE)
    steps = _suite_conftest().device_guard_steps("tests/x.py::test_y", ["gpu_env"])
    next(steps)
    _reach_the_device(monkeypatch)
    assert next(steps, None) is None


def test_the_suite_wiring_warns_instead_of_failing_in_report_mode(monkeypatch):
    monkeypatch.setenv(env.GPU_GUARD, gpu_guard.MODE_REPORT)
    steps = _suite_conftest().device_guard_steps("tests/x.py::test_y", [])
    next(steps)
    _reach_the_device(monkeypatch)
    with pytest.warns(UserWarning, match=r"\[gpu-guard\]"):
        assert next(steps, None) is None
