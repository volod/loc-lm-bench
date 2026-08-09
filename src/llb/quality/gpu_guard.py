"""The lightweight pytest tier does no GPU work, checked per test instead of assumed.

`make ci` / `make ci-github` run the non-slow suite, which is meant to be the no-GPU, no-download
tier: everything heavier is marked. That was a convention with nothing behind it, and a convention
is exactly what an end-to-end test of an entrypoint breaks by accident -- driving an entrypoint
executes everything that entrypoint does, including probes no stub sits in front of, which is how
`tests/llb/build/test_build_helper.py` paid for a real flashinfer JIT build for as long as it
existed (see `docs/impl/current/host-validation.md#code-quality-checks`).

The guard watches two observable effects of device work around each unmarked test:

- a CUDA CONTEXT that did not exist before the test exists after it (`torch.cuda.is_initialized()`),
- `flashinfer` newly present in `sys.modules`, whose first sampling call JIT-builds a kernel.

Both are read out of `sys.modules` and neither imports anything, so the guard no-ops where torch is
absent (GitHub CI) at the cost of two dict lookups per test. Importing torch is not itself a
finding: `import torch`, `torch.cuda.is_available()`, and a fake `torch` fixture all leave
`is_initialized()` False, so a test that pulls torch in without touching the device stays green.

Escape hatches, in order of preference: mark the test `slow` (the tier where a real backend run
belongs) or `gpu_env` (a quick test that must touch the device anyway). `LLB_GPU_GUARD=report`
downgrades the refusal to a warning for a host that wants the finding without the red build, and
`off` disables it.

A CHILD process leaves nothing here to read -- and the build tests above are exactly that shape,
running the installer through `subprocess.run` -- so children are DENIED the device rather than
observed: for the duration of an unmarked test, `subprocess.Popen` hands every child an empty
`CUDA_VISIBLE_DEVICES`, and a child that tries to open a context finds no device. The denial is
deliberately child-only. Setting the variable in THIS process would poison it for the rest of the
session: `torch.cuda.is_available()` caches, and on torch 2.11 it keeps reporting False after the
variable is restored, so one unmarked test would silently take the GPU away from every `slow` /
`gpu_env` test that follows.

Residual: attribution is first-offender -- once a context exists it exists for the rest of the
session, so a later unmarked test that would have initialized one runs unobserved. The denial
reaches what goes through `subprocess` (`run`, `call`, `check_output`, `Popen`), not
`os.system`/`os.exec*`/`posix_spawn` or a forked child. And it hides the device from the CUDA
runtime, not from NVML: `nvidia-smi` still lists the GPU under an empty `CUDA_VISIBLE_DEVICES`, so a
child that only ASKS whether hardware exists still gets yes -- it just cannot open a context.
"""

import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from llb.core import env

# A test carrying one of these DECLARES device work, so the guard steps aside for it.
EXEMPT_MARKERS = ("slow", "gpu_env")
# Importing one of these is device work in itself, whether or not a context is live yet.
WATCHED_IMPORTS = ("flashinfer",)
# What a child of an unmarked test inherits. An empty value is how the CUDA runtime is told there is
# no device; flashinfer needs no switch of its own, since every probe here asks torch first.
CHILD_DENIAL = {"CUDA_VISIBLE_DEVICES": ""}
# `Popen(args, bufsize, executable, stdin, stdout, stderr, preexec_fn, close_fds, shell, cwd, env)`
_POPEN_ENV_POSITION = 10

MODE_REFUSE = "refuse"
MODE_REPORT = "report"
MODE_OFF = "off"
GUARD_MODES = (MODE_REFUSE, MODE_REPORT, MODE_OFF)
DEFAULT_MODE = MODE_REFUSE

_CUDA_FINDING = "initialized a CUDA context (torch.cuda.is_initialized() went False -> True)"
_IMPORT_FINDING = "imported {name}, which JIT-builds a kernel on first use"
_REMEDY = (
    f"the non-slow suite is the no-GPU tier: inject a fake at the device seam, or mark the test "
    f"{' / '.join(EXEMPT_MARKERS)} to declare the device work "
    f"({env.GPU_GUARD}=report downgrades this to a warning)"
)


@dataclass(frozen=True)
class DeviceState:
    """What the process has already done to the device, read without importing anything."""

    cuda_initialized: bool
    imported: tuple[str, ...]


def device_state(modules: Mapping[str, Any] | None = None) -> DeviceState:
    """Snapshot the watched device effects from an already-imported module table."""
    table = modules if modules is not None else sys.modules
    return DeviceState(
        cuda_initialized=_cuda_initialized(table.get("torch")),
        imported=tuple(name for name in WATCHED_IMPORTS if name in table),
    )


def device_findings(before: DeviceState, after: DeviceState) -> tuple[str, ...]:
    """The device work that happened BETWEEN two snapshots, one phrase per effect."""
    findings: list[str] = []
    if after.cuda_initialized and not before.cuda_initialized:
        findings.append(_CUDA_FINDING)
    for name in after.imported:
        if name not in before.imported:
            findings.append(_IMPORT_FINDING.format(name=name))
    return tuple(findings)


def guard_mode(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the guard mode from the environment; an unknown value is refused, not ignored."""
    raw = (environ if environ is not None else os.environ).get(env.GPU_GUARD)
    if raw is None or not raw.strip():
        return DEFAULT_MODE
    mode = raw.strip().lower()
    if mode not in GUARD_MODES:
        raise ValueError(f"{env.GPU_GUARD}={raw!r} is not one of {', '.join(GUARD_MODES)}")
    return mode


def exempting_marker(marker_names: Iterable[str]) -> str | None:
    """The marker that declares this test's device work, or None when it declares none."""
    names = set(marker_names)
    return next((marker for marker in EXEMPT_MARKERS if marker in names), None)


@dataclass(frozen=True)
class DeviceGuard:
    """One test's before-snapshot plus what to do with a finding after it."""

    mode: str
    before: DeviceState

    @classmethod
    def start(
        cls,
        marker_names: Iterable[str],
        *,
        environ: Mapping[str, str] | None = None,
        modules: Mapping[str, Any] | None = None,
    ) -> "DeviceGuard | None":
        """Open the guard for a test, or None when the test is exempt or the guard is off."""
        mode = guard_mode(environ)
        if mode == MODE_OFF or exempting_marker(marker_names) is not None:
            return None
        return cls(mode=mode, before=device_state(modules))

    @property
    def denies_children(self) -> bool:
        """Only the refusing mode denies. `report` exists to let a run THROUGH while saying what it
        did, and a denial says nothing -- the child simply finds no device."""
        return self.mode == MODE_REFUSE

    def verdict(
        self, test_id: str, *, modules: Mapping[str, Any] | None = None
    ) -> tuple[str, str] | None:
        """`(mode, message)` for a test that reached the device, or None when it stayed clean."""
        findings = device_findings(self.before, device_state(modules))
        if not findings:
            return None
        return self.mode, guard_message(test_id, findings)


def denied_child_env(inherited: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a child gets: what it would have inherited, with the device taken out."""
    base = dict(os.environ if inherited is None else inherited)
    base.update(CHILD_DENIAL)
    return base


def denying_popen(original: type[Any]) -> type[Any]:
    """A `subprocess.Popen` that starts every child with no visible CUDA device.

    Substituted for the module attribute, so `subprocess.run` / `call` / `check_output` -- which all
    reach for `subprocess.Popen` at call time -- are covered by the one patch.
    """

    class _DeviceDeniedPopen(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if len(args) > _POPEN_ENV_POSITION:
                mutable = list(args)
                mutable[_POPEN_ENV_POSITION] = denied_child_env(mutable[_POPEN_ENV_POSITION])
                args = tuple(mutable)
            else:
                kwargs["env"] = denied_child_env(kwargs.get("env"))
            super().__init__(*args, **kwargs)

    return _DeviceDeniedPopen


def guard_message(test_id: str, findings: Iterable[str]) -> str:
    """The operator-facing line: what the test did to the device, and the ways out."""
    did = "; ".join(findings)
    return f"[gpu-guard] {test_id} {did}. {_REMEDY}"


def _cuda_initialized(torch_module: Any) -> bool:
    """True only when a real torch reports a live context; a fake or a stub reads as False."""
    try:
        return bool(torch_module.cuda.is_initialized())
    except Exception:
        return False
