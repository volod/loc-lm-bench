"""Keep an install inside the versions `uv.lock` already resolved.

`make venv` syncs the lock (exactly the versions GitHub CI installs) and THEN runs a second
installer over the same venv. Both of the second steps this repo drives resolve freely: vLLM's own
requirements are mostly unpinned and a handful of them name packages this project also declares
(`mcp`, `numpy`, `openai`, `psutil`, `pydantic`, `pyyaml`), and uv's pip interface has no lockfile
at all, so `uv pip install -e ".[review]"` re-resolves the whole requirement set and takes the
newest version each specifier admits. Either way the damage surfaces two steps later as a `make ci`
lint or type error inside `src/llb/...`, which reads like a source bug and is a dependency
resolution. This module stops the drift where it happens, for both callers -- a `Guard` carries the
three strings that differ (what the install calls itself, which variable relaxes it, and the
command that puts the venv back); the mechanism below is identical.

The constraint is derived from `uv.lock` (read through `llb.build.lock_reader`), never from
whatever the venv happens to hold: a package is constrained only to versions the lock actually
carries. Packages the lock does not declare are deliberately left alone -- torch on a CUDA host is
hardware-matched and trails the lock, so constraining it would break the vLLM install this guard
runs beside. After the install the environment is re-read and compared against a pre-install
snapshot, because the two kinds of drift are not the same finding: a package this install MOVED off
the lock is the failure this guard exists for, while a package that was already off the lock is an
earlier install's leftover and is reported without failing someone else's work. The guard's
variable picks what a caused drift costs: `refuse` (default) fails the install, `report` logs and
continues, `off` skips both the constraint and the check.
"""

import logging
import os
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from llb.build import lock_reader
from llb.core import env
from llb.core.paths import PROJECT_ROOT

_LOG = logging.getLogger(__name__)

GUARD_REFUSE = "refuse"
GUARD_REPORT = "report"
GUARD_OFF = "off"
GUARD_MODES = (GUARD_REFUSE, GUARD_REPORT, GUARD_OFF)
DEFAULT_GUARD_MODE = GUARD_REFUSE

CONSTRAINT_FILENAME = "uv-lock-constraints.txt"


@dataclass(frozen=True)
class Guard:
    """Which install is being held to the lock, so one policy serves more than one caller."""

    subject: str
    variable: str
    remedy: str


VLLM_GUARD = Guard(
    subject="the vLLM install",
    variable=env.VLLM_LOCK_GUARD,
    remedy="re-run `make venv` to sync them back",
)
EXTRAS_GUARD = Guard(
    subject="this extras install",
    variable=env.EXTRAS_LOCK_GUARD,
    remedy="re-run `make install-extras EXTRAS=<groups>` for the groups that declare them",
)


def conflict_hint(guard: Guard) -> str:
    """The one failure the constraint introduces: an installer asking for a version the lock lacks.

    That is a real disagreement about what the environment is, so it is named rather than resolved
    silently -- otherwise it reads as an unrelated uv resolution error.
    """
    return (
        f"{guard.subject} ran under a uv.lock constraint -- a uv version conflict above means it "
        "now needs a package version the lock does not carry. Refresh it (`uv lock "
        "--upgrade-package <name>`, then commit uv.lock), or re-run with "
        f"{guard.variable}=off to install unconstrained and accept the drift."
    )


@dataclass(frozen=True)
class Drift:
    """One declared package whose installed version is not a version the lock carries."""

    name: str
    installed: str
    locked: tuple[str, ...]


@dataclass(frozen=True)
class DriftReport:
    """Off-lock packages split by whether THIS install is what moved them."""

    caused: tuple[Drift, ...] = ()
    preexisting: tuple[Drift, ...] = ()


def plan_constraints(
    locked: dict[str, set[str]], declared: Iterable[str], installed: dict[str, str]
) -> dict[str, str]:
    """Choose the version specifier to constrain each declared package with.

    Prefer the installed version when the lock carries it -- the sync that ran just before put it
    there, so pinning it is "do not move what the lock already chose for THIS extra set". A package
    the environment does not hold yet is the case that actually matters (`mcp` is in no `make venv`
    extra; the vLLM install is what first pulls it in), so it still gets a constraint: an exact pin
    when the lock carries one version, and the CLOSED RANGE the lock spans when conflicting extras
    forked it. The range neither guesses which fork this environment is nor lets the resolver leave
    what the lock resolved -- and `enforce` still refuses anything that lands outside it.
    """
    plan: dict[str, str] = {}
    for name in sorted(set(declared)):
        versions = lock_reader.sorted_versions(locked.get(name, ()))
        if not versions:
            continue
        current = installed.get(name)
        if current is not None and current in versions:
            plan[name] = f"=={current}"
        elif len(versions) == 1:
            plan[name] = f"=={versions[0]}"
        else:
            plan[name] = f">={versions[0]},<={versions[-1]}"
    return plan


def find_drift(
    locked: dict[str, set[str]], declared: Iterable[str], installed: dict[str, str]
) -> list[Drift]:
    """Declared packages whose installed version is not one the lock carries."""
    return [
        Drift(name, installed[name], tuple(sorted(locked[name])))
        for name in sorted(set(declared))
        if name in installed and name in locked and installed[name] not in locked[name]
    ]


def split_drift(drifts: Iterable[Drift], before: dict[str, str]) -> DriftReport:
    """Attribute each off-lock package to this install or to whatever left it that way."""
    caused = tuple(drift for drift in drifts if before.get(drift.name) != drift.installed)
    names = {drift.name for drift in caused}
    return DriftReport(
        caused=caused,
        preexisting=tuple(drift for drift in drifts if drift.name not in names),
    )


def describe(drift: Drift) -> str:
    return f"{drift.name} {drift.installed} is off the lock (pinned: {', '.join(drift.locked)})"


def guard_mode(guard: Guard = VLLM_GUARD) -> str:
    """Read the guard's variable (refuse | report | off); refuse is the default."""
    raw = os.environ.get(guard.variable, "").strip().lower() or DEFAULT_GUARD_MODE
    if raw not in GUARD_MODES:
        raise ValueError(f"{guard.variable} must be one of {'/'.join(GUARD_MODES)}: {raw!r}")
    return raw


def lock_inputs(root: Path) -> tuple[dict[str, set[str]], set[str]] | None:
    """Load the lock index + declared names, or None when this root carries no lock to respect."""
    lock_path = root / lock_reader.LOCK_FILENAME
    pyproject_path = root / lock_reader.PYPROJECT_FILENAME
    if not lock_path.is_file() or not pyproject_path.is_file():
        _LOG.warning(
            "[lock-guard] no %s under %s; nothing to constrain", lock_reader.LOCK_FILENAME, root
        )
        return None
    return lock_reader.lock_index(lock_path), lock_reader.declared_names(pyproject_path)


def snapshot(mode: str, *, root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Declared-package versions BEFORE the install: the baseline `enforce` attributes drift with."""
    inputs = None if mode == GUARD_OFF else lock_inputs(root)
    if inputs is None:
        return {}
    return lock_reader.installed_versions(inputs[1])


@contextmanager
def lock_constraint(
    mode: str, before: dict[str, str], *, root: Path = PROJECT_ROOT, guard: Guard = VLLM_GUARD
) -> Iterator[Path | None]:
    """Yield a uv constraint file pinning the declared packages to their locked versions."""
    inputs = None if mode == GUARD_OFF else lock_inputs(root)
    if inputs is None:
        yield None
        return
    locked, declared = inputs
    plan = plan_constraints(locked, declared, before)
    if not plan:
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="llb-vllm-lock-") as scratch:
        path = Path(scratch) / CONSTRAINT_FILENAME
        path.write_text(
            "".join(f"{name}{specifier}\n" for name, specifier in plan.items()), encoding="utf-8"
        )
        _LOG.info("[lock-guard] holding %d declared packages at their locked versions", len(plan))
        try:
            yield path
        except subprocess.CalledProcessError:
            _LOG.error("[lock-guard] %s", conflict_hint(guard))
            raise


def constraint_args(constraint: Path | None) -> list[str]:
    """uv flags for an optional constraint file (empty when the guard produced none)."""
    return ["--constraint", str(constraint)] if constraint is not None else []


def enforce(
    mode: str, before: dict[str, str], *, root: Path = PROJECT_ROOT, guard: Guard = VLLM_GUARD
) -> DriftReport:
    """Re-read the environment after the install and name every package that left the lock."""
    inputs = None if mode == GUARD_OFF else lock_inputs(root)
    if inputs is None:
        return DriftReport()
    locked, declared = inputs
    installed = lock_reader.installed_versions(declared)
    report = split_drift(find_drift(locked, declared, installed), before)
    for drift in report.preexisting:
        # Not this install's doing -- an earlier free resolution left it off the lock.
        _LOG.warning("[lock-guard] %s (already, before this install)", describe(drift))
    if not report.caused:
        _LOG.info("[lock-guard] %s moved nothing off %s", guard.subject, lock_reader.LOCK_FILENAME)
        return report
    for drift in report.caused:
        _LOG.error("[lock-guard] %s", describe(drift))
    if mode == GUARD_REFUSE:
        raise RuntimeError(
            f"{guard.subject} moved {len(report.caused)} package(s) off "
            f"{lock_reader.LOCK_FILENAME}: "
            + "; ".join(describe(drift) for drift in report.caused)
            + ". `make ci` type-checks against the locked versions -- "
            f"{guard.remedy}, or set {guard.variable}=report to install anyway."
        )
    return report
