"""Install an optional extra without re-resolving the venv off `uv.lock`.

`make venv` syncs the lock, and a follow-up `uv pip install -e ".[review]"` throws that away: uv's
pip interface has no lockfile, so it resolves the WHOLE requirement set fresh and takes the newest
version each specifier admits. Measured on a clean venv, the `dev` extra alone lands ten declared
packages off the lock -- `ruff` and `mypy` among them -- so a local `make ci` runs a different
linter and a different type checker than GitHub CI, which is the split verdict the exact `dev` pins
exist to prevent.

The fix is what `llb.build.lock_guard` already does for vLLM: derive a constraint file from
`uv.lock` and install under it. Nothing here is extras-specific beyond assembling the `.[a,b]`
requirement; the constraint planner, the drift report, and the three guard modes are shared with
the vLLM path. What deliberately stays UNconstrained is everything the lock does not declare,
because torch on a CUDA host is hardware-matched -- it trails the lock while vLLM pins it -- and an
extras install must leave it exactly where `scripts/build_vllm.sh` put it.

The report is the other half. An off-lock package is put back by re-installing the group that
declares it, so `--check` names each drifted package WITH its declaring groups and prints the one
`make install-extras` command that resolves the whole set.
"""

import argparse
import logging
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from llb.build import lock_guard, lock_reader
from llb.core.paths import PROJECT_ROOT

_LOG = logging.getLogger(__name__)

GROUP_SEPARATOR = ","
EDITABLE_PROJECT = "."
GUARD = lock_guard.EXTRAS_GUARD


def parse_groups(text: str) -> tuple[str, ...]:
    """Split an `EXTRAS=review,pdf-quality` comma list, dropping blanks and duplicates."""
    parts = (part.strip() for part in text.split(GROUP_SEPARATOR))
    return tuple(dict.fromkeys(part for part in parts if part))


def project_requirement(groups: Iterable[str]) -> str:
    """The editable requirement these groups install: `.[a,b]`, or `.` when none were named."""
    selected = tuple(groups)
    if not selected:
        return EDITABLE_PROJECT
    return f"{EDITABLE_PROJECT}[{GROUP_SEPARATOR.join(selected)}]"


def extra_names(declared: Iterable[str]) -> set[str]:
    """The installable extra names in a `declared_groups` mapping (the base list is not one)."""
    return {name for name in declared if name != lock_reader.BASE_GROUP}


def unknown_groups(groups: Iterable[str], declared: Iterable[str]) -> tuple[str, ...]:
    """Named groups `pyproject.toml` has no extra for -- a typo installs the base project silently."""
    known = extra_names(declared)
    return tuple(group for group in groups if group not in known)


def install_args(python: Path, groups: Iterable[str], constraint: Path | None) -> list[str]:
    """The uv invocation: an editable install of the named extras under the lock constraint."""
    return [
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        *lock_guard.constraint_args(constraint),
        "-e",
        project_requirement(groups),
    ]


def declaring_groups(name: str, groups: dict[str, set[str]]) -> tuple[str, ...]:
    """Every extra that declares this package (base-only packages report no extra)."""
    return tuple(
        sorted(
            group
            for group, names in groups.items()
            if name in names and group != lock_reader.BASE_GROUP
        )
    )


def remedy_command(drifts: Iterable[lock_guard.Drift], groups: dict[str, set[str]]) -> str | None:
    """The single `make install-extras` that re-resolves every drifted package under the lock.

    A package only the base list declares needs no extra, so it is covered by the editable install
    the command runs anyway; `None` means there is nothing left to put back.
    """
    drifted = tuple(drifts)
    if not drifted:
        return None
    needed = sorted({group for drift in drifted for group in declaring_groups(drift.name, groups)})
    if not needed:
        return "make install-extras EXTRAS="
    return f"make install-extras EXTRAS={GROUP_SEPARATOR.join(needed)}"


def report_lines(drifts: Iterable[lock_guard.Drift], groups: dict[str, set[str]]) -> list[str]:
    """The off-lock report: what drifted, which group declares it, and the command that fixes it."""
    drifted = tuple(drifts)
    if not drifted:
        return [f"every declared package matches {lock_reader.LOCK_FILENAME}"]
    lines = [f"{len(drifted)} declared package(s) sit off {lock_reader.LOCK_FILENAME}:"]
    for drift in drifted:
        declared_by = declaring_groups(drift.name, groups)
        owner = f"declared by {GROUP_SEPARATOR.join(declared_by)}" if declared_by else "base"
        lines.append(f"  {lock_guard.describe(drift)} -- {owner}")
    lines.append(f"put them back with: {remedy_command(drifted, groups)}")
    return lines


def find_off_lock(root: Path) -> tuple[list[lock_guard.Drift], dict[str, set[str]]]:
    """Every declared package whose installed version is not one the lock carries, plus the groups."""
    inputs = lock_guard.lock_inputs(root)
    if inputs is None:
        return [], {}
    locked, declared = inputs
    groups = lock_reader.declared_groups(root / lock_reader.PYPROJECT_FILENAME)
    installed = lock_reader.installed_versions(declared)
    return lock_guard.find_drift(locked, declared, installed), groups


def _log_report(drifts: list[lock_guard.Drift], groups: dict[str, set[str]]) -> None:
    level = logging.WARNING if drifts else logging.INFO
    for line in report_lines(drifts, groups):
        _LOG.log(level, "[extras] %s", line)


def _log_remedy(drifts: Iterable[lock_guard.Drift], *, root: Path) -> None:
    """One actionable line after an install: the command that puts leftover drift back."""
    remaining = tuple(drifts)
    if not remaining:
        return
    groups = lock_reader.declared_groups(root / lock_reader.PYPROJECT_FILENAME)
    _LOG.warning("[extras] put them back with: %s", remedy_command(remaining, groups))


def check(root: Path = PROJECT_ROOT, *, mode: str = lock_guard.DEFAULT_GUARD_MODE) -> int:
    """Report the off-lock declared packages; `refuse` also makes that a non-zero exit."""
    drifts, groups = find_off_lock(root)
    _log_report(drifts, groups)
    return 1 if drifts and mode == lock_guard.GUARD_REFUSE else 0


def install(groups: tuple[str, ...], *, root: Path = PROJECT_ROOT) -> int:
    """Install the named extras under the lock constraint, then re-check the environment."""
    declared = lock_reader.declared_groups(root / lock_reader.PYPROJECT_FILENAME)
    missing = unknown_groups(groups, declared)
    if missing:
        raise ValueError(
            f"no such extra in {lock_reader.PYPROJECT_FILENAME}: {', '.join(missing)} "
            f"(available: {', '.join(sorted(extra_names(declared)))})"
        )
    mode = lock_guard.guard_mode(GUARD)
    before = lock_guard.snapshot(mode, root=root)
    _LOG.info(
        "[extras] installing %s under %s", project_requirement(groups), lock_reader.LOCK_FILENAME
    )
    with lock_guard.lock_constraint(mode, before, root=root, guard=GUARD) as constraint:
        subprocess.run(install_args(Path(sys.executable), groups, constraint), check=True)
    # `enforce` fails on drift THIS install caused and names the drift an earlier one left behind;
    # that leftover does not fail an unrelated install -- the same split the vLLM path makes -- so
    # all this adds is the command that clears it. `make lock-drift` is where it becomes an exit.
    report = lock_guard.enforce(mode, before, root=root, guard=GUARD)
    _log_remedy(report.preexisting, root=root)
    return 0


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--extras",
        default="",
        help="comma-separated optional-dependency groups to install (e.g. review,pdf-quality)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report declared packages sitting off uv.lock; install nothing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    if arguments.check:
        return check(mode=lock_guard.guard_mode(GUARD))
    try:
        return install(parse_groups(arguments.extras))
    except ValueError as error:
        # A mistyped extra is a usage mistake, not a crash: name it without a traceback.
        _LOG.error("[extras] %s", error)
        return 2


if __name__ == "__main__":
    from llb.core.runtime import run

    raise SystemExit(run(main))
