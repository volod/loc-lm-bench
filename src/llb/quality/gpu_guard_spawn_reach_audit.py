"""The refusal: a stdlib module that starts a child through something nothing declares.

`llb.quality.gpu_guard_spawn_reach` reads the stdlib and reports which modules start a child and
through which names. This module weighs that against the declared name surface and refuses three
shapes:

- a module reaching a name the declared surface does not carry at all -- the `posix` /
  `_posixsubprocess` / `_winapi` level, below anything the denial can patch -- unless
  `DECLARED_REACHERS` excuses it;
- an excuse that claims the reach sits behind a seam which is no longer patched;
- a scan that found NO module starting a child, which means the tree was not read rather than that
  the stdlib starts none. A source scan that silently reads nothing is the one way this check could
  pass for free, so it is the one result treated as a failure rather than as a clean bill.

A module reaching a name the surface DOES carry is never a finding, covered or residual: the
declaration already carries that decision, and repeating it per call site would only mean two places
to keep in step.
"""

from collections.abc import Mapping, Sequence

from llb.quality.gpu_guard_spawn_reach import DECLARED_REACHERS, ModuleReach, stdlib_spawn_reaches
from llb.quality.gpu_guard_spawn_surface import (
    COVERAGE_THROUGH,
    DECLARED_SPAWN_SURFACE,
    ObservedSurface,
    SpawnCoverage,
)
from llb.quality.gpu_guard_spawn_surface_audit import (
    PROBLEM_UNREACHED,
    SurfaceFinding,
    reaches_a_seam,
)

PROBLEM_UNCOVERED_REACH = "uncovered-reach"
PROBLEM_UNSCANNED = "unscanned"


def audit_spawn_reach(
    reaches: Sequence[ModuleReach] | None = None,
    declared: Mapping[str, SpawnCoverage] = DECLARED_SPAWN_SURFACE,
    reachers: Mapping[str, SpawnCoverage] = DECLARED_REACHERS,
    surface: ObservedSurface | None = None,
) -> tuple[SurfaceFinding, ...]:
    """Stdlib modules that start a child through something the declared surface does not name."""
    scanned = reaches if reaches is not None else stdlib_spawn_reaches()
    if not scanned:
        return (
            SurfaceFinding(
                "<stdlib>",
                PROBLEM_UNSCANNED,
                "no module in the scanned tree starts a child, which means the tree was not read "
                "rather than that the stdlib starts none",
            ),
        )
    observed = surface if surface is not None else ObservedSurface.read()
    findings = (_reach_finding(reach, observed, declared, reachers) for reach in scanned)
    return tuple(finding for finding in findings if finding is not None)


def absent_reachers(
    reaches: Sequence[ModuleReach], reachers: Mapping[str, SpawnCoverage] = DECLARED_REACHERS
) -> tuple[str, ...]:
    """Excused modules the scan no longer finds -- an excuse that has outlived what it excused."""
    scanned = {reach.path for reach in reaches}
    return tuple(path for path in reachers if path not in scanned)


def _reach_finding(
    reach: ModuleReach,
    observed: ObservedSurface,
    declared: Mapping[str, SpawnCoverage],
    reachers: Mapping[str, SpawnCoverage],
) -> SurfaceFinding | None:
    """A module's uncovered starts, weighed against whatever excuses them."""
    uncovered = tuple(name for name in reach.primitives if name not in declared)
    if not uncovered:
        return None
    excuse = reachers.get(reach.path)
    if excuse is None:
        return SurfaceFinding(
            reach.path,
            PROBLEM_UNCOVERED_REACH,
            f"starts a child through {', '.join(uncovered)}, which the declared spawn surface does "
            "not name -- so an unmarked test reaching this module keeps the device",
        )
    if excuse.kind == COVERAGE_THROUGH and not reaches_a_seam(
        excuse.through or "", observed, declared
    ):
        return SurfaceFinding(
            reach.path,
            PROBLEM_UNREACHED,
            f"excused as reached through {excuse.through}, which no patched seam covers",
        )
    return None
