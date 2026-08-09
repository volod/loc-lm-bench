"""The refusal: a stdlib module that starts a child through something nothing declares.

`llb.quality.gpu_guard_spawn_reach` reads the stdlib and reports which modules start a child and
through which names. This module weighs that against the declared name surface and refuses three
shapes:

- a module reaching a name the declared surface does not carry at all -- the `posix` /
  `_posixsubprocess` / `_winapi` level, below anything the denial can patch -- unless
  `DECLARED_REACHERS` excuses it;
- an excuse that claims the reach sits behind a seam which is no longer patched;
- a scan that read NO source, which says where the tree is rather than what is in it. A source scan
  that silently reads nothing is the one way this check could pass for free, so an empty read is
  treated as a failure rather than as a clean bill.

A module reaching a name the surface DOES carry is never a finding, covered or residual: the
declaration already carries that decision, and repeating it per call site would only mean two places
to keep in step.

`audit_installed_reach` asks the same question of the installed packages. Only the granularity of
the excuse differs -- per package rather than per file, since a release moves its modules around --
and an excuse is looked up as the exact path first, then the top-level package, so both tables read
through one lookup.
"""

from collections.abc import Mapping

from llb.quality.gpu_guard_spawn_reach import (
    DECLARED_PACKAGE_REACHERS,
    DECLARED_REACHERS,
    ModuleReach,
    SpawnScan,
    installed_spawn_reaches,
    stdlib_spawn_reaches,
)
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
    scan: SpawnScan | None = None,
    declared: Mapping[str, SpawnCoverage] = DECLARED_SPAWN_SURFACE,
    reachers: Mapping[str, SpawnCoverage] = DECLARED_REACHERS,
    surface: ObservedSurface | None = None,
) -> tuple[SurfaceFinding, ...]:
    """Stdlib modules that start a child through something the declared surface does not name."""
    scanned = scan if scan is not None else stdlib_spawn_reaches()
    if not scanned.files_read:
        return (_unscanned(scanned),)
    observed = surface if surface is not None else ObservedSurface.read()
    findings = (_reach_finding(reach, observed, declared, reachers) for reach in scanned.reaches)
    return tuple(finding for finding in findings if finding is not None)


def audit_installed_reach(
    scan: SpawnScan | None = None,
    declared: Mapping[str, SpawnCoverage] = DECLARED_SPAWN_SURFACE,
    reachers: Mapping[str, SpawnCoverage] = DECLARED_PACKAGE_REACHERS,
    surface: ObservedSurface | None = None,
) -> tuple[SurfaceFinding, ...]:
    """The same question of the installed packages, whose excuses are declared per package."""
    return audit_spawn_reach(
        scan if scan is not None else installed_spawn_reaches(), declared, reachers, surface
    )


def absent_reachers(
    scan: SpawnScan, reachers: Mapping[str, SpawnCoverage] = DECLARED_REACHERS
) -> tuple[str, ...]:
    """Excused names the scan no longer finds -- an excuse that has outlived what it excused."""
    scanned = {reach.path for reach in scan.reaches} | {_package(reach) for reach in scan.reaches}
    return tuple(name for name in reachers if name not in scanned)


def _unscanned(scan: SpawnScan) -> SurfaceFinding:
    return SurfaceFinding(
        scan.root,
        PROBLEM_UNSCANNED,
        "the scan read no source at all, so finding nothing that starts a child says where the "
        "tree is rather than what is in it",
    )


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
    excuse = reachers.get(reach.path) or reachers.get(_package(reach))
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


def _package(reach: ModuleReach) -> str:
    """The top-level package a module belongs to -- the unit a dependency is declared at."""
    return reach.path.split("/")[0]
