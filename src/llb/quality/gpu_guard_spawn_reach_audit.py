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

`audit_read_coverage` closes the middle of that last one. An empty read is the degenerate end; a
host that ships PART of its library as source reads exactly like a library that starts no children,
and the file count cannot tell, because it says how much was read and not what was missed.
`gpu_guard_spawn_reach_coverage` measures the reading against `sys.stdlib_module_names` and
classifies every unread name; this module refuses the one class that is neither compiled in, an
extension, nor declared sourceless -- a module with a `.pyc` under the stdlib root and no `.py`
beside it, which this host can import and the scan did not read.

A module reaching a name the surface DOES carry is never a finding, covered or residual: the
declaration already carries that decision, and repeating it per call site would only mean two places
to keep in step.

`audit_installed_reach` asks the same question of the installed packages. Only the granularity of
the excuse differs -- per package rather than per file, since a release moves its modules around --
and an excuse is looked up as the exact path first, then the top-level package, so both tables read
through one lookup. It refuses a fourth shape the stdlib table cannot have: a declared package whose
reach has GROWN past the primitives and file count its excuse was measured against. Package
granularity survives a release bump and, for exactly that reason, excuses a backend the declaration
never saw; `outgrown_reachers` is what turns that widening into a line to re-read.
"""

from collections.abc import Mapping

from llb.quality.gpu_guard_spawn_reach import (
    DECLARED_PACKAGE_REACHERS,
    DECLARED_REACHERS,
    ModuleReach,
    PackageReacher,
    SpawnScan,
    installed_spawn_reaches,
    package_coverage,
    stdlib_spawn_reaches,
)
from llb.quality.gpu_guard_spawn_reach_coverage import ReadCoverage, stdlib_read_coverage
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
PROBLEM_OUTGROWN_REACH = "outgrown-reach"
PROBLEM_UNREAD_MODULE = "unread-module"


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
    reachers: Mapping[str, PackageReacher] = DECLARED_PACKAGE_REACHERS,
    surface: ObservedSurface | None = None,
) -> tuple[SurfaceFinding, ...]:
    """The same question of the installed packages, whose excuses are declared per package.

    Plus the one the stdlib table cannot ask: whether a declared package still reaches only what its
    excuse was measured on, since the package granularity that survives a release bump is also what
    would excuse a backend that release adds.
    """
    scanned = scan if scan is not None else installed_spawn_reaches()
    return (
        *audit_spawn_reach(scanned, declared, package_coverage(reachers), surface),
        *outgrown_reachers(scanned, reachers),
    )


def audit_read_coverage(
    scan: SpawnScan, coverage: ReadCoverage | None = None
) -> tuple[SurfaceFinding, ...]:
    """Declared stdlib modules this host can import and the scan read no source for.

    Only the compiled-only class is refused. A module that is compiled into the interpreter, ships
    as an extension, or is declared sourceless has no `.py` BY CONSTRUCTION -- two thirds of
    `sys.stdlib_module_names` are one of those here, and refusing them is the naive gate that fails
    on every host. A module absent from the root entirely is not refused either: this host cannot
    import what it does not have, so the claim holds vacuously for it. What is left is the layout
    this exists for -- a frozen, zipped, or source-stripped stdlib, where the module IS importable
    and the scan could not see whether it starts a child.
    """
    measured = coverage if coverage is not None else stdlib_read_coverage(scan)
    return tuple(
        SurfaceFinding(
            name,
            PROBLEM_UNREAD_MODULE,
            "a compiled module under the stdlib root with no source beside it, so this host can "
            "import it and the scan could not read whether it starts a child",
        )
        for name in measured.compiled_only
    )


def absent_reachers(
    scan: SpawnScan,
    reachers: Mapping[str, SpawnCoverage] | Mapping[str, PackageReacher] = DECLARED_REACHERS,
) -> tuple[str, ...]:
    """Excused names the scan no longer finds -- an excuse that has outlived what it excused."""
    scanned = {reach.path for reach in scan.reaches} | {_package(reach) for reach in scan.reaches}
    return tuple(name for name in reachers if name not in scanned)


def outgrown_reachers(
    scan: SpawnScan, reachers: Mapping[str, PackageReacher] = DECLARED_PACKAGE_REACHERS
) -> tuple[SurfaceFinding, ...]:
    """Declared packages whose reach has grown past what their excuse was measured against.

    Growth only: a package that reaches the same way from FEWER files -- a release that drops a
    backend, a slimmer build, a host that vendors less -- is not a decision to revisit, and an
    excuse that stops matching anything at all is what `absent_reachers` reports. An UNdeclared
    package is not this function's finding either; `audit_installed_reach` already refuses it.
    """
    findings = (
        _growth_finding(package, modules, reachers[package])
        for package, modules in _package_reaches(scan).items()
        if package in reachers
    )
    return tuple(finding for finding in findings if finding is not None)


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


def _package_reaches(scan: SpawnScan) -> dict[str, list[ModuleReach]]:
    """The scan regrouped onto the unit the excuses are written at."""
    grouped: dict[str, list[ModuleReach]] = {}
    for reach in scan.reaches:
        grouped.setdefault(_package(reach), []).append(reach)
    return grouped


def _growth_finding(
    package: str, modules: list[ModuleReach], declared: PackageReacher
) -> SurfaceFinding | None:
    """How one package's reach has widened since its excuse was written, or None if it has not."""
    widened = tuple(
        sorted(
            {name for module in modules for name in module.primitives} - set(declared.primitives)
        )
    )
    if not widened and len(modules) <= declared.files:
        return None
    return SurfaceFinding(
        package, PROBLEM_OUTGROWN_REACH, _growth_detail(modules, widened, declared)
    )


def _growth_detail(
    modules: list[ModuleReach], widened: tuple[str, ...], declared: PackageReacher
) -> str:
    clauses = []
    if widened:
        clauses.append(f"now reaches {', '.join(widened)}, which its excuse was not measured on")
    if len(modules) > declared.files:
        listed = ", ".join(module.path for module in modules)
        clauses.append(
            f"now starts children from {len(modules)} files rather than the {declared.files} its "
            f"excuse was measured on ({listed})"
        )
    return (
        f"{'; '.join(clauses)} -- the package excuse still covers it, so re-read that reason "
        "against what the package does now and re-measure it"
    )
