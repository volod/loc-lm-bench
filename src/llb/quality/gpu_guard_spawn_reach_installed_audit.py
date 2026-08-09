"""The refusals only a DEPENDENCY can earn -- the installed half of the reach audit.

`llb.quality.gpu_guard_spawn_reach_audit` refuses a module that starts a child through something
nothing declares, and `audit_installed_reach` asks that same question of the installed packages.
Only the granularity of the excuse differs -- per package rather than per file, since a release
moves its modules around -- and an excuse is looked up as the exact path first and then the
top-level package, so the stdlib and package tables read through one lookup.

Two shapes are left that the stdlib table cannot have, and this module is both:

- a declared package whose reach has GROWN past the primitives and file count its excuse was
  measured against. Package granularity survives a release bump and, for exactly that reason,
  excuses a backend the declaration never saw; `outgrown_reachers` is what turns that widening into
  a line to re-read rather than silence under the old reason.
- a zipped dependency the scan could not read. `gpu_guard_spawn_reach_installed_archive` parses the
  archives on the import path wherever they carry source, so `unread_archived_packages` refuses only
  what is left after that -- a module a zip ships compiled with no copy on disk. It is the archive
  half of the same `unread-module` problem the stdlib coverage raises, named per package because
  that is the unit the excuses are written at and the unit an operator acts on.
"""

from collections.abc import Mapping

from llb.quality.gpu_guard_spawn_reach import ModuleReach, SpawnScan
from llb.quality.gpu_guard_spawn_reach_audit import (
    PROBLEM_UNREAD_MODULE,
    audit_spawn_reach,
    top_level_package,
)
from llb.quality.gpu_guard_spawn_reach_installed import (
    DECLARED_PACKAGE_REACHERS,
    PackageReacher,
    installed_spawn_reaches,
    package_coverage,
)
from llb.quality.gpu_guard_spawn_surface import (
    DECLARED_SPAWN_SURFACE,
    ObservedSurface,
    SpawnCoverage,
)
from llb.quality.gpu_guard_spawn_surface_audit import SurfaceFinding

PROBLEM_OUTGROWN_REACH = "outgrown-reach"


def audit_installed_reach(
    scan: SpawnScan | None = None,
    declared: Mapping[str, SpawnCoverage] = DECLARED_SPAWN_SURFACE,
    reachers: Mapping[str, PackageReacher] = DECLARED_PACKAGE_REACHERS,
    surface: ObservedSurface | None = None,
) -> tuple[SurfaceFinding, ...]:
    """The same question of the installed packages, whose excuses are declared per package.

    Plus the two the stdlib table cannot ask: whether a declared package still reaches only what its
    excuse was measured on, since the package granularity that survives a release bump is also what
    would excuse a backend that release adds -- and whether a dependency that ships ZIPPED left
    anything unread, which `unread_archived_packages` refuses.
    """
    scanned = scan if scan is not None else installed_spawn_reaches()
    return (
        *audit_spawn_reach(scanned, declared, package_coverage(reachers), surface),
        *outgrown_reachers(scanned, reachers),
        *unread_archived_packages(scanned, reachers),
    )


def unread_archived_packages(
    scan: SpawnScan, reachers: Mapping[str, PackageReacher] = DECLARED_PACKAGE_REACHERS
) -> tuple[SurfaceFinding, ...]:
    """Zipped dependencies the scan could not read, one finding per package.

    The archives on the import path are parsed where they carry source, so what is left here is the
    part that could not be: a module a zip ships compiled, with no copy in the directory tree, which
    this host can import and nothing measured. Refused for the same reason a `.pyc` under the stdlib
    root is -- the name says the module is there and says nothing about whether it starts a child.

    Per PACKAGE, because that is the unit the excuses are written at and the unit an operator acts
    on: a `.pyc`-only egg is one line naming the modules it hid, not one line per module. A package
    `DECLARED_PACKAGE_REACHERS` already names is not a finding, because the declaration is the
    decision that this package starts children and that it is accepted -- refusing it as unmeasured
    would be the same finding twice.
    """
    grouped: dict[str, list[str]] = {}
    for name in scan.unread_archived:
        grouped.setdefault(name.split(".")[0], []).append(name)
    return tuple(
        SurfaceFinding(
            package,
            PROBLEM_UNREAD_MODULE,
            "ships inside an archive on this interpreter's import path with no source to parse "
            f"({', '.join(names)}), so this host can import it and the scan could not read whether "
            f"it starts a child (archives read: {', '.join(scan.archives)})",
        )
        for package, names in sorted(grouped.items())
        if package not in reachers
    )


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


def _package_reaches(scan: SpawnScan) -> dict[str, list[ModuleReach]]:
    """The scan regrouped onto the unit the excuses are written at."""
    grouped: dict[str, list[ModuleReach]] = {}
    for reach in scan.reaches:
        grouped.setdefault(top_level_package(reach), []).append(reach)
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
        listed = ", ".join(module.location for module in modules)
        clauses.append(
            f"now starts children from {len(modules)} files rather than the {declared.files} its "
            f"excuse was measured on ({listed})"
        )
    return (
        f"{'; '.join(clauses)} -- the package excuse still covers it, so re-read that reason "
        "against what the package does now and re-measure it"
    )
