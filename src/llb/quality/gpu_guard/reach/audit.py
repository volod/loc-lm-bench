"""The refusal: a stdlib module that starts a child through something nothing declares.

`llb.quality.gpu_guard.reach.scan` reads the stdlib and reports which modules start a child and
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
beside it, which this host can import and the scan did not read. That list is per top-level name, so
the same refusal reads one level down off the package directories themselves: a submodule with a
cached module and no source is the same stripped layout inside a package that otherwise counts as
read.

A module reaching a name the surface DOES carry is never a finding, covered or residual: the
declaration already carries that decision, and repeating it per call site would only mean two places
to keep in step.

The same question of the INSTALLED packages, plus the two refusals only a dependency can earn, is
`llb.quality.gpu_guard.reach.installed_audit`. It reuses the machinery here -- the excuse
lookup, the reach finding, the unread-module problem -- and differs only in the granularity of the
excuse and in having an archive on the import path to account for.
"""

from collections.abc import Mapping

from llb.quality.gpu_guard.reach.scan import (
    DECLARED_REACHERS,
    ModuleReach,
    SpawnScan,
    stdlib_spawn_reaches,
)
from llb.quality.gpu_guard.reach.coverage import ReadCoverage, stdlib_read_coverage
from llb.quality.gpu_guard.reach.installed import PackageReacher
from llb.quality.gpu_guard.surface import (
    COVERAGE_THROUGH,
    DECLARED_SPAWN_SURFACE,
    ObservedSurface,
    SpawnCoverage,
)
from llb.quality.gpu_guard.surface_audit import (
    PROBLEM_UNREACHED,
    SurfaceFinding,
    reaches_a_seam,
)

PROBLEM_UNCOVERED_REACH = "uncovered-reach"
PROBLEM_UNSCANNED = "unscanned"
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

    The same class one level down is refused on the same evidence. The declared list is per
    top-level name, so a package that ships its `__init__.py` and not its submodules classifies as
    read; `compiled_only_submodules` compares each walked package's `.py` stems against its cached
    ones, and a submodule with a `.pyc` and no `.py` is exactly the stripped layout above -- the only
    difference being that the finding names `multiprocessing.util` rather than `multiprocessing`.

    `archived` and `archived_submodules` are that same refusal for the layout both of those reads
    are too directory-shaped to see: a stdlib imported from a zip, where nothing is under the root
    to compare and every name would otherwise fall through to `absent` -- recorded as *this host
    does not ship it* for a module the interpreter imports on demand. The name list of the archive
    says the module is there and says nothing about whether it starts a child, so it is refused as
    unread rather than counted as read.
    """
    measured = coverage if coverage is not None else stdlib_read_coverage(scan)
    return (
        *(
            SurfaceFinding(
                name,
                PROBLEM_UNREAD_MODULE,
                "a compiled module under the stdlib root with no source beside it, so this host can "
                "import it and the scan could not read whether it starts a child",
            )
            for name in measured.compiled_only
        ),
        *(
            SurfaceFinding(
                name,
                PROBLEM_UNREAD_MODULE,
                "a compiled submodule of a package the scan otherwise read, with no source beside "
                "it, so this host can import it and the scan could not read whether it starts a "
                "child",
            )
            for name in measured.compiled_only_submodules
        ),
        *(
            SurfaceFinding(
                name,
                PROBLEM_UNREAD_MODULE,
                "ships inside an archive this interpreter imports from, which the directory-shaped "
                "scan does not walk, so this host can import it and the scan could not read "
                f"whether it starts a child (archives read: {', '.join(measured.archives)})",
            )
            for name in measured.archived
        ),
        *(
            SurfaceFinding(
                name,
                PROBLEM_UNREAD_MODULE,
                "an archived submodule of a package the scan otherwise read off the directory tree, "
                "with no source there, so this host can import it and the scan could not read "
                f"whether it starts a child (archives read: {', '.join(measured.archives)})",
            )
            for name in measured.archived_submodules
        ),
    )


def absent_reachers(
    scan: SpawnScan,
    reachers: Mapping[str, SpawnCoverage] | Mapping[str, PackageReacher] = DECLARED_REACHERS,
) -> tuple[str, ...]:
    """Excused names the scan no longer finds -- an excuse that has outlived what it excused."""
    scanned = {reach.path for reach in scan.reaches} | {
        top_level_package(reach) for reach in scan.reaches
    }
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
    excuse = reachers.get(reach.path) or reachers.get(top_level_package(reach))
    if excuse is None:
        return SurfaceFinding(
            reach.location,
            PROBLEM_UNCOVERED_REACH,
            f"starts a child through {', '.join(uncovered)}, which the declared spawn surface does "
            "not name -- so an unmarked test reaching this module keeps the device",
        )
    if excuse.kind == COVERAGE_THROUGH and not reaches_a_seam(
        excuse.through or "", observed, declared
    ):
        return SurfaceFinding(
            reach.location,
            PROBLEM_UNREACHED,
            f"excused as reached through {excuse.through}, which no patched seam covers",
        )
    return None


def top_level_package(reach: ModuleReach) -> str:
    """The top-level package a module belongs to -- the unit a dependency is declared at.

    Read off the relative path whether that path came out of a directory or out of an archive, so a
    zipped `pkg/backend/start.py` is looked up under the same `pkg` an operator declared.
    """
    return reach.path.split("/")[0]
