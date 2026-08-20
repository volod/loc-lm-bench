"""What the DEPENDENCIES start, over the whole import path -- directory tree and archives alike.

The installed half of the reach measurement, asked for a narrower question than the stdlib half.
A dependency calling `subprocess.Popen` says nothing the declaration does not already say, and
looking for it means parsing the whole tree, so `installed_spawn_reaches` uses the `below_the_seams`
alphabet: only the starts that go past every patchable name. Those are declared per package rather
than per file (`DECLARED_PACKAGE_REACHERS`), because a release moves its modules and the decision is
about the dependency. Package granularity is the right unit for surviving a release bump and the
wrong unit for a residual -- it excuses every module in the package, so a future `joblib` that starts
children a second way, from a file the declaration never saw, would be excused by a line written
about `loky`. What narrows it without per-file churn is the MEASUREMENT each declaration carries
(`PackageReacher`): the primitives and the file count it was written against, so a package whose
reach grows past them is reported by `gpu_guard_spawn_reach_audit.outgrown_reachers` rather than
silently covered.

A tree is not the whole import path, though, and this scan reads one directory. A dependency that
ships zipped -- a zipped egg, a `--zip-ok` install, any `sys.path` entry that is an archive rather
than a directory -- has no package directory to walk, so `rglob("*.py")` finds nothing in it,
nothing is parsed, and `audit_installed_reach` would return clean for a venv half of which it never
opened. `llb.quality.gpu_guard.reach.installed_archive` is the other half of the pass and
states the reading it decided on. Nor is an archive the only entry that sits outside the tree: a
`.pth` file adds DIRECTORIES to the path as well, which is how this repo's own `src` is importable
at all, and `llb.quality.gpu_guard.reach.installed_sites` reads those.
`installed_spawn_reaches` folds all three into ONE `SpawnScan`, so the counts add up over
everything this interpreter can import from.

The refusal that leaves is `gpu_guard_spawn_reach_audit.unread_archived_packages`, at PACKAGE
granularity because that is the unit the excuses are written at and the unit an operator acts on: a
`.pyc`-only egg is one line naming the modules it hid, not one line per module. An archived package
`DECLARED_PACKAGE_REACHERS` already names is not refused at all -- the declaration is the decision
that the package starts children and that this is accepted, so refusing it as unmeasured would be
the same finding twice. Its measurement then only shrinks, and shrinking is not growth, so
`outgrown_reachers` stays quiet about it.
"""

import sys
import sysconfig
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from llb.quality.gpu_guard.reach.scan import (
    LOW_LEVEL_STARTS,
    SpawnScan,
    module_triggers,
    spawn_primitives,
    spawn_scan,
)
from llb.quality.gpu_guard.reach.archive import openable_archives
from llb.quality.gpu_guard.reach.installed_archive import installed_archives, with_archives
from llb.quality.gpu_guard.reach.installed_sites import (
    SitePathEntry,
    SitePathReading,
    site_path_entries,
    with_path_entries,
)
from llb.quality.gpu_guard.surface import COVERAGE_RESIDUAL, SpawnCoverage

# The two below-the-seams names every package declaration here was measured against. Named once so
# a declaration records what it was written on rather than restating the alphabet.
_VENDORED_MULTIPROCESSING_STARTS = ("_posixsubprocess.fork_exec", "_winapi.CreateProcess")


@dataclass(frozen=True)
class PackageReacher:
    """A dependency's excuse, plus the reach that excuse was MEASURED against.

    The excuse is declared per package because a release moves its modules around, and that is also
    what makes it too wide: it covers every module in the package, including a backend a future
    release adds. The measurement is the narrowing -- the primitives resolved and the number of files
    that resolved them when the reason was written -- so a package that grows a second way to start a
    child arrives as a line to re-read instead of as silence. Shrinking is not growth: a release that
    drops a backend, or a host that installs a slimmer build, is not a decision to revisit.
    """

    coverage: SpawnCoverage
    primitives: tuple[str, ...]
    files: int

    def __post_init__(self) -> None:
        if not self.primitives or self.files < 1:
            raise ValueError(
                "a package excuse must record the primitives and file count it was measured against"
            )


# Installed packages are declared per PACKAGE rather than per file: a release moves its modules
# around, and the decision an operator makes is about the dependency, not about a path inside it.
DECLARED_PACKAGE_REACHERS: Mapping[str, PackageReacher] = {
    "joblib": PackageReacher(
        SpawnCoverage(
            COVERAGE_RESIDUAL,
            reason="vendors `loky`, whose `backend/fork_exec.py` calls "
            "`_posixsubprocess.fork_exec` and whose Windows backend and resource tracker call "
            "`_winapi.CreateProcess` -- a private copy of the low-level bypass closed at the "
            "stdlib's public `multiprocessing.util.spawnv_passfds` seam",
        ),
        primitives=_VENDORED_MULTIPROCESSING_STARTS,
        files=3,
    ),
    "multiprocess": PackageReacher(
        SpawnCoverage(
            COVERAGE_RESIDUAL,
            reason="a `dill`-based fork of `multiprocessing`, so its private `util.spawnv_passfds` "
            "still reaches `_posixsubprocess.fork_exec` below the stdlib seam, plus the "
            "`popen_spawn_win32` half",
        ),
        primitives=_VENDORED_MULTIPROCESSING_STARTS,
        files=2,
    ),
}


def package_coverage(
    reachers: Mapping[str, PackageReacher] = DECLARED_PACKAGE_REACHERS,
) -> Mapping[str, SpawnCoverage]:
    """The excuse half of the package declarations, so both tables read through one lookup."""
    return {package: declared.coverage for package, declared in reachers.items()}


def below_the_seams() -> Mapping[str, frozenset[str]]:
    """The alphabet for a tree that only has to be checked for reaches PAST the declared surface.

    An installed package calling `subprocess.Popen` says nothing the declaration does not already
    say, and looking for it costs the whole tree: on this host, scanning site-packages for the
    covered names too means parsing 7420 files instead of 301 (measured). What is worth finding is a
    package that goes below every patchable name, so the alphabet is the C modules only.

    `nt` is deliberately absent where `posix` is present: it is the Windows twin of the same names,
    and its two-letter module name matches too much text to prefilter on, so including it would cost
    a full-tree parse for an alias of names `os` re-exports on a platform whose denial mechanism is
    already a declared residual.
    """
    return {
        "posix": frozenset(spawn_primitives()["os"]),
        **LOW_LEVEL_STARTS,
    }


def installed_spawn_reaches(
    root: Path | None = None,
    primitives: Mapping[str, frozenset[str]] | None = None,
    archives: Iterable[Path] | None = None,
    sites: Iterable[Path] | None = None,
) -> SpawnScan:
    """The installed packages, read for the starts that go BELOW every name the denial patches.

    One pass over the directory tree, one over each archive the same import path carries, and one
    over each extra DIRECTORY a `.pth` file adds to it, folded into a single scan: the counts add up
    over all three, so neither a venv that ships zipped nor the editable source tree this repo
    itself installs is a part of the path the audit never opened.

    `sys.path` is consulted only for the DEFAULT root. A caller naming a tree is asking about that
    tree, so archive discovery is scoped to it -- otherwise a fabricated case would answer partly
    out of the interpreter that happens to be running the test. The `.pth` files need no such rule:
    they live IN the root, so reading them is already scoped to the tree the caller named.
    """
    tree = root if root is not None else Path(sysconfig.get_paths()["purelib"])
    alphabet = primitives if primitives is not None else below_the_seams()
    triggers = module_triggers(alphabet)
    found = (
        openable_archives(archives)
        if archives is not None
        else installed_archives(tree, sys.path if root is None else None)
    )
    path_reading = (
        site_path_entries(tree)
        if sites is None
        else SitePathReading(tuple(SitePathEntry(path.resolve()) for path in sites), ())
    )
    return with_path_entries(
        with_archives(spawn_scan(tree, alphabet, triggers), tree, found, alphabet),
        path_reading.entries,
        alphabet,
        triggers,
        path_reading.unread,
    )
