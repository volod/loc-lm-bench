"""Which stdlib module starts a child, MEASURED -- so the enumerated surface is not two modules on
faith.

`llb.quality.gpu_guard_spawn_surface` enumerates the process-starting names of `os` and
`subprocess`, and every other way a test reaches a child is covered because the helper it calls
resolves a name in one of those two. That last sentence was the one claim the name check left
standing: `pty.spawn` forks and execs, `asyncio`'s unix transport starts a `Popen`,
`multiprocessing.util.spawnv_passfds` does neither, and nothing said which of those is which.

So this module reads the stdlib instead of asserting about it. Every `*.py` under the stdlib root
goes through `llb.quality.gpu_guard_spawn_source`, which resolves its process-starting CALL SITES
through that module's own imports (`os.fork`, `from subprocess import Popen`,
`import os as operating`), against an alphabet that is the declared
surface plus the C modules underneath it -- `posix` / `nt` (what `os` re-exports), and
`_posixsubprocess` / `_winapi` (what `subprocess` and `multiprocessing` call below any patchable
name). A module reaching a DECLARED name needs nothing: the declaration already carries that
decision, covered or residual. A module reaching something the declared surface does not name is
excused by `DECLARED_REACHERS` here, or refused by `llb.quality.gpu_guard_spawn_reach_audit`.

The INSTALLED packages are read the same way and for a narrower question, which
`llb.quality.gpu_guard_spawn_reach_installed` owns: a dependency calling `subprocess.Popen` says
nothing the declaration does not already say, so that scan looks only for the starts that go past
every patchable name, over the venv's directory tree plus the archives on `sys.path`.

CPython's own regression suite (`test/`, `*/tests/`, `idlelib/idle_test`) is left out by a stated
rule: it is a corpus that starts children on purpose, is not runtime code any llb path imports, and
costs 4s and an extra declaration to include.

The measurements are the interesting half. On CPython 3.13 the stdlib scan finds 25 modules that
start a child, of which 23 resolve a name the denial covers; the exceptions are the residuals
already on the record -- `multiprocessing/util.py` and `multiprocessing/popen_spawn_win32.py` --
plus `subprocess.py`, whose low-level starts sit BEHIND the `Popen` seam. Two modules is the right
enumerated NAME surface, and that is now a result rather than a claim. Over this host's
site-packages (40119 files), a one-off full-alphabet pass found 362 packages that start a child and
exactly 5 files that go below the seams, all in two packages: `joblib`'s vendored `loky` (3 files)
and `multiprocess` (2). Both are private copies of the `multiprocessing` residual, and neither is
closable from here -- so they are declared, measured against those files, and a THIRD package
arriving is what the installed module refuses.
"""

import sysconfig
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from llb.quality.gpu_guard_spawn_source import source_reaches
from llb.quality.gpu_guard_spawn_surface import (
    COVERAGE_NOT_A_SPAWN,
    COVERAGE_RESIDUAL,
    COVERAGE_THROUGH,
    DECLARED_SPAWN_SURFACE,
    SpawnCoverage,
)

# Directory names that mark CPython's own tests, and the third-party tree that may live beside the
# stdlib. Matched on directory segments, so a module called `tests.py` is still scanned.
_EXCLUDED_SEGMENTS = frozenset({"test", "tests", "idle_test", "site-packages"})
# The C modules `os` and `subprocess` are written on. A caller that imports one of these directly
# reaches past every name the denial can patch, which is what makes them worth scanning for -- and
# what makes them the whole alphabet of the installed scan, which reads them from here.
LOW_LEVEL_STARTS: Mapping[str, frozenset[str]] = {
    "_posixsubprocess": frozenset({"fork_exec"}),
    "_winapi": frozenset({"CreateProcess"}),
}

DECLARED_REACHERS: Mapping[str, SpawnCoverage] = {
    "subprocess.py": SpawnCoverage(
        COVERAGE_THROUGH,
        through="subprocess.Popen",
        reason="its `_posixsubprocess` / `_winapi` starts are reached only from inside `Popen`",
    ),
    "multiprocessing/util.py": SpawnCoverage(
        COVERAGE_RESIDUAL,
        reason="`spawnv_passfds` calls `_posixsubprocess.fork_exec` with no environment list -- the "
        "`spawn` / `forkserver` residual, declared as a start method in gpu_guard_spawn_surface",
    ),
    "multiprocessing/popen_spawn_win32.py": SpawnCoverage(
        COVERAGE_RESIDUAL,
        reason="`_winapi.CreateProcess` is the Windows half of the same residual, and the denial's "
        "`os.system` mechanism is POSIX-shell specific anyway",
    ),
}


@dataclass(frozen=True)
class ModuleReach:
    """One module and the process-starting names its source resolves.

    `archive` names the zip the source was read out of, and is empty for the ordinary case of a file
    on disk. `path` stays the path RELATIVE to whatever contained it either way, because it is what
    the excuse tables are keyed on -- the top-level package of an archived `pkg/backend/start.py` is
    the same `pkg` an operator declared.
    """

    path: str
    primitives: tuple[str, ...]
    archive: str = ""

    @property
    def location(self) -> str:
        """The path plus the archive it came out of -- what an operator has to open to check it."""
        return f"{self.path} (in {self.archive})" if self.archive else self.path

    def __str__(self) -> str:
        return f"{self.location} -> {', '.join(self.primitives)}"


@dataclass(frozen=True)
class SpawnScan:
    """One pass over a tree: what it read, and what it found starting children.

    What it READ is the half that keeps the result honest. A tree whose source is absent -- a frozen
    or `.pyc`-only install, a path that is not there at all -- yields no reaches, which is the same
    answer as a tree where nothing starts a child. `files_read` tells those two apart at the
    degenerate end; `modules_read` -- the top-level names the pass parsed at least one file of --
    is what `gpu_guard_spawn_reach_coverage` weighs against `sys.stdlib_module_names` to tell them
    apart in the middle, where a host ships half its library as source.

    The last two fields carry what an ARCHIVE on the import path contributed, and are empty for a
    directory-only pass: `archives` is what was opened, and `unread_archived` the dotted names those
    archives ship with no source to parse. `llb.quality.gpu_guard_spawn_reach_installed` fills them.
    """

    root: str
    files_read: int
    modules_read: tuple[str, ...]
    reaches: tuple[ModuleReach, ...]
    archives: tuple[str, ...] = ()
    unread_archived: tuple[str, ...] = ()


def spawn_primitives(
    declared: Mapping[str, SpawnCoverage] = DECLARED_SPAWN_SURFACE,
) -> Mapping[str, frozenset[str]]:
    """The alphabet the scan looks for: module -> the calls that start a process.

    Taken from the declared surface rather than restated, so a name added there is a name the scan
    starts recognizing, plus the C modules below it -- including `posix` / `nt`, whose entry points
    `os` re-exports and which a caller can therefore reach without going through `os` at all.
    """
    families: dict[str, set[str]] = {}
    for name, coverage in declared.items():
        if coverage.kind == COVERAGE_NOT_A_SPAWN:
            continue
        module, _, attribute = name.partition(".")
        families.setdefault(module, set()).add(attribute)
    os_family = frozenset(families.get("os", set()))
    return {
        **{module: frozenset(attributes) for module, attributes in families.items()},
        "posix": os_family,
        "nt": os_family,
        **LOW_LEVEL_STARTS,
    }


def spawn_scan(
    root: Path,
    alphabet: Mapping[str, frozenset[str]],
    triggers: Sequence[bytes],
) -> SpawnScan:
    """Read one tree: every module whose source starts a child, and how many files were read.

    A trigger is a byte string whose ABSENCE proves a file cannot contain a call in the alphabet,
    which is what lets a file be skipped without parsing it. Attribute names qualify (`os.fork()`
    names `fork`), and so do module names (calling `posix.fork` means naming `posix`) -- each scan
    passes whichever set is cheaper for the tree it reads.
    """
    found: list[ModuleReach] = []
    modules: set[str] = set()
    read = 0
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        source = None if is_excluded(relative) else _read(path)
        if source is None:
            continue
        read += 1
        modules.add(relative.split("/")[0].removesuffix(".py"))
        if not any(trigger in source for trigger in triggers):
            continue
        reached = source_reaches(source, alphabet)
        if reached:
            found.append(ModuleReach(path=relative, primitives=reached))
    return SpawnScan(
        root=str(root),
        files_read=read,
        modules_read=tuple(sorted(modules)),
        reaches=tuple(found),
    )


def stdlib_spawn_reaches(
    root: Path | None = None, primitives: Mapping[str, frozenset[str]] | None = None
) -> SpawnScan:
    """The stdlib, read for every process-starting name the declared surface knows about."""
    tree = root if root is not None else Path(sysconfig.get_paths()["stdlib"])
    alphabet = primitives if primitives is not None else spawn_primitives()
    return spawn_scan(tree, alphabet, _attribute_triggers(alphabet))


def _attribute_triggers(alphabet: Mapping[str, frozenset[str]]) -> tuple[bytes, ...]:
    return tuple(sorted({name.encode() for names in alphabet.values() for name in names}))


def module_triggers(alphabet: Mapping[str, frozenset[str]]) -> tuple[bytes, ...]:
    """The cheaper prefilter for an alphabet of whole modules -- what the installed scan passes."""
    return tuple(sorted(module.encode() for module in alphabet))


def is_excluded(relative: str) -> bool:
    """CPython's own tests and any third-party tree beside the stdlib, by directory segment.

    Public because what the scan SKIPPED is part of every statement made about what it read:
    `gpu_guard_spawn_reach_coverage` measures the same tree one level down and has to skip the same
    directories, and a second copy of this rule is a second thing to keep in step.
    """
    return bool(_EXCLUDED_SEGMENTS & set(relative.split("/")[:-1]))


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None
