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

The INSTALLED packages are read the same way and for a narrower question. A dependency calling
`subprocess.Popen` says nothing the declaration does not already say, and looking for it means
parsing the whole tree, so `installed_spawn_reaches` uses the `below_the_seams` alphabet: only the
starts that go past every patchable name. Those are declared per package rather than per file
(`DECLARED_PACKAGE_REACHERS`), because a release moves its modules and the decision is about the
dependency. Package granularity is the right unit for surviving a release bump and the wrong unit
for a residual -- it excuses every module in the package, so a future `joblib` that starts children a
second way, from a file the declaration never saw, would be excused by a line written about `loky`.
What narrows it without per-file churn is the MEASUREMENT each declaration carries
(`PackageReacher`): the primitives and the file count it was written against, so a package whose
reach grows past them is reported by `gpu_guard_spawn_reach_audit.outgrown_reachers` rather than
silently covered.

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
arriving is what this refuses.
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
# reaches past every name the denial can patch, which is what makes them worth scanning for.
_LOW_LEVEL_STARTS: Mapping[str, frozenset[str]] = {
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
            "`_winapi.CreateProcess` -- a private copy of the same `spawn` / `forkserver` residual "
            "`multiprocessing/util.py` carries",
        ),
        primitives=_VENDORED_MULTIPROCESSING_STARTS,
        files=3,
    ),
    "multiprocess": PackageReacher(
        SpawnCoverage(
            COVERAGE_RESIDUAL,
            reason="a `dill`-based fork of `multiprocessing`, so it carries that module's residual "
            "verbatim: `util.spawnv_passfds` -> `_posixsubprocess.fork_exec`, plus the "
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


@dataclass(frozen=True)
class ModuleReach:
    """One module and the process-starting names its source resolves."""

    path: str
    primitives: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.path} -> {', '.join(self.primitives)}"


@dataclass(frozen=True)
class SpawnScan:
    """One pass over a tree: what it read, and what it found starting children.

    What it READ is the half that keeps the result honest. A tree whose source is absent -- a frozen
    or `.pyc`-only install, a path that is not there at all -- yields no reaches, which is the same
    answer as a tree where nothing starts a child. `files_read` tells those two apart at the
    degenerate end; `modules_read` -- the top-level names the pass parsed at least one file of --
    is what `gpu_guard_spawn_reach_coverage` weighs against `sys.stdlib_module_names` to tell them
    apart in the middle, where a host ships half its library as source.
    """

    root: str
    files_read: int
    modules_read: tuple[str, ...]
    reaches: tuple[ModuleReach, ...]


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
        **_LOW_LEVEL_STARTS,
    }


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
        **_LOW_LEVEL_STARTS,
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


def installed_spawn_reaches(
    root: Path | None = None, primitives: Mapping[str, frozenset[str]] | None = None
) -> SpawnScan:
    """The installed packages, read for the starts that go BELOW every name the denial patches."""
    tree = root if root is not None else Path(sysconfig.get_paths()["purelib"])
    alphabet = primitives if primitives is not None else below_the_seams()
    return spawn_scan(tree, alphabet, _module_triggers(alphabet))


def _attribute_triggers(alphabet: Mapping[str, frozenset[str]]) -> tuple[bytes, ...]:
    return tuple(sorted({name.encode() for names in alphabet.values() for name in names}))


def _module_triggers(alphabet: Mapping[str, frozenset[str]]) -> tuple[bytes, ...]:
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
