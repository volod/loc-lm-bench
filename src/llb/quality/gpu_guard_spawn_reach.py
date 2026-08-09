"""Which stdlib module starts a child, MEASURED -- so the enumerated surface is not two modules on
faith.

`llb.quality.gpu_guard_spawn_surface` enumerates the process-starting names of `os` and
`subprocess`, and every other way a test reaches a child is covered because the helper it calls
resolves a name in one of those two. That last sentence was the one claim the name check left
standing: `pty.spawn` forks and execs, `asyncio`'s unix transport starts a `Popen`,
`multiprocessing.util.spawnv_passfds` does neither, and nothing said which of those is which.

So this module reads the stdlib instead of asserting about it. Every `*.py` under the stdlib root is
parsed and its process-starting CALL SITES are resolved through the module's own imports (`os.fork`,
`from subprocess import Popen`, `import os as operating`), against an alphabet that is the declared
surface plus the C modules underneath it -- `posix` / `nt` (what `os` re-exports), and
`_posixsubprocess` / `_winapi` (what `subprocess` and `multiprocessing` call below any patchable
name). A module reaching a DECLARED name needs nothing: the declaration already carries that
decision, covered or residual. A module reaching something the declared surface does not name is
excused by `DECLARED_REACHERS` here, or refused by `llb.quality.gpu_guard_spawn_reach_audit`.

Two trees are left out, both stated rather than assumed: CPython's own regression suite (`test/`,
`*/tests/`, `idlelib/idle_test`), a corpus that starts children on purpose, is not runtime code any
llb path imports, and costs 4s and an extra declaration to include; and `site-packages`, which is
third-party rather than stdlib and is a different axis entirely (torch, uv, and vLLM all start
children). Both are residuals of this check, not blind spots of it.

The measurement itself is the interesting half: on CPython 3.13 the scan finds 25 stdlib modules
that start a child, of which 23 resolve a name the denial covers. The exceptions are exactly the
residuals already on the record -- `multiprocessing/util.py` and `multiprocessing/popen_spawn_win32
.py` -- plus `subprocess.py`, whose low-level starts sit BEHIND the `Popen` seam. Two modules is the
right enumerated surface, and now it is a result rather than a claim.
"""

import ast
import sysconfig
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class ModuleReach:
    """One stdlib module and the process-starting names its source resolves."""

    path: str
    primitives: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.path} -> {', '.join(self.primitives)}"


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


def stdlib_spawn_reaches(
    root: Path | None = None, primitives: Mapping[str, frozenset[str]] | None = None
) -> tuple[ModuleReach, ...]:
    """Every stdlib module whose source starts a child, and the names it starts it through."""
    tree = root if root is not None else Path(sysconfig.get_paths()["stdlib"])
    alphabet = primitives if primitives is not None else spawn_primitives()
    triggers = tuple(sorted({name.encode() for names in alphabet.values() for name in names}))
    found: list[ModuleReach] = []
    for path in sorted(tree.rglob("*.py")):
        relative = path.relative_to(tree).as_posix()
        source = None if _is_excluded(relative) else _read(path)
        if source is None or not any(trigger in source for trigger in triggers):
            continue
        reached = _source_reaches(source, alphabet)
        if reached:
            found.append(ModuleReach(path=relative, primitives=reached))
    return tuple(found)


def _is_excluded(relative: str) -> bool:
    """CPython's own tests and any third-party tree beside the stdlib, by directory segment."""
    return bool(_EXCLUDED_SEGMENTS & set(relative.split("/")[:-1]))


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _source_reaches(source: bytes, alphabet: Mapping[str, frozenset[str]]) -> tuple[str, ...]:
    """The process-starting names one module's source resolves, through its own imports."""
    try:
        parsed = ast.parse(source)
    except (SyntaxError, ValueError):
        return ()
    modules, names, calls = _imports_and_calls(parsed, alphabet)
    reached = {_call_label(call, modules, names, alphabet) for call in calls}
    return tuple(sorted(label for label in reached if label is not None))


def _imports_and_calls(
    parsed: ast.AST, alphabet: Mapping[str, frozenset[str]]
) -> tuple[dict[str, str], dict[str, str], list[ast.expr]]:
    """One walk: local name -> module, local name -> label, and every call target in the module."""
    modules: dict[str, str] = {}
    names: dict[str, str] = {}
    calls: list[ast.expr] = []
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            calls.append(node.func)
        elif isinstance(node, ast.Import):
            modules.update(_module_aliases(node, alphabet))
        elif isinstance(node, ast.ImportFrom):
            names.update(_name_aliases(node, alphabet))
    return modules, names, calls


def _module_aliases(node: ast.Import, alphabet: Mapping[str, frozenset[str]]) -> dict[str, str]:
    """`import os as operating` -> `{"operating": "os"}`, for the modules the alphabet names."""
    return {
        (alias.asname or alias.name): alias.name for alias in node.names if alias.name in alphabet
    }


def _name_aliases(node: ast.ImportFrom, alphabet: Mapping[str, frozenset[str]]) -> dict[str, str]:
    """`from subprocess import Popen as Runner` -> `{"Runner": "subprocess.Popen"}`."""
    module = node.module
    if module is None or module not in alphabet:
        return {}
    return {
        (alias.asname or alias.name): f"{module}.{alias.name}"
        for alias in node.names
        if alias.name in alphabet[module]
    }


def _call_label(
    call: ast.expr,
    modules: Mapping[str, str],
    names: Mapping[str, str],
    alphabet: Mapping[str, frozenset[str]],
) -> str | None:
    """`os.fork(...)` / `fork(...)` resolved back to the declared name it calls, or None."""
    if isinstance(call, ast.Attribute) and isinstance(call.value, ast.Name):
        module = modules.get(call.value.id)
        if module is not None and call.attr in alphabet[module]:
            return f"{module}.{call.attr}"
        return None
    if isinstance(call, ast.Name):
        return names.get(call.id)
    return None
