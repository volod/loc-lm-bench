"""The denial's coverage claim, restated as something the running interpreter can refute.

`llb.quality.gpu_guard_spawn` makes two claims about the surface it covers, and both were written
against one CPython. The ten seams reach the whole `os` spawn family only because `os.py` builds
`execl*` / `spawn*` in Python on top of the four `exec*v*` names; the residual list in that module's
docstring is accurate only because `multiprocessing` defaults to `fork` on Linux, which the fork seam
covers. A Python upgrade can move either -- 3.14 changes that default -- and nothing fails on its
own, because a name the seam set never heard of is indistinguishable from one it deliberately
excluded.

This module is the other half of that: the process-starting surface ENUMERATED from the interpreter
that is running, plus a declaration per name saying how the denial reaches it. Four states make up
the vocabulary:

- `COVERAGE_SEAM` -- patched by `spawn_seams()` itself.
- `COVERAGE_THROUGH` -- reached because the name's own implementation calls another declared name at
  call time, so patching that one covers this one. The claim is checked rather than believed:
  `delegation_is_live` reads the callable's code object and asks whether the target is still a name
  it resolves. A C entry point resolves nothing this process can patch, so it fails the check --
  which is the point, since `os.spawnv` is Python on POSIX and native on Windows.
- `COVERAGE_RESIDUAL` -- reached by nothing, with the reason stated. This is the declared-residual
  vocabulary: a residual is a decision on the record, not an omission.
- `COVERAGE_NOT_A_SPAWN` -- enumerated by the rule below but not an entry point at all (a result
  record), with the reason stated.

The enumeration is deliberately a rule and not a list: every `os` name in the exec / fork / spawn /
posix_spawn families plus `system` / `popen` / `startfile`, and every public non-exception callable
of `subprocess`. A name the rule catches and no declaration names is what
`llb.quality.gpu_guard_spawn_surface_audit` refuses.

Out of enumeration on purpose: `_posixsubprocess.fork_exec`, a private C signature that changes
between versions and is reachable from ordinary Python only through `multiprocessing`'s
`spawn` / `forkserver` start methods -- which are declared residuals HERE, keyed by start method, so
an interpreter whose default moves onto one of them is refused rather than silently uncovered.
"""

import multiprocessing
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from types import ModuleType
from typing import Any

from llb.quality.gpu_guard_spawn import spawn_seams

COVERAGE_SEAM = "seam"
COVERAGE_THROUGH = "through"
COVERAGE_RESIDUAL = "residual"
COVERAGE_NOT_A_SPAWN = "not-a-spawn"

# `os` groups its process-starting entry points into families by name, so the enumeration can be a
# rule: a public callable whose name opens with one of these prefixes, or is one of the loners.
_OS_SPAWN_PREFIXES = ("exec", "fork", "posix_spawn", "spawn")
_OS_SPAWN_NAMES = ("popen", "startfile", "system")
# The modules the declarations name. Also what a delegation is read out of.
_SURFACE_MODULES: Mapping[str, ModuleType] = {"os": os, "subprocess": subprocess}
_DEFAULT_START_METHOD_SCRIPT = "import multiprocessing; print(multiprocessing.get_start_method())"


@dataclass(frozen=True)
class SpawnCoverage:
    """How the denial reaches one process-starting name -- or why it does not."""

    kind: str
    through: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.kind == COVERAGE_THROUGH and not self.through:
            raise ValueError("a delegation must name the entry point it is covered through")
        if self.kind in (COVERAGE_RESIDUAL, COVERAGE_NOT_A_SPAWN) and not self.reason:
            raise ValueError(f"a {self.kind} declaration must say why")


def _seam() -> SpawnCoverage:
    return SpawnCoverage(COVERAGE_SEAM)


def _through(target: str) -> SpawnCoverage:
    return SpawnCoverage(COVERAGE_THROUGH, through=target)


def _residual(reason: str) -> SpawnCoverage:
    return SpawnCoverage(COVERAGE_RESIDUAL, reason=reason)


def _not_a_spawn(reason: str) -> SpawnCoverage:
    return SpawnCoverage(COVERAGE_NOT_A_SPAWN, reason=reason)


DECLARED_SPAWN_SURFACE: Mapping[str, SpawnCoverage] = {
    # Patched by `spawn_seams()`.
    "subprocess.Popen": _seam(),
    "os.system": _seam(),
    "os.execve": _seam(),
    "os.execvpe": _seam(),
    "os.execv": _seam(),
    "os.execvp": _seam(),
    "os.posix_spawn": _seam(),
    "os.posix_spawnp": _seam(),
    "os.fork": _seam(),
    "os.forkpty": _seam(),
    # Written in `os.py` on top of the four `exec*v*` seams, which they resolve as module globals.
    "os.execl": _through("os.execv"),
    "os.execle": _through("os.execve"),
    "os.execlp": _through("os.execvp"),
    "os.execlpe": _through("os.execvpe"),
    # `_spawnvef` forks and then calls the exec function it was handed -- the global one, at call
    # time. Two seams cover each of these; the exec name is the one that carries the environment.
    "os.spawnv": _through("os.execv"),
    "os.spawnve": _through("os.execve"),
    "os.spawnvp": _through("os.execvp"),
    "os.spawnvpe": _through("os.execvpe"),
    "os.spawnl": _through("os.spawnv"),
    "os.spawnle": _through("os.spawnve"),
    "os.spawnlp": _through("os.spawnvp"),
    "os.spawnlpe": _through("os.spawnvpe"),
    # `os.popen` imports `subprocess` inside its body and reads `Popen` off it at call time.
    "os.popen": _through("subprocess.Popen"),
    "os.startfile": _residual(
        "Windows only: it hands the path to the shell association rather than starting a child "
        "this process describes, and the `os.system` mechanism the denial rides is POSIX-shell "
        "specific anyway"
    ),
    # `subprocess`'s convenience functions all reach `Popen` as a module global, directly or one
    # hop away, which is why the class is the only seam the module needs.
    "subprocess.run": _through("subprocess.Popen"),
    "subprocess.call": _through("subprocess.Popen"),
    "subprocess.check_call": _through("subprocess.call"),
    "subprocess.check_output": _through("subprocess.run"),
    "subprocess.getstatusoutput": _through("subprocess.check_output"),
    "subprocess.getoutput": _through("subprocess.getstatusoutput"),
    "subprocess.CompletedProcess": _not_a_spawn("the record `run` returns, not an entry point"),
    "subprocess.STARTUPINFO": _not_a_spawn(
        "Windows only: the creation-flags record a caller hands to `Popen`, not an entry point"
    ),
}

DECLARED_START_METHODS: Mapping[str, SpawnCoverage] = {
    # A `fork` child is a copy of this process, so the fork seam denies it from the inside; proved
    # end to end by the `multiprocessing(fork)` case in `test_gpu_guard_spawn_children.py`.
    "fork": _through("os.fork"),
    "spawn": _residual(
        "`multiprocessing.util.spawnv_passfds` calls `_posixsubprocess.fork_exec` with no "
        "environment list, which is neither `subprocess.Popen` nor an `os` entry point"
    ),
    "forkserver": _residual(
        "the server is started the same way as a `spawn` child, and every later child is forked "
        "from IT rather than from this process"
    ),
}


def interpreter_spawn_names(
    os_module: Any = os, subprocess_module: Any = subprocess
) -> tuple[str, ...]:
    """Every process-starting name the running interpreter exposes, by the rule above."""
    return tuple(sorted(_os_spawn_names(os_module) + _subprocess_spawn_names(subprocess_module)))


def delegation_is_live(
    name: str, target: str, modules: Mapping[str, Any] = _SURFACE_MODULES
) -> bool:
    """Whether `name`'s own implementation still resolves `target` when it is called.

    Read off the code object rather than trusted: a name is resolved at call time only if the
    callable is written in Python and still names the target, so a family CPython rewrites in C --
    or points somewhere else -- stops being covered here instead of quietly.
    """
    code = getattr(_lookup(name, modules), "__code__", None)
    if code is None:
        return False
    target_module, _, target_attribute = target.rpartition(".")
    if target_attribute not in code.co_names:
        return False
    return target_module == name.partition(".")[0] or target_module in code.co_names


@cache
def _child_default_start_method(executable: str = sys.executable) -> str:
    """Resolve the default in a disposable interpreter, once per executable and parent process."""
    result = subprocess.run(
        [executable, "-c", _DEFAULT_START_METHOD_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def default_start_method(
    context: Any = multiprocessing,
    child_default: Callable[[], str] = _child_default_start_method,
) -> str:
    """The start method a bare `multiprocessing.Process()` would use, read without fixing it.

    `get_start_method()` RESOLVES the default context as a side effect, after which a
    `set_start_method` call raises, so a check must not be what pins the suite's start method. An
    already-set method is exact. Otherwise a disposable child resolves its own default, and the
    documented default-first ordering of `get_all_start_methods()` is checked against that answer
    rather than trusted. The child read is cached because an interpreter's default does not move.
    """
    chosen = context.get_start_method(allow_none=True)
    if chosen is not None:
        return str(chosen)
    ordered_first = str(context.get_all_start_methods()[0])
    child_chosen = child_default()
    if child_chosen != ordered_first:
        raise RuntimeError(
            "multiprocessing default start method disagrees with get_all_start_methods(): "
            f"child interpreter reported {child_chosen!r}, ordering reports {ordered_first!r} first"
        )
    return child_chosen


@dataclass(frozen=True)
class ObservedSurface:
    """What THIS interpreter exposes, patches, and defaults to -- the facts the audit reads."""

    names: tuple[str, ...]
    seams: tuple[str, ...]
    start_methods: tuple[str, ...]
    default_start_method: str
    modules: Mapping[str, Any]

    @classmethod
    def read(cls) -> "ObservedSurface":
        return cls(
            names=interpreter_spawn_names(),
            seams=tuple(seam.label for seam in spawn_seams()),
            start_methods=tuple(multiprocessing.get_all_start_methods()),
            default_start_method=default_start_method(),
            modules=_SURFACE_MODULES,
        )

    def delegates(self, name: str, target: str) -> bool:
        """Whether the declared delegation from `name` to `target` is one this interpreter makes."""
        return delegation_is_live(name, target, self.modules)


def _os_spawn_names(module: Any) -> tuple[str, ...]:
    return tuple(
        f"os.{attribute}"
        for attribute in dir(module)
        if (attribute.startswith(_OS_SPAWN_PREFIXES) or attribute in _OS_SPAWN_NAMES)
        and not attribute.startswith("_")
        and callable(getattr(module, attribute, None))
    )


def _subprocess_spawn_names(module: Any) -> tuple[str, ...]:
    """`subprocess.__all__` minus its exceptions and constants -- every public callable it ships.

    Wider than the one seam on purpose: a module whose public callables are enumerated grows a new
    one as an undeclared name, where a hand-written list would grow it silently.
    """
    return tuple(
        f"subprocess.{attribute}"
        for attribute in getattr(module, "__all__", ())
        if callable(getattr(module, attribute, None)) and not _is_exception(module, attribute)
    )


def _is_exception(module: Any, attribute: str) -> bool:
    value = getattr(module, attribute, None)
    return isinstance(value, type) and issubclass(value, BaseException)


def _lookup(name: str, modules: Mapping[str, Any]) -> Any:
    module_name, _, attribute = name.partition(".")
    return getattr(modules.get(module_name), attribute, None)
