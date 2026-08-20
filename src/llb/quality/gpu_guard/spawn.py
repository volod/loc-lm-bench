"""Every way an unmarked test can start a child, closed against the same device denial.

`llb.quality.gpu_guard.guard` OBSERVES what the pytest process itself did to the device. A child leaves
nothing here to read, so a child is denied the device instead: for the duration of an unmarked test
it starts with an empty `CUDA_VISIBLE_DEVICES` and cannot open a CUDA context. This module owns that
denial -- the environment rewrite and the set of spawn entry points it is installed at.

Per-seam rather than per-process: the alternative -- a pytest that re-execs itself once with an
empty `CUDA_VISIBLE_DEVICES` -- has no per-test granularity, and `make test` runs `slow` and
unmarked tests in ONE process while `gpu_env` is exempt from `slow`, so both tiers share a process
under `make ci` too. That evidence is written down in
`docs/impl/current/host-validation.md#code-quality-checks`.

**The seams.** `spawn_seams()` names each entry point, the callable the denial goes THROUGH, and the
replacement. Four shapes cover them:

- an entry point that takes an environment (`subprocess.Popen`, `os.execve`, `os.execvpe`,
  `os.posix_spawn`, `os.posix_spawnp`) has that argument rewritten, whatever the caller passed;
- an entry point that takes NO environment reaches its `*e` sibling with a denied one
  (`os.execv` -> `os.execve`, `os.execvp` -> `os.execvpe`), or -- for `os.system`, which has no
  sibling -- carries the denial as an exported assignment in front of the command;
- a FORK is a copy of this process, so `os.fork` / `os.forkpty` apply the denial inside the CHILD,
  where poisoning the variable costs nothing because the child is not the session.
- POSIX `multiprocessing` routes its public `util.spawnv_passfds` helper through the patched
  `os.posix_spawn`. Dense temporary copies preserve the descriptors it was asked to pass while
  individual close actions plus `POSIX_SPAWN_CLOSEFROM` close every other descriptor without an
  open-fd race. A persistent forkserver is stopped on both sides of the context so its inherited
  environment stays per-test.

Patching those covers more than it names, because the rest of the `os` spawn surface is written in
Python on top of them and resolves the names as module globals at call time: `execl` / `execlp` go
through `execv` / `execvp`, `execle` / `execlpe` through `execve` / `execvpe`, and the whole
`spawnv*` / `spawnl*` family through `_spawnvef`, which forks and then calls `execv` / `execve`.
`multiprocessing` with `fork` reaches `os.fork` directly; `spawn` and `forkserver` reach the
`spawnv_passfds` seam.

Both halves of that -- the delegation above and the residual list below -- were written against one
CPython, so neither is left as prose: `llb.quality.gpu_guard.surface` enumerates the
process-starting names the RUNNING interpreter exposes and declares each a seam, a delegation
(checked against the callable's code object), or a residual, and `gpu_guard_spawn_surface_audit`
refuses one that is none of the three -- so a Python that grows a spawn function, rewrites a
delegation in C, or defaults `multiprocessing` to a residual start method fails a test instead.

Residual -- what a child can still do with the device:

- on a POSIX Python that does not expose `POSIX_SPAWN_CLOSEFROM` (including Python 3.12), explicit
  `spawn` / `forkserver` remains uncovered: the public API cannot reproduce close-all-except-passfds
  without racing another thread, so the seam is not installed.
- a caller reaching `_posixsubprocess.fork_exec` itself instead of the public
  `multiprocessing.util.spawnv_passfds` helper.
- a native extension that calls `fork(2)` / `execve(2)` / `system(3)` in C without coming back
  through the `os` module.
- a child that sets `CUDA_VISIBLE_DEVICES` back itself -- the denial is a default, not a sandbox.
- the `os.system` denial is a POSIX-shell mechanism (`export VAR=''`); it means nothing to `cmd.exe`.
- NVML, as before: `nvidia-smi` still lists the GPU under an empty `CUDA_VISIBLE_DEVICES`, so a
  child that only ASKS whether hardware exists still gets yes -- it just cannot open a context.
"""

import inspect
import multiprocessing.util
import os
import subprocess
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from llb.quality.gpu_guard.spawn_multiprocessing import (
    routing_spawnv_passfds,
    stop_forkserver,
    supports_descriptor_closure,
)

# What a child of an unmarked test inherits. An empty value is how the CUDA runtime is told there is
# no device; flashinfer needs no switch of its own, since every probe here asks torch first.
CHILD_DENIAL = {"CUDA_VISIBLE_DEVICES": ""}
# `Popen(args, bufsize, executable, stdin, stdout, stderr, preexec_fn, close_fds, shell, cwd, env)`
_POPEN_ENV_POSITION = 10
# `os.system` takes no environment, so the denial rides the command text. Newline-terminated rather
# than `;`-joined so a command that opens with a comment or a shell keyword still parses, and
# `export` rather than a bare assignment so it reaches every command in the script, not just the
# first one.
_SYSTEM_DENIAL = "".join(f"export {name}='{value}'\n" for name, value in CHILD_DENIAL.items())


@contextmanager
def denied_children() -> Iterator[None]:
    """Hand every child an empty `CUDA_VISIBLE_DEVICES` for the duration, however it is started.

    Patching the seams rather than this process's environment is the whole point: the parent keeps
    its device (see `llb.quality.gpu_guard.guard`, where the caching that forces that is written down).
    """
    stop_forkserver()
    seams = spawn_seams()
    for seam in seams:
        seam.apply()
    try:
        yield
    finally:
        try:
            stop_forkserver()
        finally:
            for seam in reversed(seams):
                seam.restore()


def denied_child_env(inherited: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a child gets: what it would have inherited, with the device taken out."""
    base = dict(os.environ if inherited is None else inherited)
    base.update(CHILD_DENIAL)
    return base


@dataclass(frozen=True)
class SpawnSeam:
    """One patched spawn entry point: where it lives, what it was, and what stands in for it."""

    module: ModuleType
    attribute: str
    original: Any
    replacement: Any

    @property
    def label(self) -> str:
        """`os.execv` -- what an operator reads when the seam is named."""
        return f"{self.module.__name__}.{self.attribute}"

    def apply(self) -> None:
        setattr(self.module, self.attribute, self.replacement)

    def restore(self) -> None:
        setattr(self.module, self.attribute, self.original)


def denying_popen(original: type[Any]) -> type[Any]:
    """A `subprocess.Popen` that starts every child with no visible CUDA device.

    Substituted for the module attribute, so `subprocess.run` / `call` / `check_output` -- which all
    reach for `subprocess.Popen` at call time -- are covered by the one patch.
    """

    class _DeviceDeniedPopen(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if len(args) > _POPEN_ENV_POSITION:
                mutable = list(args)
                mutable[_POPEN_ENV_POSITION] = denied_child_env(mutable[_POPEN_ENV_POSITION])
                args = tuple(mutable)
            else:
                kwargs["env"] = denied_child_env(kwargs.get("env"))
            super().__init__(*args, **kwargs)

    return _DeviceDeniedPopen


def denying_system(original: Callable[..., int]) -> Callable[..., int]:
    """`os.system` with the denial exported in front of the command it was handed."""

    def _denied_system(command: Any) -> int:
        if isinstance(command, bytes):
            return original(_SYSTEM_DENIAL.encode() + command)
        return original(_SYSTEM_DENIAL + os.fsdecode(command))

    return _denied_system


def denying_exec_with_env(original: Callable[..., Any]) -> Callable[..., Any]:
    """`os.execve` / `os.execvpe`: the caller's environment, minus the device.

    Bound against the real signature rather than a restated one, because the two disagree about
    their parameter NAMES -- `(path, argv, env)` against `(file, args, env)` -- and both accept
    keywords. Reading the signature once, at patch time, is what lets one wrapper serve both and a
    keyword call reach the rewrite instead of slipping past it.
    """
    signature = inspect.signature(original)

    def _denied_exec(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.arguments["env"] = denied_child_env(bound.arguments["env"])
        return original(*bound.args, **bound.kwargs)

    return _denied_exec


def denying_exec_without_env(sibling: Callable[..., Any]) -> Callable[..., Any]:
    """`os.execv` / `os.execvp`, routed through the `*e` sibling that can carry the denial.

    A pair shares its parameter names (`execv`/`execve` take `path, argv`; `execvp`/`execvpe` take
    `file, args`), so the caller's arguments bind straight onto the sibling with the denied
    environment supplied as the one it does not pass.
    """
    signature = inspect.signature(sibling)

    def _denied_exec(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs, env=denied_child_env())
        return sibling(*bound.args, **bound.kwargs)

    return _denied_exec


def denying_posix_spawn(original: Callable[..., int]) -> Callable[..., int]:
    """`os.posix_spawn` / `os.posix_spawnp`, whose environment is a required third positional."""

    def _denied_posix_spawn(path: Any, argv: Any, env: Mapping[str, str], /, **kwargs: Any) -> int:
        return original(path, argv, denied_child_env(env), **kwargs)

    return _denied_posix_spawn


def denying_fork(original: Callable[[], int]) -> Callable[[], int]:
    """`os.fork`, denied in the CHILD -- there is no environment argument to rewrite.

    Poisoning the variable is the right move on this side of the fork: the child is not the pytest
    session, so nothing downstream depends on torch un-caching its answer.
    """

    def _denied_fork() -> int:
        pid = original()
        if pid == 0:
            os.environ.update(CHILD_DENIAL)
        return pid

    return _denied_fork


def denying_forkpty(original: Callable[[], tuple[int, int]]) -> Callable[[], tuple[int, int]]:
    """`os.forkpty`, the same denial over its `(pid, fd)` return."""

    def _denied_forkpty() -> tuple[int, int]:
        pid, fd = original()
        if pid == 0:
            os.environ.update(CHILD_DENIAL)
        return pid, fd

    return _denied_forkpty


def spawn_seams() -> tuple[SpawnSeam, ...]:
    """Every spawn entry point the denial covers, built against the modules as they are NOW.

    Read before anything is patched, so each replacement wraps the real entry point rather than
    another replacement, and re-reading after a restore rebuilds the same set.
    """
    candidates = (
        _seam(subprocess, "Popen", denying_popen),
        _seam(os, "system", denying_system),
        _seam(os, "execve", denying_exec_with_env),
        _seam(os, "execvpe", denying_exec_with_env),
        _seam(os, "execv", denying_exec_without_env, through="execve"),
        _seam(os, "execvp", denying_exec_without_env, through="execvpe"),
        _seam(os, "posix_spawn", denying_posix_spawn),
        _seam(os, "posix_spawnp", denying_posix_spawn),
        _seam(os, "fork", denying_fork),
        _seam(os, "forkpty", denying_forkpty),
        _multiprocessing_spawn_seam(),
    )
    return tuple(seam for seam in candidates if seam is not None)


def _multiprocessing_spawn_seam() -> SpawnSeam | None:
    """The POSIX helper below every `multiprocessing` spawn and forkserver launch."""
    original = getattr(multiprocessing.util, "spawnv_passfds", None)
    if original is None or not supports_descriptor_closure():
        return None
    return SpawnSeam(
        module=multiprocessing.util,
        attribute="spawnv_passfds",
        original=original,
        replacement=routing_spawnv_passfds(original),
    )


def _seam(
    module: ModuleType,
    attribute: str,
    build: Callable[[Any], Any],
    *,
    through: str | None = None,
) -> SpawnSeam | None:
    """`module.attribute` replaced by a denial built over `module.through` (itself by default).

    A missing attribute yields no seam rather than an error: `fork`, `forkpty`, `posix_spawnp`, and
    the `*e` exec siblings are POSIX-only, and the guard must not be what breaks a non-POSIX host.
    """
    original = getattr(module, attribute, None)
    source = original if through is None else getattr(module, through, None)
    if original is None or source is None:
        return None
    return SpawnSeam(
        module=module, attribute=attribute, original=original, replacement=build(source)
    )
