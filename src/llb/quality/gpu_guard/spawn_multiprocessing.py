"""The POSIX `multiprocessing` seam that lives below `os` and `subprocess`.

`multiprocessing.util.spawnv_passfds` normally calls the private
`_posixsubprocess.fork_exec` directly, without an environment argument. Replacing that helper with
`routing_spawnv_passfds` sends it through the public `os.posix_spawn` seam instead. Its file actions
copy requested descriptors into a dense temporary prefix, close every other prefix slot, close
everything above it, restore the requested descriptor numbers, and remove the temporaries. That
preserves the data pipes and resource-tracker descriptor that are the reason `spawnv_passfds`
exists, while retaining the original helper's `close_fds=True` behavior without enumerating open
descriptors in the parent and racing another thread that opens one.

The exception-aware close requires `POSIX_SPAWN_CLOSEFROM`, exposed by Python 3.13 on this host but
not Python 3.12. Without it, the public API offers only one-fd close actions: enumerating the open
set races another thread, while emitting one action for every possible descriptor is not a viable
per-child operation. `supports_descriptor_closure` therefore keeps the seam an explicit residual
on such an interpreter rather than silently leaking a descriptor or patching the private C call.

A forkserver keeps the environment it had when it started and forks every later child from that
long-lived process. `stop_forkserver` brackets each denied test so an older visible-device server
cannot bypass the denial and a denied server cannot leak that denial into a later `gpu_env` test.
The private `_stop` hook is explicitly provided by CPython for tests; refusing an interpreter that
offers `forkserver` without it is safer than silently making the per-test claim false.
"""

import multiprocessing
import os
from collections.abc import Callable, Iterable
from types import ModuleType
from typing import Any

_FIRST_NONSTANDARD_FD = 3


def supports_descriptor_closure(os_module: Any = os) -> bool:
    """Whether this host's public spawn API can close all descriptors above a boundary."""
    return callable(getattr(os_module, "posix_spawn", None)) and all(
        hasattr(os_module, attribute)
        for attribute in (
            "POSIX_SPAWN_CLOSE",
            "POSIX_SPAWN_CLOSEFROM",
            "POSIX_SPAWN_DUP2",
        )
    )


def descriptor_file_actions(
    passfds: Iterable[int], os_module: Any = os
) -> tuple[tuple[int, ...], ...]:
    """Close every nonstandard descriptor except `passfds`, without reading the parent's FD set.

    Temporary descriptors are chosen from the first numbers not present in `passfds`, so no copy
    overwrites a later source. The dense prefix plus `CLOSEFROM` covers every possible descriptor
    number, including one another thread opens after these actions are built.
    """
    passed = tuple(sorted(set(map(int, passfds))))
    passed_set = set(passed)
    temporary: list[int] = []
    candidate = _FIRST_NONSTANDARD_FD
    while len(temporary) < len(passed):
        if candidate not in passed_set:
            temporary.append(candidate)
        candidate += 1

    duplicate = os_module.POSIX_SPAWN_DUP2
    close = os_module.POSIX_SPAWN_CLOSE
    closefrom = os_module.POSIX_SPAWN_CLOSEFROM
    temporary_set = set(temporary)
    actions: list[tuple[int, ...]] = [
        (duplicate, source, target) for source, target in zip(passed, temporary, strict=True)
    ]
    actions.extend(
        (close, fd) for fd in range(_FIRST_NONSTANDARD_FD, candidate) if fd not in temporary_set
    )
    actions.append((closefrom, candidate))
    actions.extend(
        (duplicate, temporary_fd, passed_fd)
        for temporary_fd, passed_fd in zip(temporary, passed, strict=True)
    )
    actions.extend((close, temporary_fd) for temporary_fd in temporary)
    return tuple(actions)


def routing_spawnv_passfds(
    _original: Callable[..., int], os_module: ModuleType = os
) -> Callable[[Any, Iterable[Any], Iterable[int]], int]:
    """Route `spawnv_passfds` through the currently installed `os.posix_spawn` seam."""

    def _spawnv_passfds(path: Any, args: Iterable[Any], passfds: Iterable[int]) -> int:
        file_actions = descriptor_file_actions(passfds, os_module)
        return int(
            os_module.posix_spawn(
                path,
                args,
                dict(os_module.environ),
                file_actions=file_actions,
            )
        )

    return _spawnv_passfds


def stop_forkserver(context: Any = multiprocessing) -> None:
    """Stop the current process's persistent forkserver, when this host offers one."""
    if "forkserver" not in context.get_all_start_methods():
        return

    from multiprocessing import forkserver

    stop = getattr(getattr(forkserver, "_forkserver", None), "_stop", None)
    if not callable(stop):
        raise RuntimeError(
            "multiprocessing offers forkserver without its test lifecycle hook; "
            "the child device denial cannot keep forkserver state per-test"
        )
    stop()
