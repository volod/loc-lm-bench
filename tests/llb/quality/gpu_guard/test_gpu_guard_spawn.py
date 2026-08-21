"""The device denial has to hold at every spawn seam, not only at `subprocess.Popen`.

`llb.quality.gpu_guard.spawn` rewrites the environment a child starts with, at each entry point a
test could plausibly reach for. Three properties decide whether that is worth the seams: the rewrite
preserves whatever the caller passed apart from the device, each entry point's own argument shape is
handled (positional `env`, an exec sibling that takes none, a fork with no argument at all), and the
`os` families written on top of those names -- `execl*`, `spawn*`, `multiprocessing` under `fork` --
inherit the denial through them rather than needing seams of their own.

These cases all stand in front of a recorder, so they say what each seam PASSES.
`test_gpu_guard_spawn_children.py` is the other half: it starts a real child through each mechanism
and reads what that child actually saw.
"""

import os
from collections.abc import Callable
from types import SimpleNamespace

from llb.quality.gpu_guard import spawn as gpu_guard_spawn
from llb.quality.gpu_guard import spawn_multiprocessing as multiprocessing_spawn

_DEVICE = "CUDA_VISIBLE_DEVICES"


class _RecordingPopen:
    """Stands in for `subprocess.Popen` so the env rewrite is checkable without a process."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.seen_args = args
        self.seen_kwargs = kwargs


class _Recorder:
    """Stands in for an `os` spawn entry point, keeping what it was called with."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> int:
        self.calls.append((args, kwargs))
        return 0

    @property
    def only(self) -> tuple[tuple[object, ...], dict[str, object]]:
        assert len(self.calls) == 1
        return self.calls[0]


def _like_execve(recorder: _Recorder) -> Callable[..., int]:
    """A recorder wearing `os.execve`'s signature -- the wrappers bind against the real one."""

    def execve(path: object, argv: object, env: object) -> int:
        return recorder(path, argv, env)

    return execve


def _like_execvpe(recorder: _Recorder) -> Callable[..., int]:
    """`os.execvpe` names the same three parameters differently, which is why the wrappers bind."""

    def execvpe(file: object, args: object, env: object) -> int:
        return recorder(file, args, env)

    return execvpe


def test_a_child_inherits_the_callers_environment_minus_the_device():
    child = gpu_guard_spawn.denied_child_env({"PATH": "/bin", _DEVICE: "0"})
    assert child["PATH"] == "/bin"
    assert child[_DEVICE] == ""


def test_a_child_with_no_declared_environment_still_gets_one_with_the_device_removed():
    child = gpu_guard_spawn.denied_child_env(None)
    assert child[_DEVICE] == ""
    assert len(child) > 1


def test_the_denying_popen_rewrites_an_explicitly_passed_environment():
    denied = gpu_guard_spawn.denying_popen(_RecordingPopen)(["true"], env={"A": "1"})
    assert denied.seen_kwargs["env"]["A"] == "1"
    assert denied.seen_kwargs["env"][_DEVICE] == ""


def test_the_denying_popen_rewrites_a_positional_environment():
    """`env` is Popen's 11th positional; passing it that way must not slip past the rewrite."""
    positional = (["true"], -1, None, None, None, None, None, True, False, None, {"A": "1"})
    denied = gpu_guard_spawn.denying_popen(_RecordingPopen)(*positional)
    assert denied.seen_args[10][_DEVICE] == ""
    assert denied.seen_args[10]["A"] == "1"


def test_os_system_carries_the_denial_in_front_of_the_command():
    """`os.system` has no environment argument, so the denial has to ride the command text."""
    recorder = _Recorder()
    gpu_guard_spawn.denying_system(recorder)("echo hi")
    (command,), _ = recorder.only
    assert isinstance(command, str)
    assert command.endswith("\necho hi")
    assert f"export {_DEVICE}=''" in command


def test_os_system_keeps_a_bytes_command_a_bytes_command():
    recorder = _Recorder()
    gpu_guard_spawn.denying_system(recorder)(b"echo hi")
    (command,), _ = recorder.only
    assert isinstance(command, bytes)
    assert command.endswith(b"\necho hi")


def test_the_denial_prefix_survives_a_command_that_opens_with_a_comment():
    """Newline-joined rather than `;`-joined: a leading `#` would swallow a `;`-joined command."""
    recorder = _Recorder()
    gpu_guard_spawn.denying_system(recorder)("# nothing to do")
    (command,), _ = recorder.only
    assert command.splitlines()[-1] == "# nothing to do"


def test_an_exec_that_takes_an_environment_has_it_rewritten():
    recorder = _Recorder()
    denied = gpu_guard_spawn.denying_exec_with_env(_like_execve(recorder))
    denied("/bin/sh", ["sh", "-c", "true"], {"A": "1"})
    (path, args, env), _ = recorder.only
    assert path == "/bin/sh" and args == ["sh", "-c", "true"]
    assert env["A"] == "1" and env[_DEVICE] == ""


def test_an_exec_that_takes_no_environment_reaches_its_sibling_with_a_denied_one():
    """`os.execv` cannot carry a denial, so the seam routes it through `os.execve`."""
    recorder = _Recorder()
    gpu_guard_spawn.denying_exec_without_env(_like_execve(recorder))("/bin/sh", ["sh"])
    (path, args, env), _ = recorder.only
    assert path == "/bin/sh" and args == ["sh"]
    assert env[_DEVICE] == ""


# `os.execve` and `os.execvpe` both accept keywords and disagree about the NAMES: a wrapper that
# restated either signature would let the other's keyword call through with the device intact.
def test_an_exec_called_with_keywords_still_has_its_environment_rewritten():
    recorder = _Recorder()
    denied = gpu_guard_spawn.denying_exec_with_env(_like_execvpe(recorder))
    denied(file="sh", args=["sh"], env={"A": "1"})
    (_, _, env), _ = recorder.only
    assert env["A"] == "1" and env[_DEVICE] == ""


def test_the_sibling_route_binds_a_keyword_call_onto_the_denied_environment():
    recorder = _Recorder()
    gpu_guard_spawn.denying_exec_without_env(_like_execvpe(recorder))(file="sh", args=["sh"])
    (file, args, env), _ = recorder.only
    assert file == "sh" and args == ["sh"]
    assert env[_DEVICE] == ""


def test_posix_spawn_has_its_third_positional_rewritten_and_keeps_its_keywords():
    recorder = _Recorder()
    denied = gpu_guard_spawn.denying_posix_spawn(recorder)
    denied("/bin/sh", ["sh"], {"A": "1"}, file_actions=[])
    (path, argv, env), kwargs = recorder.only
    assert path == "/bin/sh" and argv == ["sh"]
    assert env["A"] == "1" and env[_DEVICE] == ""
    assert kwargs == {"file_actions": []}


def test_spawnv_passfds_routes_through_posix_spawn_and_clears_close_on_exec():
    """Dense temporary copies preserve passfds while every other descriptor number is closed."""
    recorder = _Recorder()
    fake_os = SimpleNamespace(
        POSIX_SPAWN_CLOSE=11,
        POSIX_SPAWN_CLOSEFROM=19,
        POSIX_SPAWN_DUP2=17,
        environ={"A": "1", _DEVICE: "0"},
        posix_spawn=recorder,
    )
    routed = multiprocessing_spawn.routing_spawnv_passfds(lambda: None, fake_os)
    assert routed("python", ["python", "-c", "pass"], [9, 4]) == 0
    (path, args, env), kwargs = recorder.only
    assert path == "python" and args == ["python", "-c", "pass"]
    assert env == {"A": "1", _DEVICE: "0"}
    assert kwargs == {
        "file_actions": (
            (17, 4, 3),
            (17, 9, 5),
            (11, 4),
            (19, 6),
            (17, 3, 4),
            (17, 5, 9),
            (11, 3),
            (11, 5),
        )
    }


def test_no_passfds_closes_every_nonstandard_descriptor_with_one_action():
    fake_os = SimpleNamespace(
        POSIX_SPAWN_CLOSE=11,
        POSIX_SPAWN_CLOSEFROM=19,
        POSIX_SPAWN_DUP2=17,
    )
    assert multiprocessing_spawn.descriptor_file_actions([], fake_os) == ((19, 3),)


def test_close_actions_scale_with_the_number_passed_not_the_highest_descriptor():
    fake_os = SimpleNamespace(
        POSIX_SPAWN_CLOSE=11,
        POSIX_SPAWN_CLOSEFROM=19,
        POSIX_SPAWN_DUP2=17,
    )
    high_fd = 1_000_000
    assert multiprocessing_spawn.descriptor_file_actions([high_fd], fake_os) == (
        (17, high_fd, 3),
        (19, 4),
        (17, 3, high_fd),
        (11, 3),
    )


def test_the_multiprocessing_seam_requires_race_free_closefrom_support():
    incomplete = SimpleNamespace(
        POSIX_SPAWN_CLOSE=11,
        POSIX_SPAWN_DUP2=17,
        posix_spawn=lambda: None,
    )
    assert not multiprocessing_spawn.supports_descriptor_closure(incomplete)


def test_a_fork_applies_the_denial_in_the_child_and_leaves_the_parent_alone(monkeypatch):
    """There is no environment argument to rewrite, so the child rewrites its own."""
    monkeypatch.setenv(_DEVICE, "0")
    assert gpu_guard_spawn.denying_fork(lambda: 4242)() == 4242
    assert os.environ[_DEVICE] == "0"
    assert gpu_guard_spawn.denying_fork(lambda: 0)() == 0
    assert os.environ[_DEVICE] == ""


def test_forkpty_denies_the_child_the_same_way_and_keeps_its_descriptor(monkeypatch):
    monkeypatch.setenv(_DEVICE, "0")
    assert gpu_guard_spawn.denying_forkpty(lambda: (0, 7))() == (0, 7)
    assert os.environ[_DEVICE] == ""


# The two below pin a CPython delegation the seam set RELIES on: `os.execl*` is written in `os.py`
# on top of the four `exec*v*` names and resolves them as module globals at call time, so patching
# those four covers the family. If that stops being true the denial silently narrows -- exactly the
# failure this file exists to close. (`os.spawn*` rides the same names; it is proved end to end in
# `test_gpu_guard_spawn_children.py`, where a real child reports what it saw.)
def test_execl_reaches_the_patched_execv(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(os, "execve", _like_execve(recorder))
    monkeypatch.setattr(
        os, "execv", gpu_guard_spawn.denying_exec_without_env(_like_execve(recorder))
    )
    os.execl("/bin/sh", "sh", "-c", "true")
    (_, args, env), _ = recorder.only
    assert args == ("sh", "-c", "true")
    assert env[_DEVICE] == ""


def test_execlp_reaches_the_patched_execvp(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(os, "execvpe", _like_execvpe(recorder))
    monkeypatch.setattr(
        os, "execvp", gpu_guard_spawn.denying_exec_without_env(_like_execvpe(recorder))
    )
    os.execlp("sh", "sh", "-c", "true")
    (_, args, env), _ = recorder.only
    assert env[_DEVICE] == ""


def test_the_seam_set_names_every_entry_point_the_denial_claims():
    labels = {seam.label for seam in gpu_guard_spawn.spawn_seams()}
    expected = {
        "subprocess.Popen",
        "os.system",
        "os.execve",
        "os.execvpe",
        "os.execv",
        "os.execvp",
        "os.posix_spawn",
        "os.posix_spawnp",
        "os.fork",
        "os.forkpty",
    }
    if multiprocessing_spawn.supports_descriptor_closure():
        expected.add("multiprocessing.util.spawnv_passfds")
    assert labels == expected


def test_applying_the_seams_replaces_every_entry_point_and_restoring_puts_it_back():
    """`denied_children` is what the suite's own fixture installs, so the round trip is the wiring."""
    before = gpu_guard_spawn.spawn_seams()
    with gpu_guard_spawn.denied_children():
        for seam in before:
            assert getattr(seam.module, seam.attribute) is not seam.original
    for seam in before:
        assert getattr(seam.module, seam.attribute) is seam.original


def test_an_entry_point_the_host_does_not_have_yields_no_seam():
    """`fork` / `forkpty` / `posix_spawnp` are POSIX-only; the guard must not be what breaks a host."""
    empty = SimpleNamespace(__name__="fake")
    assert gpu_guard_spawn._seam(empty, "fork", gpu_guard_spawn.denying_fork) is None
    assert (
        gpu_guard_spawn._seam(
            empty, "execv", gpu_guard_spawn.denying_exec_without_env, through="execve"
        )
        is None
    )
