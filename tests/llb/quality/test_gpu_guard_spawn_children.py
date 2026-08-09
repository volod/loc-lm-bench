"""The denial, proved against real children rather than against a recording stand-in.

`test_gpu_guard_spawn.py` pins each seam's argument shape; this file starts an ACTUAL child through
every mechanism the denial claims and reads what that child saw. It is the half that catches a seam
which is patched correctly and still misses -- a family that resolves its exec function some other
way, or a start method that reaches past `os` entirely.

Every case carries `gpu_env` for the same reason the suite-wiring tests do: it takes the SUITE's own
denial out of the way, so what a child sees is decided by the seams the test applies rather than by
the guard already wrapping it. With the suite's denial live, every child reads "" and the denied
case and its control cannot differ.
"""

import multiprocessing
import multiprocessing.util
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from llb.quality import gpu_guard_spawn
from llb.quality import gpu_guard_spawn_multiprocessing as multiprocessing_spawn

_DEVICE = "CUDA_VISIBLE_DEVICES"
# Written by every child below, so one reader covers every mechanism: the denied value is "" and an
# untouched child reports the sentinel.
_UNSET = "unset"
_SHELL_REPORT = 'printf %s "${' + _DEVICE + "-" + _UNSET + '}" > '
_FD_REPORT_SCRIPT = """
import os
import sys

passed_fd, unlisted_fd = map(int, sys.argv[1:])
try:
    os.fstat(unlisted_fd)
except OSError:
    unlisted = "closed"
else:
    unlisted = "open"
os.write(passed_fd, f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}:{unlisted}".encode())
"""


@pytest.fixture(scope="module", autouse=True)
def _stop_forkserver_after_cases():
    """The control starts a visible-device server; do not leave that process behind the module."""
    yield
    multiprocessing_spawn.stop_forkserver()


def _shell_writing(report: Path) -> str:
    return _SHELL_REPORT + str(report)


def _by_subprocess(report: Path) -> None:
    subprocess.run(["sh", "-c", _shell_writing(report)], check=True)


def _by_os_system(report: Path) -> None:
    assert os.system(_shell_writing(report)) == 0


def _by_os_popen(report: Path) -> None:
    """`os.popen` is written on top of `subprocess.Popen`, which it reads off the module at call
    time -- the delegation `llb.quality.gpu_guard_spawn_surface` declares and re-checks."""
    with os.popen(_shell_writing(report)) as pipe:
        assert pipe.close() is None


def _by_spawnv(report: Path) -> None:
    """`os.spawnv` forks and then calls the module-global `execv` -- two seams in one call."""
    assert os.spawnv(os.P_WAIT, "/bin/sh", ["sh", "-c", _shell_writing(report)]) == 0


def _by_spawnlp(report: Path) -> None:
    """The `p` variants resolve on PATH through `execvp`, the other exec seam."""
    assert os.spawnlp(os.P_WAIT, "sh", "sh", "-c", _shell_writing(report)) == 0


def _by_posix_spawn(report: Path) -> None:
    pid = os.posix_spawn("/bin/sh", ["sh", "-c", _shell_writing(report)], dict(os.environ))
    assert os.waitpid(pid, 0)[1] == 0


def _by_posix_spawnp(report: Path) -> None:
    pid = os.posix_spawnp("sh", ["sh", "-c", _shell_writing(report)], dict(os.environ))
    assert os.waitpid(pid, 0)[1] == 0


def _by_raw_fork(report: Path) -> None:
    """A fork is a copy of this process, so the child reads its OWN environment, not an argument."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never reports back to pytest
        try:
            report.write_text(os.environ.get(_DEVICE, _UNSET))
        finally:
            os._exit(0)
    assert os.waitpid(pid, 0)[1] == 0


def _write_own_view(report: Path) -> None:
    report.write_text(os.environ.get(_DEVICE, _UNSET))


def _by_multiprocessing_fork(report: Path) -> None:
    """The default start method on Linux, which reaches `os.fork` rather than any exec seam."""
    child = multiprocessing.get_context("fork").Process(target=_write_own_view, args=(report,))
    child.start()
    child.join()
    assert child.exitcode == 0


def _by_multiprocessing_spawn(report: Path) -> None:
    """The child bootstraps through `spawnv_passfds`; its successful report proves its data pipe
    and resource-tracker descriptor both survived the replacement."""
    child = multiprocessing.get_context("spawn").Process(target=_write_own_view, args=(report,))
    child.start()
    child.join()
    assert child.exitcode == 0


def _by_multiprocessing_forkserver(report: Path) -> None:
    """The server starts through `spawnv_passfds`, then forks the reporting child from itself."""
    child = multiprocessing.get_context("forkserver").Process(
        target=_write_own_view, args=(report,)
    )
    child.start()
    child.join()
    assert child.exitcode == 0


_MECHANISMS: dict[str, Callable[[Path], None]] = {
    "subprocess.run": _by_subprocess,
    "os.system": _by_os_system,
    "os.popen": _by_os_popen,
    "os.spawnv": _by_spawnv,
    "os.spawnlp": _by_spawnlp,
    "os.posix_spawn": _by_posix_spawn,
    "os.posix_spawnp": _by_posix_spawnp,
    "os.fork": _by_raw_fork,
    "multiprocessing(fork)": _by_multiprocessing_fork,
}
if multiprocessing_spawn.supports_descriptor_closure():
    _MECHANISMS.update(
        {
            "multiprocessing(forkserver)": _by_multiprocessing_forkserver,
            "multiprocessing(spawn)": _by_multiprocessing_spawn,
        }
    )


@pytest.mark.gpu_env
def test_spawnv_passfds_keeps_only_the_listed_descriptor(monkeypatch):
    """The real child receives its non-inheritable report pipe, not an unlisted inheritable pipe."""
    report_read, report_write = os.pipe()
    unlisted_read, unlisted_write = os.pipe()
    os.set_inheritable(unlisted_read, True)
    monkeypatch.setenv(_DEVICE, "0")
    try:
        with gpu_guard_spawn.denied_children():
            pid = multiprocessing.util.spawnv_passfds(
                os.fsencode(sys.executable),
                [
                    sys.executable,
                    "-c",
                    _FD_REPORT_SCRIPT,
                    str(report_write),
                    str(unlisted_read),
                ],
                [report_write],
            )
        os.close(report_write)
        report_write = -1
        report = os.read(report_read, 64)
        assert os.waitpid(pid, 0)[1] == 0
        expected = (
            b":closed" if multiprocessing_spawn.supports_descriptor_closure() else b"0:closed"
        )
        assert report == expected
    finally:
        for fd in (report_read, report_write, unlisted_read, unlisted_write):
            if fd >= 0:
                os.close(fd)


def _child_device_view(start_child: Callable[[Path], None], tmp_path: Path) -> str:
    report = tmp_path / "device_view"
    start_child(report)
    return report.read_text()


# The fork cases warn that pytest is multi-threaded, which is true and harmless here: the child
# writes one file and `_exit`s without touching a lock. Silenced by message so a DIFFERENT
# DeprecationWarning out of a spawn seam still surfaces.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.gpu_env
@pytest.mark.parametrize("name", sorted(_MECHANISMS))
def test_a_child_started_any_of_these_ways_finds_no_device(name: str, tmp_path: Path, monkeypatch):
    monkeypatch.setenv(_DEVICE, "0")
    with gpu_guard_spawn.denied_children():
        assert _child_device_view(_MECHANISMS[name], tmp_path) == ""


# The fork cases warn that pytest is multi-threaded, which is true and harmless here: the child
# writes one file and `_exit`s without touching a lock. Silenced by message so a DIFFERENT
# DeprecationWarning out of a spawn seam still surfaces.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.gpu_env
@pytest.mark.parametrize("name", sorted(_MECHANISMS))
def test_the_same_child_keeps_the_device_when_no_seam_is_applied(
    name: str, tmp_path: Path, monkeypatch
):
    """The control: without the seams each mechanism passes the parent's device straight through, so
    the assertion above is about the denial rather than about a host that never had a device."""
    monkeypatch.setenv(_DEVICE, "0")
    assert _child_device_view(_MECHANISMS[name], tmp_path) == "0"
