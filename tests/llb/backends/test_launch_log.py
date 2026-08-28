"""A failed launch keeps its server log, and the error says where it is readable.

The log is written inside the run's staging dir, which is deleted as the failure unwinds -- so
these tests remove the staging dir the way a screen cell or tuning trial does, and then assert the
log is still readable at the path the error named.
"""

import shutil
from pathlib import Path

import pytest

from llb.backends.launch_log import ServerLog, failed_log_dir, preserve_log
from llb.backends.llamacpp import LlamaCppLauncher
from llb.backends.vllm import VllmLauncher
from tests.llb.backends.test_vllm import FakeProc


def _staging(tmp_path):
    """A run's staging dir (the temp dir a cell's launcher logs into) and its backend log dir."""
    staging = tmp_path / ".run-eval-20260101T000000Z.tmp"
    return staging, staging / "vllm"


def _spawn(proc, writes):
    """A `popen` stand-in that writes `writes` to the launcher's log handle, as the server would."""

    def popen(cmd, **kw):
        stdout = kw.get("stdout")
        if writes and hasattr(stdout, "write"):
            stdout.write(writes)
        return proc

    return popen


def make_vllm(tmp_path, proc, responses, *, writes="", **kwargs):
    seq = iter(responses)
    _, log_dir = _staging(tmp_path)
    return VllmLauncher(
        "org/Model",
        startup_timeout=5,
        poll_interval=0.1,
        log_dir=log_dir,
        failed_log_dir=tmp_path / "llb" / "logs",
        popen=_spawn(proc, writes),
        http_get=lambda url, timeout=3.0: next(seq, None),
        sleep=lambda _s: None,
        **kwargs,
    )


def _named_path(message):
    """The log path an error message points at."""
    marker = "startup log: "
    assert marker in message, message
    return Path(message.split(marker, 1)[1].rstrip(") "))


def test_dead_engine_log_survives_the_staging_dir_and_the_error_names_it(tmp_path):
    """The whole failure this guards: the temp dir goes, the diagnosis must not go with it."""
    staging, _ = _staging(tmp_path)
    # `writes` stands in for what the engine puts on stdout before it dies.
    launcher = make_vllm(tmp_path, FakeProc(dead=True), [None], writes="CUDA out of memory\n")

    with pytest.raises(RuntimeError, match="exited") as excinfo:
        launcher.start()

    shutil.rmtree(staging)  # the cell returns; its temp run dir is removed
    kept = _named_path(str(excinfo.value))
    assert kept.exists() and kept.read_text(encoding="utf-8") == "CUDA out of memory\n"
    assert staging not in kept.parents  # kept OUTSIDE the dir that was deleted
    assert launcher.failed_log_path == kept


def test_timed_out_launch_preserves_its_log_too(tmp_path):
    launcher = make_vllm(tmp_path, FakeProc(), [None] * 100)
    with pytest.raises(RuntimeError, match="not ready") as excinfo:
        launcher.start()
    assert _named_path(str(excinfo.value)).exists()


def test_llamacpp_launch_failure_preserves_its_log(tmp_path):
    """Every backend that writes a startup log keeps it, not just vLLM."""
    launcher = LlamaCppLauncher(
        "model.gguf",
        startup_timeout=5,
        poll_interval=0.1,
        log_dir=tmp_path / ".run.tmp" / "llamacpp",
        failed_log_dir=tmp_path / "llb" / "logs",
        popen=lambda cmd, **kw: FakeProc(dead=True),
        http_get=lambda url, timeout=3.0: None,
        sleep=lambda _s: None,
    )
    with pytest.raises(RuntimeError, match="exited") as excinfo:
        launcher.start()
    kept = _named_path(str(excinfo.value))
    assert kept.exists() and kept.name.startswith("failed-llamacpp-")


def test_a_launch_that_never_started_raises_with_no_log_claim(tmp_path):
    """`log_dir` unset means stdout went to DEVNULL: the error must not name a log that is not
    there, and preservation must not invent one."""
    launcher = VllmLauncher(
        "org/Model",
        startup_timeout=1,
        poll_interval=0.1,
        popen=lambda cmd, **kw: FakeProc(dead=True),
        http_get=lambda url, timeout=3.0: None,
        sleep=lambda _s: None,
    )
    with pytest.raises(RuntimeError) as excinfo:
        launcher.start()
    assert "startup log" not in str(excinfo.value)
    assert launcher.preserve_failed_log() is None


def test_successful_launch_preserves_nothing(tmp_path):
    """A healthy cell's log going away with its temp dir is the temp dir working as intended."""
    launcher = make_vllm(tmp_path, FakeProc(), [(200, '{"data": []}')])
    launcher.start()
    assert launcher.failed_log_path is None
    assert not (tmp_path / "llb" / "logs").exists()


def test_each_relaunch_attempt_preserves_its_own_log(tmp_path):
    """A durable run relaunches a crashed backend; the second failure must not report the first
    attempt's copy, or the reader diagnoses a log that predates the crash they are chasing."""
    launcher = make_vllm(tmp_path, FakeProc(dead=True), [None] * 10, writes="first crash\n")
    with pytest.raises(RuntimeError):
        launcher.start()
    first = launcher.failed_log_path
    launcher._proc = None  # the runner's relaunch() calls stop() then start()
    with pytest.raises(RuntimeError):
        launcher.start()
    second = launcher.failed_log_path
    assert second != first
    assert first is not None and first.read_text(encoding="utf-8") == "first crash\n"


def test_preserving_twice_within_one_attempt_keeps_one_copy(tmp_path):
    """The launcher preserves on the way out of start(); the runner asks again while tearing the
    staging dir down. One copy, one path."""
    launcher = make_vllm(tmp_path, FakeProc(dead=True), [None], writes="boom\n")
    with pytest.raises(RuntimeError):
        launcher.start()
    kept = launcher.failed_log_path
    assert launcher.preserve_failed_log() == kept
    assert len(list((tmp_path / "llb" / "logs").iterdir())) == 1


def test_two_failures_in_the_same_second_do_not_overwrite_each_other(tmp_path):
    src = tmp_path / "vllm-8000.log"
    dest_dir = tmp_path / "logs"
    src.write_text("first", encoding="utf-8")
    first = preserve_log(src, dest_dir)
    src.write_text("second", encoding="utf-8")
    second = preserve_log(src, dest_dir)
    assert first is not None and second is not None and first != second
    assert first.read_text(encoding="utf-8") == "first"
    assert second.read_text(encoding="utf-8") == "second"


def test_an_unwritable_destination_does_not_mask_the_launch_failure(tmp_path):
    """A lost log is reported as lost; it never replaces the error that lost it."""
    src = tmp_path / "vllm-8000.log"
    src.write_text("boom", encoding="utf-8")
    blocked = tmp_path / "file"
    blocked.write_text("not a directory", encoding="utf-8")
    assert preserve_log(src, blocked / "logs") is None

    class Stub(ServerLog):
        pass

    stub = Stub()
    stub.log_path, stub.failed_log_dir = src, blocked / "logs"
    exc = stub.annotate_launch_failure(RuntimeError("vLLM exited (code 1) during startup"))
    assert "could not be preserved" in str(exc) and str(src) in str(exc)


def test_failed_log_dir_defaults_under_the_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    assert failed_log_dir().parts[-2:] == ("llb", "logs")
    assert failed_log_dir(tmp_path / "explicit") == tmp_path / "explicit" / "llb" / "logs"
