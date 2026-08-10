"""Contract and suite wiring for the lightweight tier's no-download guard."""

import importlib.util
import socket
from pathlib import Path
from types import ModuleType

import pytest

from llb.core import env
from llb.quality import download_guard

_CONFTEST = Path(__file__).resolve().parents[2] / "conftest.py"
_REMOTE = ("example.invalid", 443)


def _suite_conftest() -> ModuleType:
    spec = importlib.util.spec_from_file_location("llb_suite_conftest_download", _CONFTEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("family", "address"),
    [
        (socket.AF_INET, ("localhost", 8000)),
        (socket.AF_INET, ("localhost.", 8000)),
        (socket.AF_INET, ("127.0.0.2", 8000)),
        (socket.AF_INET6, ("::1", 8000, 0, 0)),
        (socket.AF_INET6, ("::ffff:127.0.0.1", 8000, 0, 0)),
        (socket.AF_UNIX, "/tmp/fake-server.sock"),
    ],
)
def test_local_destinations_are_allowed(family: int, address: object):
    assert download_guard.is_loopback_destination(family, address)


@pytest.mark.parametrize("host", ["example.invalid", "192.0.2.10", "2001:db8::1"])
def test_non_loopback_destinations_are_guarded(host: str):
    assert not download_guard.is_loopback_destination(socket.AF_INET, (host, 443))


@pytest.mark.parametrize("marker", download_guard.EXEMPT_MARKERS)
def test_a_declared_marker_takes_the_guard_out_of_the_way(marker: str):
    assert download_guard.DownloadGuard.start("tests/x.py::test_y", [marker], environ={}) is None


def test_default_refusal_blocks_before_the_connector_runs(monkeypatch):
    called = False

    def connector(_sock, _address):
        nonlocal called
        called = True

    monkeypatch.setattr(socket.socket, "connect", connector)
    guard = download_guard.DownloadGuard.start("tests/x.py::test_y", [], environ={})
    assert guard is not None
    with guard.connections(), socket.socket() as client:
        with pytest.raises(download_guard.DownloadGuardError, match=r"\[download-guard\].*test_y"):
            client.connect(_REMOTE)
    assert called is False


def test_loopback_reaches_a_real_fake_server():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        address = server.getsockname()
        with socket.socket() as client:
            client.connect(address)
            connection, _ = server.accept()
            connection.close()


def test_report_mode_warns_and_allows_the_connector(monkeypatch):
    reached: list[object] = []

    def connector(_sock, address):
        reached.append(address)

    monkeypatch.setattr(socket.socket, "connect_ex", connector)
    guard = download_guard.DownloadGuard.start(
        "tests/x.py::test_y", [], environ={env.DOWNLOAD_GUARD: download_guard.MODE_REPORT}
    )
    assert guard is not None
    with guard.connections(), socket.socket() as client:
        with pytest.warns(UserWarning, match=r"\[download-guard\]"):
            client.connect_ex(_REMOTE)
    assert reached == [_REMOTE]


def test_off_disables_the_guard():
    environ = {env.DOWNLOAD_GUARD: download_guard.MODE_OFF}
    assert download_guard.DownloadGuard.start("tests/x.py::test_y", [], environ=environ) is None


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="LLB_DOWNLOAD_GUARD"):
        download_guard.guard_mode({env.DOWNLOAD_GUARD: "yes"})


@pytest.mark.network_env
def test_the_suite_wiring_refuses_an_unmarked_tests_connection(monkeypatch):
    called = False

    def connector(_sock, _address):
        nonlocal called
        called = True

    monkeypatch.setattr(socket.socket, "connect", connector)
    monkeypatch.setenv(env.DOWNLOAD_GUARD, download_guard.MODE_REFUSE)
    steps = _suite_conftest().download_guard_steps("tests/x.py::test_y", [])
    next(steps)
    with socket.socket() as client:
        with pytest.raises(download_guard.DownloadGuardError, match=r"\[download-guard\]"):
            client.connect(_REMOTE)
    assert called is False
    assert next(steps, None) is None


@pytest.mark.network_env
def test_the_suite_wiring_leaves_a_declared_test_alone(monkeypatch):
    original = socket.socket.connect
    monkeypatch.setenv(env.DOWNLOAD_GUARD, download_guard.MODE_REFUSE)
    steps = _suite_conftest().download_guard_steps("tests/x.py::test_y", ["network_env"])
    next(steps)
    assert socket.socket.connect is original
    assert next(steps, None) is None
