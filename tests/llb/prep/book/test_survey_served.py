"""The vLLM-in-Docker smoke route, over a faked Docker CLI and server -- no container, no GPU."""

import json
import subprocess
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from llb.prep.book.survey import served, vllm_server
from llb.prep.book.survey.candidates import (
    CANDIDATES,
    ROUTE_DOCKER,
    SMOKE_SERVED,
    Candidate,
    Smoke,
    VllmServer,
)
from llb.prep.book.survey.fixture import write_fixture
from llb.prep.book.survey.results import (
    STATUS_FAILED,
    STATUS_NOT_INSTALLED,
    STATUS_OK,
    STATUS_OVER_BUDGET,
)
from llb.prep.book.survey.smoke import run_smoke

GB = 1024**3


def _docker_cli(monkeypatch, runtimes: str, pulled: bool, free_gb: float) -> None:
    def fake(*argv, timeout=120.0):
        if argv[0] == "info":
            return subprocess.CompletedProcess(argv, 0, f"{runtimes}|/docker-root\n", "")
        if argv[:2] == ("image", "inspect"):
            return subprocess.CompletedProcess(argv, 0 if pulled else 1, "", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(vllm_server, "_docker", fake)
    monkeypatch.setattr(vllm_server.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(vllm_server.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(
        vllm_server.shutil, "disk_usage", lambda path: SimpleNamespace(free=free_gb * GB)
    )


def test_docker_without_the_nvidia_runtime_is_refused(monkeypatch):
    _docker_cli(monkeypatch, '{"runc":{}}', pulled=True, free_gb=500)
    assert "NVIDIA Container Toolkit" in vllm_server.docker_problem("img")


def test_an_unpulled_image_is_refused_when_the_docker_root_lacks_room(monkeypatch):
    _docker_cli(monkeypatch, '{"nvidia":{}}', pulled=False, free_gb=15)
    problem = vllm_server.docker_problem("img")
    assert "has 15 GB free" in problem and "data-root" in problem
    _docker_cli(monkeypatch, '{"nvidia":{}}', pulled=False, free_gb=200)
    assert vllm_server.docker_problem("img") is None
    _docker_cli(monkeypatch, '{"nvidia":{}}', pulled=True, free_gb=1)
    assert vllm_server.docker_problem("img") is None  # already pulled: nothing to make room for


def test_a_layout_json_reads_as_its_text_fields_in_order():
    layout = [{"category": "Title", "text": "# A"}, {"category": "Picture"}, {"text": "b"}]
    assert vllm_server.response_text(json.dumps(layout)) == "# A\n\nb"
    assert vllm_server.response_text("plain *markdown*") == "plain *markdown*"


def _served_candidate() -> Candidate:
    return Candidate(
        id="fake-vlm",
        name="fake",
        stages=("ocr",),
        version="0",
        licence="MIT",
        locality="local",
        route=ROUTE_DOCKER,
        source="-",
        claims=(),
        smoke=Smoke(SMOKE_SERVED, model="m", prompt="read", server=VllmServer("m")),
    )


@pytest.fixture()
def page(tmp_path):
    return write_fixture(tmp_path / "page")


def test_a_served_reading_is_scored_and_the_load_time_recorded(tmp_path, page, monkeypatch):
    monkeypatch.setattr(vllm_server, "docker_problem", lambda image: None)

    @contextmanager
    def fake_server(server, label, data_dir, log_path):
        log_path.write_text("ready\n", encoding="utf-8")
        yield "http://127.0.0.1:1"

    monkeypatch.setattr(vllm_server, "running_server", fake_server)
    monkeypatch.setattr(vllm_server, "read_page", lambda *args: page.truth.reading_text())
    result = run_smoke(_served_candidate(), page, tmp_path / "out", tmp_path)
    assert result.status == STATUS_OK, result.error
    assert result.score["word_recall"] == 1.0
    assert "server_load_s" in result.extra


@pytest.mark.parametrize(
    ("log", "status"),
    [
        ("model failed to load", STATUS_FAILED),
        ("CUDA error: out of memory", STATUS_OVER_BUDGET),
        (
            "0.66 GiB KV cache is needed, which is larger than the available KV cache memory",
            STATUS_OVER_BUDGET,
        ),
    ],
)
def test_a_server_that_never_comes_up_is_recorded(tmp_path, page, monkeypatch, log, status):
    monkeypatch.setattr(vllm_server, "docker_problem", lambda image: None)

    @contextmanager
    def failing(server, label, data_dir, log_path):
        log_path.write_text(log, encoding="utf-8")
        raise vllm_server.ServerError("the vLLM container exited during startup")
        yield ""  # pragma: no cover

    monkeypatch.setattr(vllm_server, "running_server", failing)
    result = run_smoke(_served_candidate(), page, tmp_path / "out", tmp_path)
    assert result.status == status
    assert "exited during startup" in result.error and log in result.error


def test_no_docker_room_marks_every_served_row_not_installed(tmp_path, page, monkeypatch):
    monkeypatch.setattr(vllm_server, "docker_problem", lambda image: "no room")
    for candidate in [c for c in CANDIDATES if c.smoke and c.smoke.server]:
        result = run_smoke(candidate, page, tmp_path / candidate.id, tmp_path)
        if candidate.route == ROUTE_DOCKER:
            assert (result.status, result.error) == (STATUS_NOT_INSTALLED, "no room")
    assert served.SERVER_LOG_NAME == "server.log"


def test_a_rejected_request_names_the_server_reason_and_sends_no_token_budget(page, monkeypatch):
    import httpx

    sent = {}

    def post(url, json, timeout):
        sent.update(json)
        return SimpleNamespace(status_code=400, text="max_tokens is too large")

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(vllm_server.ServerError, match="HTTP 400: max_tokens is too large"):
        vllm_server.read_page("http://127.0.0.1:1", "m", "read", page.image, 5.0)
    assert "max_tokens" not in sent and sent["temperature"] == 0
