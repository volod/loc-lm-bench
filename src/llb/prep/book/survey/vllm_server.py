"""Serve a candidate model from vLLM in Docker for one smoke, then read a page through it.

Several document VLMs ship only for vLLM, and every development host is a CUDA host with the NVIDIA
Container Toolkit (`make venv` refuses one without it), so the survey runs them from the pinned
`vllm/vllm-openai` image. The container is the survey's own, not a tool's self-spawned one, because
a tool's spawn runs as root and mounts the home directory's model cache: here it runs as the
invoking user, with Hugging Face downloads under `$DATA_DIR/cache/book-survey/huggingface`, bound
to 127.0.0.1, and is stopped and removed -- after its log is kept -- when the smoke ends, whatever happened.

The image is about 20 GB unpacked. Before pulling one, the Docker data root must have room for it;
a host whose Docker root lacks the room is refused by name instead of filling its filesystem.
"""

import base64
import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from llb.prep.book.survey.candidates import VllmServer
from llb.prep.book.survey.envs import CACHE_DIRNAME, SURVEY_CACHE_DIRNAME

CONTAINER_PORT = 8000
IMAGE_ROOM_GB = 30  # free space the Docker root needs before an image pull
BYTES_PER_GB = 1024**3
LOAD_TIMEOUT_S = 1800.0  # includes the first download of the model weights
HEALTH_POLL_S = 3.0
LOG_TAIL_LINES = 200
CONTAINER_PREFIX = "llb-survey-"


class ServerError(RuntimeError):
    """The server could not start, died, or never became healthy."""


def _docker(*argv: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *argv], capture_output=True, text=True, timeout=timeout, check=False
    )


def docker_problem(image: str) -> str | None:
    """Why `image` cannot be served here, or None when it can."""
    if shutil.which("docker") is None:
        return "docker is not installed"
    info = _docker("info", "--format", "{{json .Runtimes}}|{{.DockerRootDir}}")
    if info.returncode != 0:
        return f"docker is not usable by this user: {info.stderr.strip()[:200]}"
    runtimes, _, root = info.stdout.strip().partition("|")
    if '"nvidia"' not in runtimes:
        return "docker has no 'nvidia' runtime: install the NVIDIA Container Toolkit"
    if _docker("image", "inspect", image).returncode == 0:
        return None
    free_gb = shutil.disk_usage(root).free / BYTES_PER_GB if os.path.isdir(root) else None
    if free_gb is not None and free_gb < IMAGE_ROOM_GB:
        return (
            f"{image} is not pulled and the Docker data root {root} has {free_gb:.0f} GB free "
            f"(needs {IMAGE_ROOM_GB} GB); move Docker's data-root to a larger volume"
        )
    return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


def _run_command(server: VllmServer, name: str, port: int, hf_home: Path) -> list[str]:
    return [
        "run",
        "-d",
        "--name",
        name,
        "--runtime",
        "nvidia",
        "--gpus",
        "all",
        "--ipc=host",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        # The invoking uid has no passwd entry in the image; getpass and torch's compile cache ask.
        "-e",
        "USER=llb",
        "-e",
        "LOGNAME=llb",
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
        "-e",
        "HF_HOME=/hf",
        "-e",
        "VLLM_NO_USAGE_STATS=1",
        "-e",
        "DO_NOT_TRACK=1",
        "-v",
        f"{hf_home}:/hf",
        "-p",
        f"127.0.0.1:{port}:{CONTAINER_PORT}",
        server.image,
        "--model",
        server.model,
        "--served-model-name",
        server.model,
        *server.args,
    ]


def _running(name: str) -> bool:
    state = _docker("inspect", "--format", "{{.State.Running}}", name)
    return state.returncode == 0 and state.stdout.strip() == "true"


def _wait_healthy(name: str, base: str, deadline: float) -> None:
    import httpx

    while time.monotonic() < deadline:
        if not _running(name):
            raise ServerError("the vLLM container exited during startup")
        try:
            if httpx.get(f"{base}/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(HEALTH_POLL_S)
    raise ServerError(f"the vLLM server was not healthy within {LOAD_TIMEOUT_S:.0f} s")


@contextmanager
def running_server(server: VllmServer, label: str, data_dir: Path, log_path: Path) -> Iterator[str]:
    """Start the container, yield its base URL once healthy, and always stop it."""
    hf_home = data_dir / CACHE_DIRNAME / SURVEY_CACHE_DIRNAME / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    name, port = f"{CONTAINER_PREFIX}{label}", _free_port()
    _docker("rm", "-f", name)  # a leftover from an interrupted smoke
    if _docker("image", "inspect", server.image).returncode != 0:
        pulled = _docker("pull", server.image, timeout=LOAD_TIMEOUT_S)
        if pulled.returncode != 0:
            raise ServerError(f"docker pull {server.image} failed: {pulled.stderr.strip()[:300]}")
    started = _docker(*_run_command(server, name, port, hf_home))
    if started.returncode != 0:
        raise ServerError(f"docker run failed: {started.stderr.strip()[:300]}")
    try:
        base = f"http://127.0.0.1:{port}"
        _wait_healthy(name, base, time.monotonic() + LOAD_TIMEOUT_S)
        yield base
    finally:
        logs = _docker("logs", "--tail", str(LOG_TAIL_LINES), name)
        log_path.write_text(logs.stdout + logs.stderr, encoding="utf-8")
        _docker("stop", "-t", "10", name)
        _docker("rm", "-f", name)  # removed only after its logs are kept


def response_text(raw: str) -> str:
    """The page text of a response: a layout JSON's `text` fields in order, else the raw text."""
    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return "\n\n".join(str(item["text"]) for item in parsed if item.get("text"))
    return raw


def _page_bytes(image: Path, max_side: int | None) -> tuple[bytes, str]:
    """The page as sent: unchanged, or scaled so its longer side is `max_side` (PNG)."""
    kind = "png" if image.suffix == ".png" else "jpeg"
    if max_side is None:
        return image.read_bytes(), kind
    import fitz

    pix = fitz.Pixmap(str(image))
    scale = max_side / max(pix.width, pix.height)
    if scale >= 1:
        return image.read_bytes(), kind
    scaled = fitz.Pixmap(pix, round(pix.width * scale), round(pix.height * scale), None)
    data: bytes = scaled.tobytes("png")
    return data, "png"


def read_page(
    base: str, model: str, prompt: str, image: Path, timeout_s: float, max_side: int | None = None
) -> str:
    """One OpenAI-compatible chat request with the page image; returns the reading."""
    import httpx

    data, kind = _page_bytes(image, max_side)
    encoded = base64.b64encode(data).decode("ascii")
    message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64,{encoded}"}},
            {"type": "text", "text": prompt},
        ],
    }
    response = httpx.post(
        f"{base}/v1/chat/completions",
        # No max_tokens: vLLM then allows whatever the served context leaves after the prompt, and a
        # fixed budget larger than that is a 400.
        json={"model": model, "messages": [message], "temperature": 0},
        timeout=timeout_s,
    )
    if response.status_code >= 400:
        raise ServerError(f"HTTP {response.status_code}: {response.text[:500]}")
    return response_text(str(response.json()["choices"][0]["message"]["content"] or ""))
