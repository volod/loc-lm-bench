"""Focused llamacpp command implementation."""

import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

DEFAULT_LLAMACPP_HOST = "http://localhost:8080"

OFFLOAD_ALL_LAYERS = -1


class _Process(Protocol):
    returncode: int | None

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class _HttpGetter(Protocol):
    def __call__(self, url: str, timeout: float = 3.0) -> tuple[int, str] | None: ...


def llamacpp_source_args(source: str) -> list[str]:
    """Translate a model source into `llama-server` load args.

    A local `*.gguf` path loads with `-m`; an HF GGUF repo loads with `-hf <repo>[:quant]`
    (llama.cpp downloads + caches it). The Ollama-style `hf.co/<repo>:<quant>` form -- the same
    string the resolver's GGUF sources carry -- maps to the same `-hf` repo by stripping the host
    prefix, so one model spec serves on both Ollama and llama.cpp.
    """
    if source.endswith(".gguf") or source.startswith(("/", "./", "../", "~")):
        return ["-m", source]
    repo = source
    for prefix in ("https://huggingface.co/", "huggingface.co/", "hf.co/"):
        if repo.startswith(prefix):
            repo = repo[len(prefix) :]
            break
    return ["-hf", repo]


def build_llamacpp_command(
    source: str,
    *,
    binary: str = "llama-server",
    bind_host: str = "127.0.0.1",
    port: int = 8080,
    n_gpu_layers: int = OFFLOAD_ALL_LAYERS,
    ctx_size: int | None = None,
    alias: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """The `llama-server ...` argv. `n_gpu_layers` is the GPU/CPU layer split (the llama.cpp
    offload knob) and `ctx_size` (`-c`) the served context, both recorded so served-vs-requested
    context is comparable across runs."""
    cmd = [
        binary,
        *llamacpp_source_args(source),
        "--host",
        bind_host,
        "--port",
        str(port),
        "-ngl",
        str(n_gpu_layers),
    ]
    if ctx_size:
        cmd += ["-c", str(ctx_size)]
    if alias:
        cmd += ["--alias", alias]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def resolve_llama_server_binary(data_dir: Path) -> str:
    """Prefer the project-managed llama.cpp build, then fall back to PATH."""
    built = data_dir / "llb" / "llamacpp" / "build" / "bin" / "llama-server"
    if built.exists():
        return str(built)
    return shutil.which("llama-server") or "llama-server"


def _http_get(url: str, timeout: float = 3.0) -> tuple[int, str] | None:
    """GET -> (status, body) or None on connection error."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return int(resp.status), resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None


_NCTX_PATHS: tuple[tuple[str, ...], ...] = (
    ("default_generation_settings", "n_ctx"),
    ("default_generation_settings", "params", "n_ctx"),
    ("default_generation_settings", "context", "n_ctx"),
    ("generation_settings", "n_ctx"),
    ("model", "n_ctx"),
    ("props", "n_ctx"),
    ("n_ctx",),
)


def _dig(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    obj: Any = data
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def parse_served_context(props_body: str) -> int | None:
    """Pull the served `n_ctx` from a `llama-server` /props response (best-effort, llama.cpp launcher).

    `n_ctx` has moved across llama.cpp versions (top level, `default_generation_settings`, a nested
    `params`/`context`, `generation_settings`, `model`, `props`), so a known set of paths is
    checked in order. Only an exact `n_ctx` int is accepted (never `n_ctx_train`).
    """
    try:
        data = json.loads(props_body)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    for path in _NCTX_PATHS:
        value = _dig(data, path)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return None
