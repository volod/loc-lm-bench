"""llama.cpp launcher: serve a GGUF behind the OpenAI-compatible HTTP API (Premise 1).

The third backend the backend resolver resolver routes to: an oversized model that vLLM cannot hold in
VRAM (no CPU offload) resolves to its GGUF, which `llama-server` runs by splitting layers
between the GPU (`-ngl`) and system RAM. `llama-server` exposes an OpenAI-compatible endpoint,
so -- exactly like the Ollama and vLLM launchers -- only this launcher + its telemetry are
backend-specific; the eval/RAG/judge code is unchanged.

`llama-server` is invoked as a subprocess (CLI), so this module imports in the base install and
is unit-testable by injecting the process factory + HTTP probe (no llama.cpp/CUDA needed). The
binary itself is a separate hardware-matched build, like vLLM (AGENTS.md).
"""

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol, TextIO, cast
from urllib.parse import urlsplit

from llb.backends.base import BackendLauncher, ChatResult
from llb.backends.openai_client import chat_once, make_client
from llb.contracts import BackendMetadata, ChatMessage

DEFAULT_LLAMACPP_HOST = "http://localhost:8080"
# llama.cpp convention: a negative gpu-layer count offloads EVERY layer to the GPU. The runner
# passes this through from RunConfig; for an offload model set it to the planner's `gpu_layers`
# split so the layers that do not fit VRAM land in system RAM instead of OOMing.
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


# Locations of the served `n_ctx` across llama-server versions (most-specific first). NOTE: this
# is the SERVED context, never the model's `n_ctx_train`, so only an exact `n_ctx` key is read.
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


class LlamaCppLauncher(BackendLauncher):
    """Serve one GGUF model via a `llama-server` subprocess behind OpenAI-compatible HTTP."""

    def __init__(
        self,
        model: str,
        *,
        host: str = DEFAULT_LLAMACPP_HOST,
        n_gpu_layers: int = OFFLOAD_ALL_LAYERS,
        ctx_size: int | None = None,
        extra_args: list[str] | None = None,
        startup_timeout: float = 600.0,
        poll_interval: float = 2.0,
        log_dir: Path | str | None = None,
        binary: str = "llama-server",
        popen: Callable[..., _Process] | None = None,
        http_get: _HttpGetter | None = None,
        sleep: Callable[[float], None] | None = None,
    ):
        super().__init__(
            model=model,
            meta={
                "backend": "llamacpp",
                "host": host,
                "n_gpu_layers": n_gpu_layers,
                "ctx_size": ctx_size,
            },
        )
        self.host = host.rstrip("/")
        parsed = urlsplit(host)
        self._bind_host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 8080
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.extra_args = extra_args
        self.startup_timeout = startup_timeout
        self.poll_interval = poll_interval
        self.log_dir = Path(log_dir) if log_dir else None
        self.binary = binary
        self._popen = popen or cast(Callable[..., _Process], subprocess.Popen)
        self._http_get = http_get or _http_get
        self._sleep = sleep or time.sleep
        self._proc: _Process | None = None
        self._client: Any = None
        self._served_context: int | None = None
        self._last: ChatResult | None = None
        self._log_handle: TextIO | None = None
        self.log_path: Path | None = None

    def command(self) -> list[str]:
        return build_llamacpp_command(
            self.model,
            binary=self.binary,
            bind_host=self._bind_host,
            port=self.port,
            n_gpu_layers=self.n_gpu_layers,
            ctx_size=self.ctx_size,
            alias=self.model,
            extra_args=self.extra_args,
        )

    def _open_log(self) -> int | TextIO:
        if self.log_dir is None:
            return subprocess.DEVNULL
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"llamacpp-{self.port}.log"
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        return self._log_handle

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("llama.cpp launcher is already started")
        log = self._open_log()
        start = time.monotonic()
        try:
            self._proc = self._popen(self.command(), stdout=log, stderr=subprocess.STDOUT)
            where = f" (see {self.log_path})" if self.log_path else ""
            polls = max(1, int(self.startup_timeout / self.poll_interval))
            for _ in range(polls):
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited (code {self._proc.returncode}) during startup{where}"
                    )
                got = self._http_get(f"{self.host}/health")
                if got and got[0] == 200:
                    break
                self._sleep(self.poll_interval)
            else:
                raise RuntimeError(
                    f"llama-server not ready within {self.startup_timeout:.0f}s{where}"
                )
        except BaseException:
            self.stop()
            raise
        self.load_time_s = time.monotonic() - start
        self._served_context = self._read_served_context()
        self.meta["served_context"] = self._served_context
        self._client = make_client(f"{self.host}/v1", api_key="llamacpp")

    def _read_served_context(self) -> int | None:
        """The actually-served context from /props, falling back to the requested ctx_size."""
        got = self._http_get(f"{self.host}/props")
        served = parse_served_context(got[1]) if got and got[0] == 200 else None
        return served if served is not None else self.ctx_size

    def chat(
        self, messages: list[ChatMessage], max_tokens: int, temperature: float, timeout: float
    ) -> ChatResult:
        if self._client is None:
            self._client = make_client(f"{self.host}/v1", api_key="llamacpp")
        self._last = chat_once(
            self._client,
            self.model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        return self._last

    def served_context(self) -> int | None:
        return self._served_context

    def stop(self) -> None:
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=20)
        finally:
            self._proc = None
            self._client = None
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None

    def telemetry(self) -> BackendMetadata:
        out = dict(self.meta)
        if self.load_time_s is not None:
            out["load_time_s"] = round(self.load_time_s, 2)
        if self._last is not None and not self._last.error:
            out["tokens_per_s"] = round(self._last.tokens_per_s(), 2)
        return cast(BackendMetadata, out)
