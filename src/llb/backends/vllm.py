"""vLLM launcher: serve HF weights behind the OpenAI-compatible HTTP API (Premise 1).

vLLM exposes an OpenAI-compatible server (`vllm serve <model>`); this launcher starts it as
a subprocess, waits for readiness, serves chat via the shared `openai_client`, and kills it
on stop. Only the launcher + telemetry are backend-specific -- the eval/RAG/judge code is
unchanged. The actual install (a possibly from-source CUDA build) is `scripts/build_vllm.sh`;
weights are cached by `prep-models`.

`vllm` is invoked as a subprocess (CLI), so this module imports in the base install and is
unit-testable by injecting the process factory + HTTP probe (no vLLM/CUDA needed for tests).
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Protocol, TextIO, cast

from llb.backends.base import BackendLauncher, ChatResult
from llb.backends.openai_client import chat_once, make_client
from llb.contracts import BackendMetadata, ChatMessage
from llb import env


class _Process(Protocol):
    returncode: int | None

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class _HttpGetter(Protocol):
    def __call__(self, url: str, timeout: float = 3.0) -> tuple[int, str] | None: ...


def build_vllm_command(
    model: str,
    *,
    executable: str = "vllm",
    port: int = 8000,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
    dtype: str = "auto",
    quantization: str | None = None,
    served_model_name: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """The `vllm serve ...` argv. `gpu_memory_utilization` is recorded so peak VRAM is
    comparable across runs (vLLM pre-reserves a KV-cache fraction)."""
    cmd = [
        executable,
        "serve",
        model,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        f"{gpu_memory_utilization}",
    ]
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
    if dtype and dtype != "auto":
        cmd += ["--dtype", dtype]
    if quantization:
        cmd += ["--quantization", quantization]
    if served_model_name:
        cmd += ["--served-model-name", served_model_name]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def vllm_executable() -> str | None:
    """Resolve the vLLM CLI installed in the active venv, then fall back to PATH."""
    venv_cli = Path(sys.executable).with_name("vllm")
    if venv_cli.exists():
        return str(venv_cli)
    return shutil.which("vllm")


def _http_get(url: str, timeout: float = 3.0) -> tuple[int, str] | None:
    """GET -> (status, body) or None on connection error."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return int(resp.status), resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None


# vLLM JIT-compiles flashinfer's sampling kernel at engine startup. flashinfer's
# `sampling.cuh` calls `cub::BlockAdjacentDifference::FlagHeads`, which newer CCCL/CUB
# (shipped with CUDA 12.x toolchains) removed -- so the build fails on consumer GPUs such as
# the sm_89 RTX 4060 Ti and the engine never comes up. So the sampler is gated on the
# `build-vllm` preflight (vLLM serving preflight): it is enabled ONLY when the recorded verdict confirms the
# kernel builds on this host, else kept OFF (greedy / temperature-0 decoding, the eval default,
# does not need it). An explicit VLLM_USE_FLASHINFER_SAMPLER in the environment always wins.


def launch_env(
    base: Mapping[str, str] | None = None, *, flashinfer_sampler: bool | None = None
) -> dict[str, str]:
    """Subprocess environment for `vllm serve`: inherit the caller's environment, then set the
    flashinfer-sampler flag from the build-vllm preflight verdict only when the caller has not
    set it explicitly (an explicit value always wins). `flashinfer_sampler` overrides the
    verdict lookup (used in tests)."""
    out = dict(os.environ if base is None else base)
    if env.VLLM_USE_FLASHINFER_SAMPLER not in out:
        if flashinfer_sampler is None:
            from llb.backends.preflight import flashinfer_sampler_ok

            flashinfer_sampler = flashinfer_sampler_ok()
        out[env.VLLM_USE_FLASHINFER_SAMPLER] = "1" if flashinfer_sampler else "0"
    return out


def parse_served_context(models_body: str) -> int | None:
    """Pull `max_model_len` from a vLLM /v1/models response (best-effort)."""
    try:
        data = json.loads(models_body).get("data") or []
    except (ValueError, AttributeError):
        return None
    for entry in data:
        if isinstance(entry, dict) and entry.get("max_model_len"):
            return int(entry["max_model_len"])
    return None


class VllmLauncher(BackendLauncher):
    """Serve one HF model via a `vllm serve` subprocess behind OpenAI-compatible HTTP."""

    def __init__(
        self,
        model: str,
        *,
        host: str = "http://localhost:8000",
        port: int = 8000,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        dtype: str = "auto",
        quantization: str | None = None,
        extra_args: list[str] | None = None,
        startup_timeout: float = 600.0,
        poll_interval: float = 2.0,
        log_dir: Path | str | None = None,
        popen: Callable[..., _Process] | None = None,
        http_get: _HttpGetter | None = None,
        sleep: Callable[[float], None] | None = None,
    ):
        super().__init__(
            model=model,
            meta={
                "backend": "vllm",
                "host": host,
                "gpu_memory_utilization": gpu_memory_utilization,
            },
        )
        self.host = host.rstrip("/")
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.quantization = quantization
        self.extra_args = extra_args
        self.startup_timeout = startup_timeout
        self.poll_interval = poll_interval
        self.log_dir = Path(log_dir) if log_dir else None
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
        return build_vllm_command(
            self.model,
            executable=vllm_executable() or "vllm",
            port=self.port,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            quantization=self.quantization,
            extra_args=self.extra_args,
        )

    def _record_sampler(self, run_env: Mapping[str, str]) -> None:
        """Record which sampler this launch uses (vLLM serving preflight) so the manifest captures it."""
        from llb.backends.preflight import SAMPLER_FLASHINFER, SAMPLER_NATIVE, load_verdict

        use_flashinfer = run_env.get(env.VLLM_USE_FLASHINFER_SAMPLER) == "1"
        self.meta["sampler"] = SAMPLER_FLASHINFER if use_flashinfer else SAMPLER_NATIVE
        verdict = load_verdict()
        self.meta["flashinfer_version"] = verdict["flashinfer_version"] if verdict else None

    def _open_log(self) -> int | TextIO:
        if self.log_dir is None:
            return subprocess.DEVNULL
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"vllm-{self.port}.log"
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        return self._log_handle

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("vLLM launcher is already started")
        log = self._open_log()
        start = time.monotonic()
        run_env = launch_env()
        self._record_sampler(run_env)
        try:
            self._proc = self._popen(
                self.command(), stdout=log, stderr=subprocess.STDOUT, env=run_env
            )
            where = f" (see {self.log_path})" if self.log_path else ""
            polls = max(1, int(self.startup_timeout / self.poll_interval))
            ready_body = None
            for _ in range(polls):
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        f"vLLM exited (code {self._proc.returncode}) during startup{where}"
                    )
                got = self._http_get(f"{self.host}/v1/models")
                if got and got[0] == 200:
                    ready_body = got[1]
                    break
                self._sleep(self.poll_interval)
            else:
                raise RuntimeError(f"vLLM not ready within {self.startup_timeout:.0f}s{where}")
        except BaseException:
            self.stop()
            raise
        self.load_time_s = time.monotonic() - start
        self._served_context = parse_served_context(ready_body or "")
        self.meta["served_context"] = self._served_context
        self._client = make_client(f"{self.host}/v1", api_key="vllm")

    def chat(
        self, messages: list[ChatMessage], max_tokens: int, temperature: float, timeout: float
    ) -> ChatResult:
        if self._client is None:
            self._client = make_client(f"{self.host}/v1", api_key="vllm")
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
