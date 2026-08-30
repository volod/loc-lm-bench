"""vLLM launcher: serve HF weights behind the OpenAI-compatible HTTP API (Premise 1).

vLLM exposes an OpenAI-compatible server (`vllm serve <model>`); this launcher starts it as
a subprocess, waits for readiness, serves chat via the shared `openai_client`, and kills it
on stop. Only the launcher + telemetry are backend-specific -- the eval/RAG/judge code is
unchanged. The actual install (a possibly from-source CUDA build) is `scripts/build_vllm.sh`;
weights are cached by `prep-models`.

`vllm` is invoked as a subprocess (CLI), so this module imports in the base install and is
unit-testable by injecting the process factory + HTTP probe (no vLLM/CUDA needed for tests).
"""

import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, cast

from llb.backends.base import BackendLauncher, ChatResult
from llb.backends.launch_log import ServerLog
from llb.backends.openai_client import chat_once, make_client
from llb.core.contracts.hardware import BackendMetadata
from llb.core.contracts.common import ChatMessage
from llb.core import env
from llb.backends.vllm_command import (
    _HttpGetter,
    _Process,
    _http_get,
    build_vllm_command,
    launch_env,
    parse_served_context,
    served_lora_rank,
    vllm_executable,
)


# `--max-lora-rank` only accepts these values, and it defaults to 16: an adapter trained at a higher
# rank makes `add_lora` fail at startup ("LoRA rank 64 is greater than max_lora_rank 16"), so the
# launcher sizes the flag from the adapter it is about to serve, rounding UP to the nearest value.


# vLLM JIT-compiles flashinfer's sampling kernel at engine startup. flashinfer's
# `sampling.cuh` calls `cub::BlockAdjacentDifference::FlagHeads`, which newer CCCL/CUB
# (shipped with CUDA 12.x toolchains) removed -- so the build fails on consumer GPUs such as
# the sm_89 RTX 4060 Ti and the engine never comes up. So the sampler is gated on the
# `build-vllm` preflight (vLLM serving preflight): it is enabled ONLY when the recorded verdict confirms the
# kernel builds on this host, else kept OFF (greedy / temperature-0 decoding, the eval default,
# does not need it). An explicit VLLM_USE_FLASHINFER_SAMPLER in the environment always wins.


class VllmLauncher(BackendLauncher, ServerLog):
    """Serve one HF model via a `vllm serve` subprocess behind OpenAI-compatible HTTP."""

    def __init__(
        self,
        model: str,
        *,
        host: str = "http://localhost:8000",
        port: int = 8000,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        cpu_offload_gb: float | None = None,
        kv_offloading_size_gb: float | None = None,
        dtype: str = "auto",
        quantization: str | None = None,
        adapter_path: Path | str | None = None,
        adapter_name: str = "adapter",
        max_lora_rank: int | None = None,
        suppress_thinking: bool = False,
        extra_args: list[str] | None = None,
        startup_timeout: float = 600.0,
        poll_interval: float = 2.0,
        log_dir: Path | str | None = None,
        failed_log_dir: Path | str | None = None,
        popen: Callable[..., _Process] | None = None,
        http_get: _HttpGetter | None = None,
        sleep: Callable[[float], None] | None = None,
        client_factory: Callable[[], Any] | None = None,
    ):
        super().__init__(
            model=model,
            meta={
                "backend": "vllm",
                "host": host,
                "gpu_memory_utilization": gpu_memory_utilization,
                "cpu_offload_gb": cpu_offload_gb,
                "kv_offloading_size_gb": kv_offloading_size_gb,
                "adapter_path": str(adapter_path) if adapter_path else None,
                "adapter_name": adapter_name if adapter_path else None,
                "max_lora_rank": served_lora_rank(max_lora_rank)
                if (adapter_path and max_lora_rank)
                else None,
            },
        )
        self.host = host.rstrip("/")
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.cpu_offload_gb = cpu_offload_gb
        self.kv_offloading_size_gb = kv_offloading_size_gb
        self.dtype = dtype
        self.quantization = quantization
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.adapter_name = adapter_name
        self.max_lora_rank = max_lora_rank
        # OFF by default: vLLM's reasoning-control fields are not in the OpenAI schema, so a
        # server that models its request body strictly can reject them. With this unset the
        # request carries no `extra_body` at all and is byte-identical to the shipped shape.
        self.suppress_thinking = suppress_thinking
        self._extra_body: dict[str, object] = {}
        self.request_model = adapter_name if adapter_path else model
        self.extra_args = extra_args
        self.startup_timeout = startup_timeout
        self.poll_interval = poll_interval
        self.log_dir = Path(log_dir) if log_dir else None
        self.failed_log_dir = Path(failed_log_dir) if failed_log_dir else None
        self._client_factory = client_factory
        self._popen = popen or cast(Callable[..., _Process], subprocess.Popen)
        self._http_get = http_get or _http_get
        self._sleep = sleep or time.sleep
        self._proc: _Process | None = None
        self._client: Any = None
        self._served_context: int | None = None
        self._last: ChatResult | None = None

    def command(self) -> list[str]:
        return build_vllm_command(
            self.model,
            executable=vllm_executable() or "vllm",
            port=self.port,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            cpu_offload_gb=self.cpu_offload_gb,
            kv_offloading_size_gb=self.kv_offloading_size_gb,
            dtype=self.dtype,
            quantization=self.quantization,
            adapter_path=self.adapter_path,
            adapter_name=self.adapter_name,
            max_lora_rank=self.max_lora_rank,
            extra_args=self.extra_args,
        )

    def _record_sampler(self, run_env: Mapping[str, str]) -> None:
        """Record which sampler this launch uses (vLLM serving preflight) so the manifest captures it."""
        from llb.backends.preflight_verdict import (
            SAMPLER_FLASHINFER,
            SAMPLER_NATIVE,
            load_verdict,
        )

        use_flashinfer = run_env.get(env.VLLM_USE_FLASHINFER_SAMPLER) == "1"
        self.meta["sampler"] = SAMPLER_FLASHINFER if use_flashinfer else SAMPLER_NATIVE
        verdict = load_verdict()
        self.meta["flashinfer_version"] = verdict["flashinfer_version"] if verdict else None

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("vLLM launcher is already started")
        log = self.open_log(f"vllm-{self.port}.log")
        start = time.monotonic()
        run_env = launch_env()
        self._record_sampler(run_env)
        try:
            self._proc = self._popen(
                self.command(), stdout=log, stderr=subprocess.STDOUT, env=run_env
            )
            polls = max(1, int(self.startup_timeout / self.poll_interval))
            ready_body = None
            for _ in range(polls):
                if self._proc.poll() is not None:
                    raise RuntimeError(f"vLLM exited (code {self._proc.returncode}) during startup")
                got = self._http_get(f"{self.host}/v1/models")
                if got and got[0] == 200:
                    ready_body = got[1]
                    break
                self._sleep(self.poll_interval)
            else:
                raise RuntimeError(f"vLLM not ready within {self.startup_timeout:.0f}s")
        except BaseException as exc:
            # stop() first (it closes the log handle), so the preserved copy is complete; the
            # error then names where the log is readable, not the staging path about to vanish.
            self.stop()
            raise self.annotate_launch_failure(exc)
        self.load_time_s = time.monotonic() - start
        self._served_context = parse_served_context(ready_body or "")
        self.meta["served_context"] = self._served_context
        self._client = self._connect()
        self._resolve_thinking_suppression()

    def _connect(self) -> Any:
        """The OpenAI-compatible client for the served endpoint (injectable for tests)."""
        if self._client_factory is not None:
            return self._client_factory()
        return make_client(f"{self.host}/v1", api_key="vllm")

    def _resolve_thinking_suppression(self) -> None:
        """Settle which reasoning-control fields this server takes, once per vLLM version.

        Probed only when the caller opted in; the verdict is cached under `$DATA_DIR`, so the
        cost is one 1-token generation on the first launch of a given vLLM and nothing after.
        """
        from llb.backends.vllm_reasoning import FIELDS_NONE, resolve_extra_body

        if not self.suppress_thinking:
            self.meta["thinking_suppression"] = FIELDS_NONE
            return
        self._extra_body, level = resolve_extra_body(self._client, self.request_model)
        self.meta["thinking_suppression"] = level

    def chat(
        self, messages: list[ChatMessage], max_tokens: int, temperature: float, timeout: float
    ) -> ChatResult:
        if self._client is None:
            self._client = self._connect()
        self._last = chat_once(
            self._client,
            self.request_model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            extra_body=self._extra_body or None,
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
            self.close_log()

    def telemetry(self) -> BackendMetadata:
        out = dict(self.meta)
        if self.load_time_s is not None:
            out["load_time_s"] = round(self.load_time_s, 2)
        if self._last is not None and not self._last.error:
            out["tokens_per_s"] = round(self._last.tokens_per_s(), 2)
        return cast(BackendMetadata, out)
