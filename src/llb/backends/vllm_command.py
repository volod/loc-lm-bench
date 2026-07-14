"""Focused vllm command implementation."""

import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol
from llb.core import env


class _Process(Protocol):
    returncode: int | None

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class _HttpGetter(Protocol):
    def __call__(self, url: str, timeout: float = 3.0) -> tuple[int, str] | None: ...


VLLM_MAX_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def served_lora_rank(rank: int) -> int:
    """The smallest `--max-lora-rank` vLLM accepts that can still hold `rank`."""
    for allowed in VLLM_MAX_LORA_RANKS:
        if rank <= allowed:
            return allowed
    raise SystemExit(
        f"[vllm] LoRA rank {rank} exceeds the largest servable rank {VLLM_MAX_LORA_RANKS[-1]}"
    )


def build_vllm_command(
    model: str,
    *,
    executable: str = "vllm",
    port: int = 8000,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
    cpu_offload_gb: float | None = None,
    kv_offloading_size_gb: float | None = None,
    dtype: str = "auto",
    quantization: str | None = None,
    adapter_path: str | None = None,
    adapter_name: str = "adapter",
    max_lora_rank: int | None = None,
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
    if cpu_offload_gb:
        cmd += ["--cpu-offload-gb", f"{cpu_offload_gb:g}"]
    if kv_offloading_size_gb:
        cmd += ["--kv-offloading-size", f"{kv_offloading_size_gb:g}"]
    if dtype and dtype != "auto":
        cmd += ["--dtype", dtype]
    if quantization:
        cmd += ["--quantization", quantization]
    if adapter_path:
        cmd += ["--enable-lora", "--lora-modules", f"{adapter_name}={adapter_path}"]
        if max_lora_rank:
            cmd += ["--max-lora-rank", str(served_lora_rank(max_lora_rank))]
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
            from llb.backends.preflight_verdict import flashinfer_sampler_ok

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
