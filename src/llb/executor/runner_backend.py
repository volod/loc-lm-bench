"""Backend lifecycle for the runner: launcher construction, the pre-launch VRAM guard, runner
resolution, and failure-time staging/log preservation.

`runner.py` calls `_resolve_eval_runner` to wire the launcher + per-case runner + store, and
`_preserve_failed_staging` on the failure paths; the retrieval side (store, runner fn) lives in
`runner_setup.py`.
"""

import logging
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llb.backends.base import BackendLauncher
from llb.core.config import RunConfig
from llb.eval import graph as eval_graph
from llb.executor.runner_retrieval import _load_store
from llb.executor.runner_setup import _default_runner_fn
from llb.goldset.schema import GoldItem

if TYPE_CHECKING:
    from llb.core.contracts.hardware import ContentionReport

RagState = eval_graph.RagState
_LOG = logging.getLogger(__name__)


def _preserve_backend_log(launcher: BackendLauncher, config: RunConfig) -> None:
    """Copy a failed backend's startup log out of the staging dir (which is about to be
    removed) into the persistent logs dir, so a launch failure stays diagnosable instead of
    vanishing with the staging bundle (e.g. a vLLM engine that dies during startup)."""
    log_path = getattr(launcher, "log_path", None)
    src = Path(log_path) if log_path else None
    if src is None or not src.exists():
        return
    dest_dir = config.data_dir / "llb" / "logs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = dest_dir / f"failed-{src.stem}-{stamp}.log"
    try:
        shutil.copyfile(src, dest)
    except OSError:
        return
    _LOG.error("[run-eval] backend failed to start; startup log preserved -> %s", dest)


def _make_launcher(config: RunConfig, log_dir: Path | None = None) -> BackendLauncher:
    if config.adapter_path is not None and config.backend != "vllm":
        raise SystemExit(
            f"[run-eval] adapter serving is wired for vLLM LoRA modules; backend "
            f"{config.backend!r} needs a merged model artifact first"
        )
    if config.backend == "ollama":
        from llb.backends.ollama import OllamaLauncher

        return OllamaLauncher(config.model, host=config.ollama_host)
    if config.backend == "vllm":
        from llb.backends.vllm import VllmLauncher
        from llb.finetune.adapter_manifest import adapter_lora_rank

        return VllmLauncher(
            config.model,
            host=config.vllm_host,
            port=config.vllm_port,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            cpu_offload_gb=config.cpu_offload_gb,
            kv_offloading_size_gb=config.kv_offloading_size_gb,
            dtype=config.dtype,
            quantization=config.quantization,
            adapter_path=config.adapter_path,
            max_lora_rank=adapter_lora_rank(config.adapter_path),
            log_dir=log_dir,
        )
    if config.backend == "llamacpp":
        from llb.backends.llamacpp import LlamaCppLauncher
        from llb.backends.llamacpp_command import resolve_llama_server_binary

        return LlamaCppLauncher(
            config.model,
            host=config.llamacpp_host,
            n_gpu_layers=config.n_gpu_layers,
            ctx_size=config.max_model_len,
            log_dir=log_dir,
            binary=resolve_llama_server_binary(config.data_dir),
        )
    raise SystemExit(f"backend '{config.backend}' is not wired (ollama, vllm, llamacpp supported).")


def _vram_reader() -> Callable[[], int] | None:
    """Best-effort NVML reader for telemetry (None when the [telemetry] extra/GPU is absent)."""
    try:
        from llb.executor.vram import nvml_reader

        return nvml_reader()
    except (Exception, SystemExit):  # nvml_reader raises SystemExit when [telemetry] is absent
        return None


def _pid_usage_reader() -> Callable[[], dict[int, int]] | None:
    """Best-effort NVML per-PID VRAM reader (for the VRAM contention guard contention guard's resident attribution)."""
    try:
        from llb.executor.vram import nvml_process_reader

        return nvml_process_reader()
    except (Exception, SystemExit):
        return None


def _guard_vllm_contention(
    config: RunConfig, launcher: BackendLauncher, *, evict: bool, wait: bool
) -> "ContentionReport | None":
    """Pre-launch VRAM-contention guard for vLLM (VRAM contention guard): derate gpu-memory-utilization to the
    actually-free VRAM, or abort if even that cannot hold the model. No-op without a GPU."""
    from llb.backends.vllm import VllmLauncher
    from llb.executor.contention import (
        ACTION_ABORT,
        apply_contention_guard,
        default_gpu_reader,
    )
    from llb.executor.contention_memory import model_kv_headroom_mb, model_weight_floor_mb

    report = apply_contention_guard(
        requested_util=config.gpu_memory_utilization,
        weight_floor_mb=model_weight_floor_mb(config.model),
        gpu_reader=default_gpu_reader,
        process_reader=_pid_usage_reader(),
        evict=evict,
        wait=wait,
        ollama_host=config.ollama_host,
        min_kv_headroom_mb=model_kv_headroom_mb(config.model),
    )
    if report is None:
        return None
    if report["action"] == ACTION_ABORT:
        raise SystemExit(f"[run-eval] pre-launch VRAM guard: {report['note']}")
    if report["derated"] and isinstance(launcher, VllmLauncher):
        _LOG.warning("[run-eval] %s", report["note"])
        launcher.gpu_memory_utilization = report["safe_util"]
        launcher.meta["gpu_memory_utilization"] = report["safe_util"]
    else:
        _LOG.info("[run-eval] pre-launch VRAM guard: %s", report["note"])
    return report


def _resolve_eval_runner(
    config: RunConfig,
    *,
    store: Any,
    launcher: BackendLauncher | None,
    runner_fn: Callable[[GoldItem], RagState] | None,
    prompt_package: Any | None,
    staging_dir: Path,
    evict: bool,
    wait: bool,
) -> tuple[BackendLauncher, Callable[[GoldItem], RagState], Any, "ContentionReport | None"]:
    contention: ContentionReport | None = None
    if launcher is None:
        launcher = _make_launcher(config, log_dir=staging_dir / "vllm")
        if config.backend == "vllm":
            contention = _guard_vllm_contention(config, launcher, evict=evict, wait=wait)
    if runner_fn is None:
        if store is None:
            store = _load_store(config)
        runner_fn = _default_runner_fn(config, store, launcher, prompt_package)
    return launcher, runner_fn, store, contention


def _preserve_failed_staging(
    active_launcher: BackendLauncher | None,
    config: RunConfig,
    resume: Path | str | None,
    run_dir: Path,
    staging_dir: Path,
    *,
    interrupted: bool,
) -> None:
    """On failure: keep the backend log; keep staging only when it can seed a --resume."""
    if active_launcher is not None:
        _preserve_backend_log(active_launcher, config)
    if interrupted:
        _LOG.warning(
            "[run-eval] interrupted; staging preserved -- resume with --resume %s", run_dir
        )
    elif resume is None:
        shutil.rmtree(staging_dir, ignore_errors=True)
    else:
        _LOG.warning("[run-eval] resume failed; staging kept for another --resume %s", run_dir)
