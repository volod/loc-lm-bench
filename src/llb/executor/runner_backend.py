"""Backend lifecycle for the runner: launcher construction, the pre-launch VRAM guard, runner
resolution, and failure-time staging/log preservation.

`runner.py` calls `_resolve_eval_runner` to wire the launcher + per-case runner + store, and
`_preserve_failed_staging` on the failure paths; the retrieval side (store, runner fn) lives in
`runner_setup.py`.
"""

import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from llb.backends.base import BackendLauncher
from llb.backends.launch_log import ServerLog, failed_log_dir
from llb.backends.prompt_window import PromptWindow
from llb.core.config import RunConfig
from llb.executor.runner_retrieval import _load_store
from llb.executor.runner_setup import _default_runner_fn
from llb.goldset.schema import GoldItem

if TYPE_CHECKING:
    from llb.core.contracts.hardware import ContentionReport

from llb.eval.graph_contracts import RagState

_LOG = logging.getLogger(__name__)


def _preserve_backend_log(launcher: BackendLauncher) -> None:
    """Keep the backend's server log out of the staging dir, which is about to be removed.

    Idempotent with the launcher's own failed-launch preservation (`ServerLog`): a launch that
    already copied its log on the way out of `start()` keeps that one copy, and this call covers
    the other failures -- a run that dies with the backend already serving.
    """
    if not isinstance(launcher, ServerLog):
        return
    kept = launcher.preserve_failed_log()
    if kept is not None:
        _LOG.error("[run-eval] backend startup log kept at %s", kept)


def _make_launcher(config: RunConfig, log_dir: Path | None = None) -> BackendLauncher:
    if config.adapter_path is not None and config.backend != "vllm":
        raise SystemExit(
            f"[run-eval] adapter serving is wired for vLLM LoRA modules; backend "
            f"{config.backend!r} needs a merged model artifact first"
        )
    if config.backend == "ollama":
        from llb.backends.ollama import OllamaLauncher

        return OllamaLauncher(
            config.model,
            host=config.ollama_host,
            num_ctx=config.max_model_len or config.context_budget,
        )
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
            suppress_thinking=config.vllm_suppress_thinking,
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


class VramContentionAbort(RuntimeError):
    """Even the derated gpu-memory-utilization cannot hold this model on the free VRAM.

    A RuntimeError rather than a `SystemExit` so a caller that runs many launches -- the public
    screen behind a confirmation run screens one finalist after another -- can record the failure
    and keep going. `run-eval`, which owns the process, still turns it into an exit.
    """


def guard_vllm_contention(
    config: RunConfig,
    launcher: BackendLauncher,
    *,
    evict: bool = False,
    wait: bool = False,
    label: str = "run-eval",
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
        raise VramContentionAbort(f"[{label}] pre-launch VRAM guard: {report['note']}")
    if report["derated"] and isinstance(launcher, VllmLauncher):
        _LOG.warning("[%s] %s", label, report["note"])
        launcher.gpu_memory_utilization = report["safe_util"]
        launcher.meta["gpu_memory_utilization"] = report["safe_util"]
    else:
        _LOG.info("[%s] pre-launch VRAM guard: %s", label, report["note"])
    return report


class ResolvedRunner(NamedTuple):
    """Everything `run_eval` needs to score cases, plus what the manifest records about the wiring.

    `context_window` is the run's usable prompt window -- what a document lane skips against and
    what the `rag` prompt is checked against. An INJECTED `runner_fn` leaves it None: the caller
    built its own graph, so this module never resolved a window for it.
    """

    launcher: BackendLauncher
    runner_fn: Callable[[GoldItem], RagState]
    store: Any
    contention: "ContentionReport | None"
    context_window: PromptWindow | None = None


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
) -> ResolvedRunner:
    contention: ContentionReport | None = None
    context_window: PromptWindow | None = None
    if launcher is None:
        launcher = _make_launcher(config, log_dir=staging_dir / "vllm")
        if config.backend == "vllm":
            try:
                contention = guard_vllm_contention(config, launcher, evict=evict, wait=wait)
            except VramContentionAbort as exc:
                raise SystemExit(str(exc)) from exc
    # The launcher writes its log inside the staging dir this run is about to delete on failure;
    # point its failed-launch copy at this config's data root before anything starts it.
    if isinstance(launcher, ServerLog) and launcher.failed_log_dir is None:
        launcher.failed_log_dir = failed_log_dir(config.data_dir)
    if runner_fn is None:
        if store is None:
            store = _load_store(config)
        runner_fn, context_window = _default_runner_fn(config, store, launcher, prompt_package)
    return ResolvedRunner(launcher, runner_fn, store, contention, context_window)


def _preserve_failed_staging(
    active_launcher: BackendLauncher | None,
    resume: Path | str | None,
    run_dir: Path,
    staging_dir: Path,
    *,
    interrupted: bool,
) -> None:
    """On failure: keep the backend log; keep staging only when it can seed a --resume."""
    if active_launcher is not None:
        _preserve_backend_log(active_launcher)
    if interrupted:
        _LOG.warning(
            "[run-eval] interrupted; staging preserved -- resume with --resume %s", run_dir
        )
    elif resume is None:
        shutil.rmtree(staging_dir, ignore_errors=True)
    else:
        _LOG.warning("[run-eval] resume failed; staging kept for another --resume %s", run_dir)
