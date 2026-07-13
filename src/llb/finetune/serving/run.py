"""Adapter serve-plan construction and launcher lifecycle."""

import logging
import time
from pathlib import Path

from llb.core.config import RunConfig
from llb.core.contracts import ChatMessage
from llb.finetune.registry.io import load_registry, registry_path
from llb.finetune.registry.model import AdapterEntry
from llb.finetune.registry.resolve import resolve_adapter
from llb.finetune.registry.staleness import staleness
from llb.finetune.serving.launcher import backend_endpoint, default_launcher
from llb.finetune.serving.merge import ensure_merged
from llb.finetune.serving.model import (
    BACKEND_LLAMACPP,
    BACKEND_OLLAMA,
    BACKEND_VLLM,
    HOLD_POLL_S,
    PROBE_MAX_TOKENS,
    PROBE_PROMPT,
    SERVING_BACKENDS,
    LauncherFn,
    MergeFn,
    ReadyFn,
    ServePlan,
    ServeResult,
)

_LOG = logging.getLogger(__name__)


def serve_adapter(
    config: RunConfig,
    *,
    adapter: str,
    backend: str | None = None,
    registry: Path | str | None = None,
    merge_fn: MergeFn | None = None,
    launcher_factory: LauncherFn | None = None,
    hold: bool = False,
    on_ready: ReadyFn | None = None,
) -> ServeResult:
    """Resolve, serve, probe, and release a registered adapter."""
    target = backend or config.backend
    if target not in SERVING_BACKENDS:
        raise SystemExit(
            f"[serve-adapter] backend {target!r} is not wired ({', '.join(SERVING_BACKENDS)})"
        )
    registry_file = Path(registry) if registry is not None else registry_path(config.data_dir)
    entry = resolve_adapter(load_registry(registry_file), adapter)
    report = staleness(entry)
    if report.is_stale:
        _LOG.warning("[serve-adapter] %s is stale -- %s", entry.short_id, report.describe())
    plan = build_serve_plan(
        entry, backend=target, config=config, registry=registry_file, merge_fn=merge_fn
    )
    launcher = (launcher_factory or default_launcher)(plan, config)
    request_model = str(getattr(launcher, "request_model", plan.served_model))
    endpoint = backend_endpoint(target, config)
    launcher.start()
    try:
        probe = launcher.chat(
            [_probe_message()],
            max_tokens=PROBE_MAX_TOKENS,
            temperature=0.0,
            timeout=config.request_timeout_s,
        )
        probe_error = probe.error
        if probe_error is None and not (probe.text or "").strip():
            probe_error = "probe returned an empty completion"
        result = ServeResult(
            adapter_id=entry.adapter_id,
            base_model=entry.base_model,
            backend=target,
            served_model=plan.served_model,
            request_model=request_model,
            endpoint=endpoint,
            staleness=report,
            merged=plan.merged,
            probe_text=probe.text,
            probe_error=probe_error,
        )
        if on_ready is not None:
            on_ready(result)
        if hold and probe_error is None:
            _hold_until_interrupt(endpoint, request_model)
    finally:
        launcher.stop()
    return result


def build_serve_plan(
    entry: AdapterEntry,
    *,
    backend: str,
    config: RunConfig,
    registry: Path | str,
    merge_fn: MergeFn | None = None,
) -> ServePlan:
    """Choose direct vLLM LoRA loading or a cached merged backend artifact."""
    if backend == BACKEND_VLLM:
        return ServePlan(entry, backend, entry.base_model, adapter_path=entry.resolved_dir)
    merged = ensure_merged(
        entry, backend=backend, data_dir=config.data_dir, registry=registry, merge_fn=merge_fn
    )
    if backend == BACKEND_OLLAMA:
        if not merged.model_tag:
            raise SystemExit("[serve-adapter] the merge produced no Ollama model tag")
        return ServePlan(entry, backend, merged.model_tag, merged=merged)
    if backend != BACKEND_LLAMACPP or not merged.gguf_path:
        raise SystemExit("[serve-adapter] the merge produced no GGUF artifact for llama.cpp")
    return ServePlan(entry, backend, str(merged.gguf_path), merged=merged)


def _probe_message() -> ChatMessage:
    return {"role": "user", "content": PROBE_PROMPT}


def _hold_until_interrupt(endpoint: str, request_model: str) -> None:
    _LOG.info("[serve-adapter] serving %s at %s -- Ctrl-C to stop", request_model, endpoint)
    try:
        while True:
            time.sleep(HOLD_POLL_S)
    except KeyboardInterrupt:
        _LOG.info("[serve-adapter] interrupted; stopping backend")
