"""Serve a registered adapter through the existing backend seam.

vLLM loads a LoRA adapter directly (`--enable-lora --lora-modules`). Ollama and llama.cpp cannot:
they serve whole model artifacts, so the adapter is first merged into its base weights and
converted to GGUF. That merge is expensive and one-way, so it is cached under
`$DATA_DIR/adapters/merged/<short-id>/<backend>/` and recorded as a registry `merge` event -- the
merged artifact stays traceable to the adapter digest that produced it.

The merge and the launcher are both injectable, so CI exercises all three backends without CUDA,
llama.cpp, or a running Ollama daemon.
"""

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from llb.backends.base import BackendLauncher
from llb.core.config import RunConfig
from llb.core.contracts import ChatMessage, JsonObject
from llb.core.fsutil import atomic_write_text
from llb.finetune.registry import (
    AdapterEntry,
    StalenessReport,
    load_registry,
    merged_root,
    record_merge,
    registry_path,
    resolve_adapter,
    staleness,
)

_LOG = logging.getLogger(__name__)

BACKEND_VLLM = "vllm"
BACKEND_OLLAMA = "ollama"
BACKEND_LLAMACPP = "llamacpp"
SERVING_BACKENDS = (BACKEND_VLLM, BACKEND_OLLAMA, BACKEND_LLAMACPP)

MERGE_MANIFEST = "merge.json"
MERGED_WEIGHTS_DIRNAME = "merged"
CONVERT_SCRIPT = "convert_hf_to_gguf.py"
GGUF_OUTTYPE = "f16"
OLLAMA_TAG_PREFIX = "llb-adapter-"
MODELFILE_NAME = "Modelfile"
MERGE_TOOL_PEFT_GGUF = "peft-merge+llama.cpp-convert"

# vLLM serves the LoRA module under this name, so chat requests address the adapter, not the base.
ADAPTER_LORA_NAME = "adapter"
# One tiny generation proves the served endpoint answers as the adapter before the operator commits.
PROBE_PROMPT = "Reply with OK."
PROBE_MAX_TOKENS = 16
HOLD_POLL_S = 1.0


@dataclass(frozen=True)
class MergeRequest:
    entry: AdapterEntry
    backend: str
    out_dir: Path
    data_dir: Path


@dataclass(frozen=True)
class MergeArtifacts:
    merged_dir: Path
    gguf_path: Path | None = None
    model_tag: str | None = None
    tool: str = MERGE_TOOL_PEFT_GGUF

    def as_dict(self) -> JsonObject:
        return {
            "merged_dir": str(self.merged_dir),
            "gguf_path": str(self.gguf_path) if self.gguf_path else None,
            "model_tag": self.model_tag,
            "tool": self.tool,
        }

    @classmethod
    def from_dict(cls, payload: JsonObject) -> "MergeArtifacts":
        gguf = payload.get("gguf_path")
        return cls(
            merged_dir=Path(str(payload["merged_dir"])),
            gguf_path=Path(str(gguf)) if gguf else None,
            model_tag=payload.get("model_tag"),
            tool=str(payload.get("tool") or MERGE_TOOL_PEFT_GGUF),
        )


@dataclass(frozen=True)
class ServePlan:
    """What the launcher must load: a base model plus LoRA, or a merged artifact."""

    entry: AdapterEntry
    backend: str
    served_model: str
    adapter_path: Path | None = None
    merged: MergeArtifacts | None = None


@dataclass(frozen=True)
class ServeResult:
    adapter_id: str
    base_model: str
    backend: str
    served_model: str
    request_model: str
    endpoint: str
    staleness: StalenessReport
    merged: MergeArtifacts | None = None
    probe_text: str | None = None
    probe_error: str | None = None


MergeFn = Callable[[MergeRequest], MergeArtifacts]
LauncherFn = Callable[[ServePlan, RunConfig], BackendLauncher]
ReadyFn = Callable[["ServeResult"], None]


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
    """Resolve a registered adapter, serve it on `backend`, and smoke it with one generation.

    `on_ready` fires once the probe has answered and BEFORE `hold` blocks, so a caller can report
    the live endpoint while the backend is still up. A failed probe never holds: there is nothing
    to serve.
    """
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
            probe_error=probe.error,
        )
        if on_ready is not None:
            on_ready(result)
        if hold and probe.error is None:
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
    """vLLM serves the adapter directly; the GGUF backends serve a cached merge of it."""
    if backend == BACKEND_VLLM:
        return ServePlan(entry, backend, entry.base_model, adapter_path=entry.resolved_dir)
    merged = ensure_merged(
        entry, backend=backend, data_dir=config.data_dir, registry=registry, merge_fn=merge_fn
    )
    if backend == BACKEND_OLLAMA:
        if not merged.model_tag:
            raise SystemExit("[serve-adapter] the merge produced no Ollama model tag")
        return ServePlan(entry, backend, merged.model_tag, merged=merged)
    if not merged.gguf_path:
        raise SystemExit("[serve-adapter] the merge produced no GGUF artifact for llama.cpp")
    return ServePlan(entry, backend, str(merged.gguf_path), merged=merged)


def ensure_merged(
    entry: AdapterEntry,
    *,
    backend: str,
    data_dir: Path | str,
    registry: Path | str,
    merge_fn: MergeFn | None = None,
) -> MergeArtifacts:
    """Reuse a cached merge for this (adapter, backend), else merge once and record the event."""
    out_dir = merged_root(data_dir) / entry.short_id / backend
    manifest = out_dir / MERGE_MANIFEST
    if manifest.is_file():
        return MergeArtifacts.from_dict(json.loads(manifest.read_text(encoding="utf-8")))
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = (merge_fn or merge_adapter)(
        MergeRequest(entry=entry, backend=backend, out_dir=out_dir, data_dir=Path(data_dir))
    )
    atomic_write_text(
        manifest, json.dumps(artifacts.as_dict(), ensure_ascii=False, indent=2) + "\n"
    )
    record_merge(
        registry=registry,
        adapter_id=entry.adapter_id,
        backend=backend,
        artifacts=artifacts.as_dict(),
    )
    return artifacts


def merge_adapter(request: MergeRequest) -> MergeArtifacts:
    """Merge LoRA weights into the base model, convert to GGUF, and register an Ollama tag."""
    merged_dir = request.out_dir / MERGED_WEIGHTS_DIRNAME
    _merge_lora_weights(request.entry, merged_dir)
    gguf_path = _convert_to_gguf(merged_dir, request.out_dir, request.entry, request.data_dir)
    model_tag = None
    if request.backend == BACKEND_OLLAMA:
        model_tag = _ollama_create(gguf_path, request.out_dir, request.entry)
    return MergeArtifacts(merged_dir, gguf_path, model_tag, MERGE_TOOL_PEFT_GGUF)


def default_launcher(plan: ServePlan, config: RunConfig) -> BackendLauncher:
    """Build the real launcher for a serve plan, reusing the run-eval backend wiring."""
    if plan.backend == BACKEND_VLLM:
        from llb.backends.vllm import VllmLauncher

        return VllmLauncher(
            plan.served_model,
            host=config.vllm_host,
            port=config.vllm_port,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            cpu_offload_gb=config.cpu_offload_gb,
            kv_offloading_size_gb=config.kv_offloading_size_gb,
            dtype=config.dtype,
            quantization=config.quantization,
            adapter_path=plan.adapter_path,
            adapter_name=ADAPTER_LORA_NAME,
        )
    if plan.backend == BACKEND_OLLAMA:
        from llb.backends.ollama import OllamaLauncher

        return OllamaLauncher(plan.served_model, host=config.ollama_host)
    from llb.backends.llamacpp import LlamaCppLauncher, resolve_llama_server_binary

    return LlamaCppLauncher(
        plan.served_model,
        host=config.llamacpp_host,
        n_gpu_layers=config.n_gpu_layers,
        ctx_size=config.max_model_len,
        binary=resolve_llama_server_binary(config.data_dir),
    )


def backend_endpoint(backend: str, config: RunConfig) -> str:
    if backend == BACKEND_VLLM:
        return config.vllm_host
    if backend == BACKEND_OLLAMA:
        return config.ollama_host
    return config.llamacpp_host


def ollama_tag(entry: AdapterEntry) -> str:
    """Ollama tags are lowercase; a sha256 prefix already is."""
    return f"{OLLAMA_TAG_PREFIX}{entry.short_id}"


def _probe_message() -> ChatMessage:
    return {"role": "user", "content": PROBE_PROMPT}


def _hold_until_interrupt(endpoint: str, request_model: str) -> None:
    """Foreground serving: hold the launcher open until Ctrl-C (never a background daemon)."""
    _LOG.info("[serve-adapter] serving %s at %s -- Ctrl-C to stop", request_model, endpoint)
    try:
        while True:
            time.sleep(HOLD_POLL_S)
    except KeyboardInterrupt:
        _LOG.info("[serve-adapter] interrupted; stopping backend")


def _merge_lora_weights(entry: AdapterEntry, merged_dir: Path) -> None:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "[serve-adapter] merging needs the finetune extra on the CUDA host: "
            'uv pip install -e ".[finetune]"'
        ) from exc
    base = AutoModelForCausalLM.from_pretrained(entry.base_model, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, str(entry.resolved_dir)).merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(entry.base_model, trust_remote_code=True).save_pretrained(
        merged_dir
    )


def _convert_to_gguf(merged_dir: Path, out_dir: Path, entry: AdapterEntry, data_dir: Path) -> Path:
    script = data_dir / "llb" / "llamacpp" / "src" / CONVERT_SCRIPT
    if not script.is_file():
        raise SystemExit(
            f"[serve-adapter] {CONVERT_SCRIPT} not found at {script}; run `make build-llamacpp` "
            "so the llama.cpp checkout exists"
        )
    gguf_path = out_dir / f"{OLLAMA_TAG_PREFIX}{entry.short_id}.gguf"
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(merged_dir),
            "--outfile",
            str(gguf_path),
            "--outtype",
            GGUF_OUTTYPE,
        ],
        check=True,
    )
    return gguf_path


def _ollama_create(gguf_path: Path, out_dir: Path, entry: AdapterEntry) -> str:
    tag = ollama_tag(entry)
    modelfile = out_dir / MODELFILE_NAME
    atomic_write_text(modelfile, f"FROM {gguf_path}\n")
    subprocess.run(["ollama", "create", tag, "-f", str(modelfile)], check=True)
    return tag
