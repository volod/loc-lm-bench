"""Prepare candidate models for the local backends.

Two backends, two stores:
  - ollama: Ollama manages its own model store, so we shell out to `ollama pull <tag>`.
  - vllm:   vLLM loads HF weights from the standard Hugging Face cache, so we snapshot-
            download each repo ONCE (via the base `huggingface_hub` dep -- no torch/vLLM
            needed just to cache). A later vLLM launch reuses the cached snapshot.

The host GPU is detected first; oversized models are skipped (vLLM) or flagged
(Ollama, which can offload to CPU). The plan/decision logic is pure and unit-testable;
the side-effecting `ollama_pull` / `hf_cache` are injectable.

Manifest entry (YAML, see `samples/models_uk.yaml`):
  - name: <label>
    backend: ollama | vllm
    source: <ollama-tag> | <hf-repo-id>
    min_vram_gb: <int>      # rough floor to serve it on this hardware class
    notes: <free text>
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, cast

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from llb.backends.hardware import Gpu, detect_gpus, max_vram_mb
from llb import env
from llb.contracts import ModelSpec, PreparationReport, PreparedModel

ACTION_PULL = "pull"  # ollama pull
ACTION_CACHE = "cache"  # hf snapshot download for vLLM
ACTION_SKIP = "skip"
SUPPORTED_BACKENDS = ("ollama", "vllm")
OLLAMA_PULL_TIMEOUT_ENV = "LLB_OLLAMA_PULL_TIMEOUT_S"
DEFAULT_OLLAMA_PULL_TIMEOUT_S = 1800
MAX_ERROR_DETAIL_CHARS = 400

OllamaPull = Callable[[str], tuple[bool, str]]
HfCache = Callable[[str, str | None, Path | None], tuple[bool, str]]
PrepareProgress = Callable[[PreparedModel], None]


class _ModelSpecSchema(BaseModel):
    """Validation model for one external candidate-manifest entry."""

    model_config = ConfigDict(extra="forbid")

    name: str
    backend: str
    source: str
    min_vram_gb: int | float = 0
    notes: str | None = None
    license_url: str | None = None
    gated: bool = False
    params_b: float | None = None
    quant: str | None = None
    bpw: float | None = None
    n_layers: int | None = None
    kv_dim: int | None = None
    max_context: int | None = None
    vocab_size: int | None = None
    hidden_size: int | None = None
    tie_word_embeddings: bool | None = None
    embed_bpw: float | None = None
    hi_precision_params_b: float | None = None
    sources: dict[str, "str | dict[str, object]"] | None = None


def load_manifest(path: Path | str) -> list[ModelSpec]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML -- {exc}") from None
    models = data.get("models") if isinstance(data, dict) else None
    if not models:
        raise ValueError(f"{path}: expected a top-level 'models:' list")
    for model in models:
        if not isinstance(model, dict):
            raise ValueError(f"{path}: each model entry must be a mapping, got: {model!r}")
    try:
        validated = [_ModelSpecSchema.model_validate(model) for model in models]
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid model entry -- {exc}") from None
    return [cast(ModelSpec, model.model_dump(exclude_none=True)) for model in validated]


def load_serving_targets(path: Path | str) -> list[ModelSpec]:
    """Read a generated serving tier.json as concrete preparation targets."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON -- {exc}") from None
    targets = data.get("targets") if isinstance(data, dict) else None
    if not isinstance(targets, list):
        raise ValueError(f"{path}: expected a top-level 'targets' list")

    models: list[ModelSpec] = []
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError(f"{path}: each target entry must be a mapping, got: {target!r}")
        target_id = target.get("target")
        backend = target.get("backend")
        source = target.get("model")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError(f"{path}: serving target is missing a non-empty target id")
        if not isinstance(backend, str) or not isinstance(source, str) or not source:
            raise ValueError(f"{path}: target {target_id!r} must include backend and model")
        models.append(
            {
                "name": f"serving-{target_id}",
                "backend": backend,
                "source": source,
                "min_vram_gb": 0,
                "notes": "generated serving-tier target",
            }
        )
    return models


def decide(backend: str, need_mb: int, max_mb: int, has_gpu: bool, force: bool) -> tuple[str, str]:
    """Per-model action + reason given the detected hardware."""
    if backend == "ollama":
        if need_mb > max_mb and not force:
            return ACTION_PULL, f"larger than {max_mb} MB VRAM; Ollama offloads to CPU (slow)"
        return ACTION_PULL, "ok"
    if backend == "vllm":
        if not has_gpu and not force:
            return ACTION_SKIP, "vLLM needs a CUDA GPU; none detected (use --force to cache anyway)"
        if need_mb > max_mb and not force:
            return ACTION_SKIP, f"needs ~{need_mb} MB VRAM, host has {max_mb} MB (use --force)"
        return ACTION_CACHE, "ok"
    return ACTION_SKIP, f"unknown backend '{backend}'"


def plan(
    models: list[ModelSpec],
    max_mb: int,
    has_gpu: bool,
    backend_filter: str,
    force: bool,
) -> list[PreparedModel]:
    """Annotate each in-scope model with an action + reason (no side effects)."""
    rows: list[PreparedModel] = []
    for m in _expand_prepare_sources(models):
        backend = m["backend"]
        if backend_filter != "all" and backend != backend_filter:
            continue
        if backend not in SUPPORTED_BACKENDS:
            continue
        need_mb = int(m.get("min_vram_gb", 0)) * 1024
        action, reason = decide(backend, need_mb, max_mb, has_gpu, force)
        rows.append({**m, "action": action, "reason": reason})
    return rows


def _normalize_source_record(value: str | dict[str, object]) -> dict[str, object]:
    if isinstance(value, str):
        return {"source": value}
    return {k: v for k, v in value.items() if v is not None}


def _expand_prepare_sources(models: list[ModelSpec]) -> list[ModelSpec]:
    """Expand a logical model into concrete backend artifacts that can be prepared.

    The resolver already understands per-backend `sources:` records. Model preparation needs the
    same expansion so a 16 GB host pulls Ollama GGUF fallbacks such as MamayLM/Lapa while also
    caching vLLM Hugging Face weights that fit the GPU.
    """
    expanded: list[ModelSpec] = []
    for model in models:
        records: dict[str, dict[str, object]] = {
            backend: _normalize_source_record(source)
            for backend, source in (model.get("sources") or {}).items()
        }
        records.setdefault(model["backend"], {"source": model["source"]})

        for backend, record in records.items():
            if backend not in SUPPORTED_BACKENDS:
                continue
            source = record.get("source")
            if not isinstance(source, str) or not source:
                continue
            row = {**model, **record, "backend": backend, "source": source}
            if backend != model["backend"] or source != model["source"]:
                row["name"] = f"{model['name']}-{backend}"
            expanded.append(cast(ModelSpec, row))
    return expanded


def acceptance_url(spec: ModelSpec | PreparedModel) -> str | None:
    """Where to accept a gated model's license. Explicit `license_url`, else derived from the
    HF repo id when `gated: true`. None for ungated / non-HF entries."""
    if spec.get("license_url"):
        return str(spec["license_url"])
    if spec.get("gated") and spec.get("backend") == "vllm":
        return f"https://huggingface.co/{spec['source']}"
    return None


def _looks_gated(exc: Exception) -> bool:
    """True if a download error is an access gate (license not accepted / no token)."""
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(
        s in blob
        for s in (
            "gated",
            "401",
            "403",
            "awaiting",
            "must be authenticated",
            "access to model",
            "accept the conditions",
        )
    )


def _ollama_timeout_s() -> int | None:
    raw = os.environ.get(OLLAMA_PULL_TIMEOUT_ENV)
    if raw is None or raw == "":
        return DEFAULT_OLLAMA_PULL_TIMEOUT_S
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_OLLAMA_PULL_TIMEOUT_S
    return value if value > 0 else None


def _ascii_detail(value: str) -> str:
    """Return a compact ASCII-only diagnostic string for captured tool output."""
    clean = value.encode("ascii", "ignore").decode("ascii")
    clean = " ".join(clean.split())
    if len(clean) > MAX_ERROR_DETAIL_CHARS:
        return "..." + clean[-MAX_ERROR_DETAIL_CHARS:]
    return clean


def _ollama_pull(source: str) -> tuple[bool, str]:
    if shutil.which("ollama") is None:
        return False, "ollama CLI not found (install Ollama and run `ollama serve`)"
    timeout_s = _ollama_timeout_s()
    try:
        out = subprocess.run(
            ["ollama", "pull", source],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        detail = f"ollama pull timed out after {timeout_s}s"
        return False, detail
    except subprocess.SubprocessError as exc:
        return False, f"ollama pull failed: {exc}"
    if out.returncode == 0:
        return True, "success"
    detail = _ascii_detail(f"{out.stderr}\n{out.stdout}")
    suffix = f": {detail}" if detail else ""
    return False, f"ollama pull exited with code {out.returncode}{suffix}"


def _hf_cache(source: str, token: str | None, cache_dir: Path | None) -> tuple[bool, str]:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import (
            are_progress_bars_disabled,
            disable_progress_bars,
            enable_progress_bars,
        )
    except ImportError:
        return False, "huggingface_hub missing (it is a base dep; reinstall with `make venv`)"
    try:
        progress_was_disabled = are_progress_bars_disabled()
        disable_progress_bars()
        path = snapshot_download(
            repo_id=source,
            token=token or os.environ.get(env.HF_TOKEN),
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    except Exception as exc:  # 404 / auth / network -- report per-model, keep going
        msg = f"{type(exc).__name__}: {exc}"
        if _looks_gated(exc):
            msg += (
                f" -- accept the license at https://huggingface.co/{source} "
                f"and set {env.HF_TOKEN} in .env"
            )
        return False, msg
    finally:
        if "progress_was_disabled" in locals() and not progress_was_disabled:
            enable_progress_bars()
    return True, path


def _prepare_row_status(
    row: PreparedModel,
    *,
    dry_run: bool,
    ollama_pull: OllamaPull,
    hf_cache: HfCache,
    token: str | None,
    cache_dir: Path | None,
) -> tuple[str, str]:
    """Run or plan one manifest row; return (status, detail)."""
    if dry_run or row["action"] == ACTION_SKIP:
        status = (
            "planned" if dry_run else ("skipped" if row["action"] == ACTION_SKIP else "planned")
        )
        return status, row["reason"]
    if row["action"] == ACTION_PULL:
        ok, detail = ollama_pull(row["source"])
        return ("done" if ok else "failed"), detail
    ok, detail = hf_cache(row["source"], token, cache_dir)
    return ("done" if ok else "failed"), detail


def prepare_models(
    models: list[ModelSpec],
    *,
    backend_filter: str = "all",
    force: bool = False,
    dry_run: bool = False,
    token: str | None = None,
    cache_dir: Path | None = None,
    gpus: list[Gpu] | None = None,
    ollama_pull: OllamaPull | None = None,
    hf_cache: HfCache | None = None,
    progress: PrepareProgress | None = None,
) -> PreparationReport:
    """Execute (or, with dry_run, just plan) model preparation. Returns a report dict."""
    gpus = detect_gpus() if gpus is None else gpus
    max_mb = max_vram_mb(gpus)
    rows = plan(models, max_mb, bool(gpus), backend_filter, force)
    ollama_pull = ollama_pull or _ollama_pull
    hf_cache = hf_cache or _hf_cache

    results: list[PreparedModel] = []
    for row in rows:
        if progress is not None:
            progress(row)
        status, detail = _prepare_row_status(
            row,
            dry_run=dry_run,
            ollama_pull=ollama_pull,
            hf_cache=hf_cache,
            token=token,
            cache_dir=cache_dir,
        )
        url = acceptance_url(row)
        if url and "huggingface.co" not in detail:
            detail = f"{detail}  [license: {url}]"
        results.append({**row, "status": status, "detail": detail})
    return {"gpus": gpus, "max_vram_mb": max_mb, "results": results}
