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

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, cast

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from llb.backends.hardware import Gpu, detect_gpus, max_vram_mb
from llb.contracts import ModelSpec, PreparationReport, PreparedModel

ACTION_PULL = "pull"  # ollama pull
ACTION_CACHE = "cache"  # hf snapshot download for vLLM
ACTION_SKIP = "skip"

OllamaPull = Callable[[str], tuple[bool, str]]
HfCache = Callable[[str, str | None, Path | None], tuple[bool, str]]


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
    for m in models:
        backend = m["backend"]
        if backend_filter != "all" and backend != backend_filter:
            continue
        need_mb = int(m.get("min_vram_gb", 0)) * 1024
        action, reason = decide(backend, need_mb, max_mb, has_gpu, force)
        rows.append({**m, "action": action, "reason": reason})
    return rows


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


def _ollama_pull(source: str) -> tuple[bool, str]:
    if shutil.which("ollama") is None:
        return False, "ollama CLI not found (install Ollama and run `ollama serve`)"
    try:
        out = subprocess.run(["ollama", "pull", source], capture_output=True, text=True)
    except subprocess.SubprocessError as exc:
        return False, f"ollama pull failed: {exc}"
    detail = (out.stderr.strip() or "pulled").splitlines()[-1].strip()
    return (out.returncode == 0), detail


def _hf_cache(source: str, token: str | None, cache_dir: Path | None) -> tuple[bool, str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False, "huggingface_hub missing (it is a base dep; reinstall with `make venv`)"
    try:
        path = snapshot_download(
            repo_id=source,
            token=token or os.environ.get("HF_TOKEN"),
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    except Exception as exc:  # 404 / auth / network -- report per-model, keep going
        msg = f"{type(exc).__name__}: {exc}"
        if _looks_gated(exc):
            msg += (
                f" -- accept the license at https://huggingface.co/{source} "
                "and set HF_TOKEN in .env"
            )
        return False, msg
    return True, path


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
) -> PreparationReport:
    """Execute (or, with dry_run, just plan) model preparation. Returns a report dict."""
    gpus = detect_gpus() if gpus is None else gpus
    max_mb = max_vram_mb(gpus)
    rows = plan(models, max_mb, bool(gpus), backend_filter, force)
    ollama_pull = ollama_pull or _ollama_pull
    hf_cache = hf_cache or _hf_cache

    results: list[PreparedModel] = []
    for row in rows:
        if dry_run or row["action"] == ACTION_SKIP:
            status = (
                "planned" if dry_run else ("skipped" if row["action"] == ACTION_SKIP else "planned")
            )
            detail = row["reason"]
        elif row["action"] == ACTION_PULL:
            ok, detail = ollama_pull(row["source"])
            status = "done" if ok else "failed"
        else:  # ACTION_CACHE
            ok, detail = hf_cache(row["source"], token, cache_dir)
            status = "done" if ok else "failed"
        url = acceptance_url(row)
        if url and "huggingface.co" not in detail:
            detail = f"{detail}  [license: {url}]"
        results.append({**row, "status": status, "detail": detail})
    return {"gpus": gpus, "max_vram_mb": max_mb, "results": results}
