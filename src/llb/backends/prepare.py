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

import yaml

from llb.backends.hardware import Gpu, detect_gpus, max_vram_mb

ACTION_PULL = "pull"      # ollama pull
ACTION_CACHE = "cache"    # hf snapshot download for vLLM
ACTION_SKIP = "skip"


def load_manifest(path: Path | str) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models = data.get("models") if isinstance(data, dict) else None
    if not models:
        raise ValueError(f"{path}: expected a top-level 'models:' list")
    for m in models:
        for key in ("name", "backend", "source"):
            if key not in m:
                raise ValueError(f"{path}: model entry missing '{key}': {m}")
    return models


def decide(backend: str, need_mb: int, max_mb: int, has_gpu: bool,
           force: bool) -> tuple[str, str]:
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


def plan(models: list[dict], max_mb: int, has_gpu: bool, backend_filter: str,
         force: bool) -> list[dict]:
    """Annotate each in-scope model with an action + reason (no side effects)."""
    rows: list[dict] = []
    for m in models:
        backend = m["backend"]
        if backend_filter != "all" and backend != backend_filter:
            continue
        need_mb = int(m.get("min_vram_gb", 0)) * 1024
        action, reason = decide(backend, need_mb, max_mb, has_gpu, force)
        rows.append({**m, "action": action, "reason": reason})
    return rows


def _ollama_pull(source: str) -> tuple[bool, str]:
    if shutil.which("ollama") is None:
        return False, "ollama CLI not found (install Ollama and run `ollama serve`)"
    try:
        out = subprocess.run(["ollama", "pull", source], capture_output=True, text=True)
    except subprocess.SubprocessError as exc:
        return False, f"ollama pull failed: {exc}"
    return (out.returncode == 0), (out.stderr.strip() or "pulled").splitlines()[-1]


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
        return False, f"{type(exc).__name__}: {exc}"
    return True, path


def prepare_models(
    models: list[dict],
    *,
    backend_filter: str = "all",
    force: bool = False,
    dry_run: bool = False,
    token: str | None = None,
    cache_dir: Path | None = None,
    gpus: list[Gpu] | None = None,
    ollama_pull=None,
    hf_cache=None,
) -> dict:
    """Execute (or, with dry_run, just plan) model preparation. Returns a report dict."""
    gpus = detect_gpus() if gpus is None else gpus
    max_mb = max_vram_mb(gpus)
    rows = plan(models, max_mb, bool(gpus), backend_filter, force)
    ollama_pull = ollama_pull or _ollama_pull
    hf_cache = hf_cache or _hf_cache

    results: list[dict] = []
    for row in rows:
        if dry_run or row["action"] == ACTION_SKIP:
            status = "planned" if dry_run else ("skipped" if row["action"] == ACTION_SKIP else "planned")
            results.append({**row, "status": status, "detail": row["reason"]})
            continue
        if row["action"] == ACTION_PULL:
            ok, detail = ollama_pull(row["source"])
        else:  # ACTION_CACHE
            ok, detail = hf_cache(row["source"], token, cache_dir)
        results.append({**row, "status": "done" if ok else "failed", "detail": detail})
    return {"gpus": gpus, "max_vram_mb": max_mb, "results": results}
