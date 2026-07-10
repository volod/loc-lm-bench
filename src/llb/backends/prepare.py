"""Prepare candidate models for the local backends.

Two backends, two stores:
  - ollama: Ollama manages its own model store, so we shell out to `ollama pull <tag>`.
  - vllm:   vLLM loads HF weights from the standard Hugging Face cache, so we snapshot-
            download each repo ONCE (via the base `huggingface_hub` dep -- no torch/vLLM
            needed just to cache). A later vLLM launch reuses the cached snapshot.

The host GPU is detected first; oversized models are skipped (vLLM) or flagged
(Ollama, which can offload to CPU). The plan/decision logic is pure and unit-testable;
the side-effecting `ollama_pull` / `hf_cache` are injectable.

Manifest entry (YAML, see `samples/configs/models_uk.yaml`):
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

from llb.backends import hardware, planner
from llb.backends.hardware import Gpu, detect_gpus, max_vram_mb
from llb.core import env
from llb.core.config import DEFAULT_OLLAMA_HOST
from llb.core.contracts import ModelSpec, PreparationReport, PreparedModel, SourceRecord

ACTION_PULL = "pull"  # ollama pull
ACTION_CACHE = "cache"  # hf snapshot download for vLLM
ACTION_SKIP = "skip"
SUPPORTED_BACKENDS = ("ollama", "vllm")
OLLAMA_PULL_TIMEOUT_ENV = "LLB_OLLAMA_PULL_TIMEOUT_S"
DEFAULT_OLLAMA_PULL_TIMEOUT_S = 1800
MAX_ERROR_DETAIL_CHARS = 400

# Disk preflight: refuse a long download up front when the destination filesystem is too small,
# so a multi-GiB pull does not fail an hour in. It is REUSE-AWARE -- an artifact already in its
# backend store skips the check, since re-running prepare reuses the cache and fetches ~nothing.
DOWNLOAD_SAFETY_FACTOR = 1.15  # tokenizer + safetensors index + packaging beyond the raw weights
DOWNLOAD_HEADROOM_MB = 2048  # leave the store filesystem some slack rather than filling it
MIN_DOWNLOAD_MB = 512  # floor for tiny / unsized artifacts so they never block on a bogus estimate

OllamaPull = Callable[[str], tuple[bool, str]]
HfCache = Callable[[str, str | None, Path | None], tuple[bool, str]]
PrepareProgress = Callable[[PreparedModel], None]
DiskFreeReader = Callable[[Path], int]  # store path -> free MB (0 == unknown, never blocks)
PresentCheck = Callable[[ModelSpec], bool]  # is this artifact already cached in its backend store?


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
    sources: dict[str, "str | dict[str, object] | list[str | dict[str, object]]"] | None = None


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


def estimate_download_mb(spec: ModelSpec) -> int:
    """Best-effort size (MB) of the artifact `prepare` would fetch for this spec.

    Prices the weights with the embedding-aware planner (so an fp8 checkpoint whose big vocab
    embedding stays bf16 is not under-counted) plus a safety factor for the tokenizer, index, and
    packaging files. Falls back to `min_vram_gb`, then a small floor, when the spec carries no
    size hints -- an unsized entry never blocks on a bogus estimate.
    """
    bpw = planner.resolve_bpw(spec)
    params_b = spec.get("params_b")
    if bpw is not None and params_b is not None:
        weights = planner.weights_mib_detailed(
            float(params_b), bpw, planner.hi_precision_params(spec)
        )
        return max(MIN_DOWNLOAD_MB, int(weights * DOWNLOAD_SAFETY_FACTOR))
    floor = int(spec.get("min_vram_gb", 0)) * 1024
    return max(MIN_DOWNLOAD_MB, floor)


# Ollama's blob store moves with the install: the user default, the systemd-package location, and
# the OLLAMA_MODELS override. Both the reuse check and the free-space probe must look where the
# running daemon actually writes, or a system-service install (store under /usr/share/ollama) makes
# every model look un-cached and its filesystem un-probed.
_OLLAMA_STORE_CANDIDATES = (
    Path.home() / ".ollama" / "models",
    Path("/usr/share/ollama/.ollama/models"),
    Path("/var/lib/ollama/models"),
)


def _ollama_candidate_stores() -> list[Path]:
    override = os.environ.get("OLLAMA_MODELS")
    if override:
        return [Path(override).expanduser()]
    return list(_OLLAMA_STORE_CANDIDATES)


def _ollama_store_dir() -> Path:
    """The Ollama blob store a pull lands in: the first candidate that already exists, else the
    user-home default (created on first pull). Honors OLLAMA_MODELS over the install locations."""
    candidates = _ollama_candidate_stores()
    return next((c for c in candidates if c.exists()), candidates[0])


def _hf_cache_dir(cache_dir: Path | None) -> Path:
    if cache_dir is not None:
        return cache_dir
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return Path(hub).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def store_dir_for(backend: str, cache_dir: Path | None) -> Path:
    """The on-disk store a prepared artifact lands in: Ollama's blob store or the HF hub cache."""
    return _ollama_store_dir() if backend == "ollama" else _hf_cache_dir(cache_dir)


def _ollama_manifest_exists(store: Path, source: str) -> bool:
    """Offline fallback: is an Ollama tag already pulled into this blob store? (reuse signal)."""
    manifests = store / "manifests"
    if not manifests.is_dir():
        return False
    repo, _, tag = source.partition(":")
    tag = (tag or "latest").lower()
    leaf = repo.rstrip("/").split("/")[-1].lower()
    try:
        for hit in manifests.rglob("*"):
            if hit.is_file() and hit.name.lower() == tag and leaf in hit.parent.as_posix().lower():
                return True
    except OSError:
        return False
    return False


def _ollama_present(source: str) -> bool:
    """Is this Ollama tag already available, so a pull fetches ~nothing? (reuse signal).

    Authoritative path: ask the RUNNING daemon what it serves (the same `/api/tags` probe the
    resolver uses), so a tag pulled into any store the daemon is configured with counts -- not just
    the on-disk locations we guess. Falls back to scanning the candidate blob stores when the daemon
    is unreachable (offline), so the check never blocks a re-pull of an already-cached tag.
    """
    from llb.backends.resolver import _make_ollama_probe

    host = os.environ.get(env.OLLAMA_HOST) or DEFAULT_OLLAMA_HOST
    try:
        if _make_ollama_probe(host)(source):
            return True
    except Exception:
        pass
    return any(_ollama_manifest_exists(store, source) for store in _ollama_candidate_stores())


def _default_present_check(spec: ModelSpec) -> bool:
    """Reuse signal: is this artifact already in its backend store, so prepare fetches ~nothing?"""
    backend = spec.get("backend")
    source = str(spec.get("source", ""))
    if backend == "vllm":
        return planner.cached_config_path(source) is not None
    if backend == "ollama":
        return _ollama_present(source)
    return False


def disk_precheck(
    required_mb: int, free_mb: int, headroom_mb: int = DOWNLOAD_HEADROOM_MB
) -> tuple[bool, str]:
    """(ok, reason) for a download of `required_mb` onto a store with `free_mb` free.

    `free_mb <= 0` means the free space is UNKNOWN (probe failed), which never blocks -- the
    check only refuses when it can prove the filesystem is too small.
    """
    if free_mb <= 0:
        return True, ""
    need = required_mb + headroom_mb
    if free_mb >= need:
        return True, ""
    return False, (
        f"insufficient disk: need ~{need} MB (artifact ~{required_mb} MB + {headroom_mb} MB "
        f"headroom), only {free_mb} MB free -- free space or move the store before retrying"
    )


def _disk_status(
    row: PreparedModel,
    *,
    dry_run: bool,
    cache_dir: Path | None,
    free_reader: DiskFreeReader,
    present_check: PresentCheck,
) -> tuple[str | None, str]:
    """Disk preflight for one row: (block_reason, note).

    `block_reason` is a non-None string only when a real (non-dry) download must be refused for
    lack of space; the row is then failed before the long pull starts. `note` is an informational
    annotation ("cached (reuse)" or a dry-run preview of the shortfall) and never blocks.
    """
    if row["action"] not in (ACTION_PULL, ACTION_CACHE):
        return None, ""
    if present_check(row):
        return None, "cached (reuse)"
    store = store_dir_for(row["backend"], cache_dir)
    ok, reason = disk_precheck(estimate_download_mb(row), free_reader(store))
    if ok:
        return None, ""
    return (None, reason) if dry_run else (reason, "")


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


def _normalize_source_record(value: "str | SourceRecord | dict[str, object]") -> dict[str, object]:
    if isinstance(value, str):
        return {"source": value}
    return {k: v for k, v in value.items() if v is not None}


def _normalize_source_records(value: object) -> list[dict[str, object]]:
    """A backend's `sources` value is one source or a LIST of them (multiple quants of one model)."""
    if isinstance(value, list):
        return [_normalize_source_record(v) for v in value]
    return [_normalize_source_record(value)]  # type: ignore[arg-type]


def _expand_model_spec(
    records: dict[str, list[dict[str, object]]], model: ModelSpec, expanded: list[ModelSpec]
) -> None:
    for backend, recs in records.items():
        if backend not in SUPPORTED_BACKENDS:
            continue
        multi = len(recs) > 1
        for record in recs:
            source = record.get("source")
            if not isinstance(source, str) or not source:
                continue
            row = {**model, **record, "backend": backend, "source": source}
            if backend != model["backend"] or source != model["source"]:
                # Several quants of one backend (e.g. vLLM fp8 + w4a16) need distinct prep names.
                quant = record.get("quant")
                suffix = f"-{quant}" if multi and quant else ""
                row["name"] = f"{model['name']}-{backend}{suffix}"
            expanded.append(cast(ModelSpec, row))


def _expand_prepare_sources(models: list[ModelSpec]) -> list[ModelSpec]:
    """Expand a logical model into concrete backend artifacts that can be prepared.

    The resolver already understands per-backend `sources:` records. Model preparation needs the
    same expansion so a 16 GB host pulls Ollama GGUF fallbacks such as MamayLM/Lapa while also
    caching vLLM Hugging Face weights that fit the GPU. A backend that maps to a LIST of records
    (several vLLM quants of one model) expands to one prep artifact per quant.
    """
    expanded: list[ModelSpec] = []
    for model in models:
        records: dict[str, list[dict[str, object]]] = {
            backend: _normalize_source_records(source)
            for backend, source in (model.get("sources") or {}).items()
        }
        records.setdefault(model["backend"], [{"source": model["source"]}])

        _expand_model_spec(records, model, expanded)

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
    disk_free_reader: DiskFreeReader | None = None,
    present_check: PresentCheck | None = None,
) -> PreparationReport:
    """Execute (or, with dry_run, just plan) model preparation. Returns a report dict."""
    gpus = detect_gpus() if gpus is None else gpus
    max_mb = max_vram_mb(gpus)
    rows = plan(models, max_mb, bool(gpus), backend_filter, force)
    ollama_pull = ollama_pull or _ollama_pull
    hf_cache = hf_cache or _hf_cache
    disk_free_reader = disk_free_reader or hardware.disk_free_mb
    present_check = present_check or _default_present_check

    results: list[PreparedModel] = []
    for row in rows:
        if progress is not None:
            progress(row)
        block, disk_note = _disk_status(
            row,
            dry_run=dry_run,
            cache_dir=cache_dir,
            free_reader=disk_free_reader,
            present_check=present_check,
        )
        if block is not None:
            results.append({**row, "status": "failed", "detail": block})
            continue
        status, detail = _prepare_row_status(
            row,
            dry_run=dry_run,
            ollama_pull=ollama_pull,
            hf_cache=hf_cache,
            token=token,
            cache_dir=cache_dir,
        )
        if disk_note:
            detail = f"{detail}  [disk: {disk_note}]" if detail else f"[disk: {disk_note}]"
        url = acceptance_url(row)
        if url and "huggingface.co" not in detail:
            detail = f"{detail}  [license: {url}]"
        results.append({**row, "status": status, "detail": detail})
    return {"gpus": gpus, "max_vram_mb": max_mb, "results": results}
