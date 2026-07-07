"""Generate per-tier serving scripts and llb run configs from templates."""

import json
import logging
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llb.backends.hardware import detect_gpus
from llb.core.paths import PROJECT_ROOT, resolve_data_dir

_LOG = logging.getLogger(__name__)

SUPPORTED_TIERS_GB = (12, 16, 24, 32)
PRIMARY_TARGETS = ("mamaylm", "lapa", "gemma-4", "qwen3.6", "mistral")
GEMMA4_TARGET_PREFIX = "gemma-4"
DEFAULT_MANIFEST = PROJECT_ROOT / "samples" / "config-example" / "manifest.yaml"
TEMPLATE_DIR = PROJECT_ROOT / "samples" / "config-example" / "templates"


@dataclass(frozen=True)
class GpuTierInfo:
    tier_gb: int
    total_mb: int
    gpu_name: str
    detected: bool


def bucket_vram_mb_to_tier(total_mb: int) -> int:
    """Map nvidia-smi total VRAM (MiB) to a supported tier (12/16/24/32 GiB).

    Thresholds use GiB with slack for cards that report less than nominal size
    (e.g. 16380 MiB -> 16 GiB tier).
    """
    gib = total_mb / 1024
    if gib < 14:
        return 12
    if gib < 20:
        return 16
    if gib < 28:
        return 24
    return 32


def detect_gpu_tier() -> GpuTierInfo:
    """Detect the primary GPU and return its serving tier."""
    gpus = detect_gpus()
    if not gpus:
        return GpuTierInfo(tier_gb=16, total_mb=0, gpu_name="", detected=False)
    primary = max(gpus, key=lambda g: g.total_mb)
    tier = bucket_vram_mb_to_tier(primary.total_mb)
    return GpuTierInfo(
        tier_gb=tier,
        total_mb=primary.total_mb,
        gpu_name=primary.name,
        detected=True,
    )


def resolve_tier(gpu_gb: int | None) -> GpuTierInfo:
    if gpu_gb is not None:
        if gpu_gb not in SUPPORTED_TIERS_GB:
            raise ValueError(
                f"unsupported GPU tier {gpu_gb} GiB; choose one of {list(SUPPORTED_TIERS_GB)}"
            )
        detected = detect_gpu_tier()
        return GpuTierInfo(
            tier_gb=gpu_gb,
            total_mb=detected.total_mb,
            gpu_name=detected.gpu_name,
            detected=detected.detected,
        )
    return detect_gpu_tier()


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or DEFAULT_MANIFEST
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"manifest must be a mapping: {manifest_path}")
    return raw


def _tier_entries(manifest: dict[str, Any], tier_gb: int) -> dict[str, Any]:
    tiers = manifest.get("tiers")
    if not isinstance(tiers, dict):
        raise ValueError("manifest.tiers must be a mapping")
    entry = tiers.get(tier_gb)
    if entry is None:
        entry = tiers.get(str(tier_gb))
    if not isinstance(entry, dict):
        raise ValueError(f"manifest has no tier entry for {tier_gb} GiB GPU")
    return entry


def _model_size_b(target_id: str, model: str) -> float:
    text = f"{target_id} {model}".lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*b", text)
    return max((float(value) for value in matches), default=0.0)


def _gemma4_rank(row: dict[str, Any], allow_cuda: bool) -> tuple[int, float]:
    backend = str(row["backend"])
    cuda_score = 1 if allow_cuda and backend == "vllm" else 0
    return cuda_score, _model_size_b(str(row["target"]), str(row["model"]))


def _supports_min_context(row: dict[str, Any], min_context_tokens: int | None) -> bool:
    """Whether a serving row has enough configured context for the requested workflow."""
    if not min_context_tokens:
        return True
    if row.get("backend") != "vllm":
        return True
    max_model_len = row.get("max_model_len")
    return isinstance(max_model_len, int) and max_model_len >= min_context_tokens


def select_host_gemma4_target(
    *,
    gpu_gb: int | None = None,
    manifest_path: Path | None = None,
    min_context_tokens: int | None = None,
) -> dict[str, Any]:
    """Return the most capable Gemma 4 target for the resolved CUDA tier.

    CUDA hosts prefer vLLM Gemma 4 rows over larger Ollama/offload rows. Within the same backend
    class, larger Gemma 4 parameter counts win. CPU/no-GPU fallback still returns a Gemma 4 row,
    but it does not prefer vLLM unless the caller explicitly supplied a GPU tier. When
    `min_context_tokens` is set, short-context vLLM eval cells are ignored for long-prompt
    workflows such as corpus drafting.
    """
    tier_info = resolve_tier(gpu_gb)
    manifest = load_manifest(manifest_path)
    entries = _tier_entries(manifest, tier_info.tier_gb)
    allow_cuda = tier_info.detected or gpu_gb is not None
    rows: list[dict[str, Any]] = []
    for target_id, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        if target_id != GEMMA4_TARGET_PREFIX and not target_id.startswith(
            f"{GEMMA4_TARGET_PREFIX}-"
        ):
            continue
        row: dict[str, Any] = {
            "tier_gb": tier_info.tier_gb,
            "gpu_total_mb": tier_info.total_mb,
            "gpu_name": tier_info.gpu_name,
            "gpu_detected": tier_info.detected,
            "target": target_id,
            "backend": str(entry["backend"]),
            "model": str(entry["model"]),
        }
        if entry.get("gpu_memory_utilization") is not None:
            row["gpu_memory_utilization"] = float(entry["gpu_memory_utilization"])
        if entry.get("max_model_len") is not None:
            row["max_model_len"] = int(entry["max_model_len"])
        if entry.get("cpu_offload_gb") is not None:
            row["cpu_offload_gb"] = float(entry["cpu_offload_gb"])
        if entry.get("kv_offloading_size_gb") is not None:
            row["kv_offloading_size_gb"] = float(entry["kv_offloading_size_gb"])
        if _supports_min_context(row, min_context_tokens):
            rows.append(row)
    if not rows:
        suffix = f" with context >= {min_context_tokens} tokens" if min_context_tokens else ""
        raise ValueError(f"tier {tier_info.tier_gb}: no Gemma 4 serving target{suffix}")
    return max(rows, key=lambda row: _gemma4_rank(row, allow_cuda))


def _render(template_text: str, ctx: dict[str, str]) -> str:
    out = template_text
    for key, value in ctx.items():
        out = out.replace("{" + key + "}", value)
    return out


def _load_template(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")


def _optional_yaml_fields(entry: dict[str, Any]) -> str:
    lines: list[str] = []
    if entry.get("request_timeout_s") is not None:
        lines.append(f"request_timeout_s: {entry['request_timeout_s']}")
    if entry.get("backend") == "vllm":
        util = float(entry["gpu_memory_utilization"])
        lines.append(f"gpu_memory_utilization: {util:g}")
        lines.append(f"max_model_len: {entry['max_model_len']}")
        if entry.get("cpu_offload_gb") is not None:
            lines.append(f"cpu_offload_gb: {float(entry['cpu_offload_gb']):g}")
        if entry.get("kv_offloading_size_gb") is not None:
            lines.append(f"kv_offloading_size_gb: {float(entry['kv_offloading_size_gb']):g}")
    return "\n".join(lines)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _config_rel_path(config_path: Path) -> str:
    try:
        return config_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(config_path.resolve())


def _emit_target(
    *,
    tier_gb: int,
    target_id: str,
    entry: dict[str, Any],
    data_root: Path,
    eval_defaults: dict[str, Any],
    vllm_defaults: dict[str, Any],
) -> dict[str, Any]:
    backend = str(entry["backend"])
    model = str(entry["model"])
    run_name = f"serving-{tier_gb}gb-{target_id.replace('.', '-')}"
    config_name = f"run_eval_{target_id.replace('-', '_')}.yaml"
    config_path = data_root / config_name
    config_rel = _config_rel_path(config_path)

    yaml_ctx = {
        "tier_gb": str(tier_gb),
        "target_id": target_id,
        "run_name": run_name,
        "seed": str(eval_defaults.get("seed", 13)),
        "model": model,
        "backend": backend,
        "max_tokens": str(eval_defaults.get("max_tokens", 256)),
        "temperature": str(eval_defaults.get("temperature", 0.0)),
        "n_shot": str(eval_defaults.get("n_shot", 0)),
        "optional_fields": _optional_yaml_fields(entry),
        "embedding_model": str(
            eval_defaults.get("embedding_model", "intfloat/multilingual-e5-base")
        ),
        "strategy": str(eval_defaults.get("strategy", "recursive")),
        "chunk_size": str(eval_defaults.get("chunk_size", 800)),
        "chunk_overlap": str(eval_defaults.get("chunk_overlap", 120)),
        "top_k": str(eval_defaults.get("top_k", 5)),
        "retrieval_mode": str(eval_defaults.get("retrieval_mode", "flat")),
        "measure_telemetry": str(eval_defaults.get("measure_telemetry", True)).lower(),
        "goldset_path": str(
            eval_defaults.get(
                "goldset_path", "samples/goldsets/ua_squad_postedited_v1/goldset.jsonl"
            )
        ),
    }
    config_path.write_text(
        _render(_load_template("run_eval.yaml.tmpl"), yaml_ctx), encoding="utf-8"
    )

    serve_name = f"serve_{target_id.replace('-', '_')}.sh"
    if backend == "vllm":
        serve_ctx = {
            "tier_gb": str(tier_gb),
            "target_id": target_id,
            "model": model,
            "port": str(vllm_defaults.get("port", 8000)),
            "gpu_memory_utilization": str(entry["gpu_memory_utilization"]),
            "max_model_len": str(entry["max_model_len"]),
            "cpu_offload_arg": (
                f"  --cpu-offload-gb {float(entry['cpu_offload_gb']):g} \\\n"
                if entry.get("cpu_offload_gb") is not None
                else ""
            ),
            "kv_offloading_arg": (
                f"  --kv-offloading-size {float(entry['kv_offloading_size_gb']):g} \\\n"
                if entry.get("kv_offloading_size_gb") is not None
                else ""
            ),
            "kv_cache_dtype": str(vllm_defaults.get("kv_cache_dtype", "fp8")),
            "max_num_seqs": str(vllm_defaults.get("max_num_seqs", 1)),
            "limit_mm_per_prompt": str(entry.get("limit_mm_per_prompt", '{"image": 0}')),
        }
        serve_body = _render(_load_template("vllm_serve.sh.tmpl"), serve_ctx)
    else:
        serve_body = _render(
            _load_template("ollama_serve.sh.tmpl"),
            {"tier_gb": str(tier_gb), "target_id": target_id, "model": model},
        )
    _write_executable(data_root / serve_name, serve_body)

    run_sh_name = f"run_eval_{target_id.replace('-', '_')}.sh"
    _write_executable(
        data_root / run_sh_name,
        _render(
            _load_template("run_eval.sh.tmpl"),
            {"tier_gb": str(tier_gb), "target_id": target_id, "config_path": config_rel},
        ),
    )
    target_row: dict[str, Any] = {
        "target": target_id,
        "backend": backend,
        "model": model,
        "serve_script": serve_name,
        "run_eval_config": config_name,
        "run_eval_script": run_sh_name,
    }
    for key in (
        "gpu_memory_utilization",
        "max_model_len",
        "cpu_offload_gb",
        "kv_offloading_size_gb",
    ):
        if entry.get(key) is not None:
            target_row[key] = entry[key]
    return target_row


def generate_serving_configs(
    *,
    gpu_gb: int | None = None,
    output_root: Path | None = None,
    manifest_path: Path | None = None,
) -> Path:
    """Render serve/run scripts and YAML configs for the resolved GPU tier."""
    tier_info = resolve_tier(gpu_gb)
    tier_gb = tier_info.tier_gb
    manifest = load_manifest(manifest_path)
    eval_defaults = manifest.get("eval_defaults", {})
    vllm_defaults = manifest.get("vllm_defaults", {})
    tier_entries = _tier_entries(manifest, tier_gb)

    data_root = output_root or (resolve_data_dir() / "llb" / "serving" / f"gpu-{tier_gb}gb")
    data_root.mkdir(parents=True, exist_ok=True)

    target_rows: list[dict[str, Any]] = []
    written: list[str] = []

    for target_id in PRIMARY_TARGETS:
        entry = tier_entries.get(target_id)
        if not isinstance(entry, dict):
            raise ValueError(f"tier {tier_gb}: missing target {target_id!r}")
        row = _emit_target(
            tier_gb=tier_gb,
            target_id=target_id,
            entry=entry,
            data_root=data_root,
            eval_defaults=eval_defaults,
            vllm_defaults=vllm_defaults,
        )
        target_rows.append(row)
        written.extend(
            [str(row["run_eval_config"]), str(row["serve_script"]), str(row["run_eval_script"])]
        )

    for extra_id, entry in tier_entries.items():
        if extra_id in PRIMARY_TARGETS or not isinstance(entry, dict):
            continue
        row = _emit_target(
            tier_gb=tier_gb,
            target_id=extra_id,
            entry=entry,
            data_root=data_root,
            eval_defaults=eval_defaults,
            vllm_defaults=vllm_defaults,
        )
        row["extra"] = "true"
        target_rows.append(row)
        written.extend(
            [str(row["run_eval_config"]), str(row["serve_script"]), str(row["run_eval_script"])]
        )

    tier_json = {
        "tier_gb": tier_gb,
        "gpu_total_mb": tier_info.total_mb,
        "gpu_name": tier_info.gpu_name,
        "gpu_detected": tier_info.detected,
        "manifest": (manifest_path or DEFAULT_MANIFEST).relative_to(PROJECT_ROOT).as_posix(),
        "targets": target_rows,
        "files": sorted(set(written)),
    }
    (data_root / "tier.json").write_text(json.dumps(tier_json, indent=2) + "\n", encoding="utf-8")

    _LOG.info("[gen-serving-config] tier=%s GiB -> %s", tier_gb, data_root)
    return data_root


def format_detect_line(info: GpuTierInfo) -> str:
    if info.detected:
        return (
            f"gpu_tier={info.tier_gb} total_mb={info.total_mb} "
            f"name={info.gpu_name!r} supported={list(SUPPORTED_TIERS_GB)}"
        )
    return (
        f"gpu_tier={info.tier_gb} total_mb=0 name= supported={list(SUPPORTED_TIERS_GB)} "
        "(no GPU detected)"
    )
