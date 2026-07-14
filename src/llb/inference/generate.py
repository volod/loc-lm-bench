"""Generate per-tier serving scripts and llb run configs from templates."""

import json
import logging
import stat
from pathlib import Path
from typing import Any


from llb.core.paths import PROJECT_ROOT, resolve_data_dir
from llb.inference.serving_selection import (
    DEFAULT_MANIFEST,
    PRIMARY_TARGETS,
    _tier_entries,
    load_manifest,
    resolve_tier,
)

_LOG = logging.getLogger(__name__)

TEMPLATE_DIR = PROJECT_ROOT / "samples" / "config-example" / "templates"


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
