"""Render the machine-readable recommendation and concise evidence report."""

import json
from pathlib import Path
from typing import Any

import yaml

from llb.core.fsutil import atomic_write_text


def write_recommendation(run_dir: Path, outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    joint = outputs["joint_search"]
    final = outputs["final_eval"]
    retrieval = outputs["retrieval"]
    prompt = outputs["prompt_system"]
    recommended = joint["recommended"]
    overrides = dict(recommended.get("overrides") or {})
    selected_retrieval = retrieval["selected"]
    payload = {
        "schema_version": 1,
        "model": recommended["source"],
        "model_name": recommended["model"],
        "backend": recommended["backend"],
        "serving": {
            "max_model_len": overrides.get("max_model_len"),
            "gpu_memory_utilization": overrides.get("gpu_memory_utilization"),
            "cpu_offload_gb": overrides.get("cpu_offload_gb"),
            "kv_offloading_size_gb": overrides.get("kv_offloading_size_gb"),
            "dtype": overrides.get("dtype", "auto"),
            "quantization": overrides.get("quantization"),
            "n_gpu_layers": overrides.get("n_gpu_layers", -1),
        },
        "chunking": {
            "strategy": overrides.get("strategy", selected_retrieval["strategy"]),
            "size": overrides.get("chunk_size", selected_retrieval["chunk_size"]),
            "overlap": overrides.get("chunk_overlap", selected_retrieval["chunk_overlap"]),
        },
        "retrieval": {
            "backend": "faiss",
            "mode": overrides.get("retrieval_mode", selected_retrieval["retrieval_mode"]),
            "top_k": overrides.get("top_k", retrieval["k"]),
            "fusion_weight": overrides.get("fusion_weight"),
            "fusion_candidates": overrides.get("fusion_candidates"),
            "reranker": overrides.get("reranker"),
            "rerank_candidates": overrides.get("rerank_candidates"),
            "query_prep": overrides.get("query_prep", []),
            "context_budget": overrides.get("context_budget"),
        },
        "prompt_system": {
            "id": prompt["prompt_system_id"],
            "package": prompt["package"],
            "knowledge_tree": prompt["knowledge_tree"],
        },
        "evidence": {
            "verification": outputs["verification"],
            "retrieval_validation": retrieval,
            "joint_search": {
                "run_dir": joint["run_dir"],
                "scoreboard": joint["scoreboard"],
                "quality": recommended.get("quality"),
            },
            "final_split": final,
        },
    }
    yaml_path = run_dir / "rag_recommendation.yaml"
    report_path = run_dir / "report.md"
    # Break shared dict identities (retrieval.metrics also appears in attempts) so the operator
    # YAML stays anchor-free and is easy to copy into a standalone config.
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    atomic_write_text(
        yaml_path, yaml.safe_dump(payload, allow_unicode=False, sort_keys=False, width=100)
    )
    atomic_write_text(report_path, _report(payload, run_dir))
    links = {stage: result for stage, result in outputs.items() if stage != "recommendation"}
    atomic_write_text(
        run_dir / "artifacts.json", json.dumps(links, ensure_ascii=False, indent=2) + "\n"
    )
    return {
        "recommendation": str(yaml_path),
        "report": str(report_path),
        "artifacts": str(run_dir / "artifacts.json"),
    }


def _report(payload: dict[str, Any], run_dir: Path) -> str:
    retrieval = payload["evidence"]["retrieval_validation"]
    verification = payload["evidence"]["verification"]
    final = payload["evidence"]["final_split"]
    lines = [
        "# Auto-RAG recommendation",
        "",
        f"Run: `{run_dir.name}`",
        "",
        "## Selected configuration",
        "",
        f"- model: `{payload['model_name']}` (`{payload['model']}`)",
        f"- backend: `{payload['backend']}`",
        f"- chunking: `{payload['chunking']}`",
        f"- retrieval: `{payload['retrieval']}`",
        f"- prompt system: `{payload['prompt_system']['id']}`",
        "",
        "## Evidence",
        "",
        f"- verification: {verification['n_accepted']}/{verification['n_total']} accepted "
        f"({verification['accept_rate']:.1%}) via `{verification['policy']}`",
        f"- retrieval: recall@{retrieval['k']}={retrieval['metrics']['recall_at_k']:.3f}, "
        f"MRR={retrieval['metrics']['mrr']:.3f}, repaired={retrieval['repaired']}",
        f"- final split: n={final['n_cases']}, quality={final['quality']:.4f}",
    ]
    if final.get("parity"):
        lines.append(
            f"- manual-chain parity: delta={final['parity']['quality_delta']:.6f} "
            f"(tolerance={final['parity']['tolerance']:.6f})"
        )
    lines.append("")
    return "\n".join(lines)
