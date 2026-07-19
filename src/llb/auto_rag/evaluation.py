"""Recommended-config final evaluation and independent parity rerun."""

from typing import Any

from llb.auto_rag.models import AutoRagSettings


def final_eval_stage(settings: AutoRagSettings, outputs: dict[str, Any]) -> dict[str, Any]:
    from llb.core.config import RunConfig
    from llb.executor.runner import run_eval
    from llb.optimize.tuner_runtime import _build_store
    from llb.prompt_system.selection import resolve_prompt_package

    recommended = outputs["joint_search"]["recommended"]
    overrides = dict(recommended.get("overrides") or {})
    base = RunConfig(
        data_dir=settings.run_dir / "stages" / "final_eval" / "data",
        corpus_root=outputs["ingest"]["corpus"],
        goldset_path=outputs["verification"]["goldset"],
        model=recommended["source"],
        backend=recommended["backend"],
        run_name=f"auto-rag-{settings.run_id}",
        scorer_policy="human",
        seed=settings.seed,
    )
    config = base.with_overrides(**overrides)
    selected = resolve_prompt_package(
        config.data_dir,
        outputs["prompt_system"]["prompt_system_id"],
        outputs["prompt_system"]["package"],
    )

    def evaluate() -> Any:
        return run_eval(
            config,
            store=_build_store(config),
            prompt_package=selected.package,
            prompt_system_provenance=selected.provenance,
            split="final",
            limit=settings.eval_limit,
            emit=True,
        )

    summary = _eval_summary(evaluate())
    if settings.parity_check:
        manual = _eval_summary(evaluate())
        tolerance = 1e-9
        delta = abs(summary["quality"] - manual["quality"])
        if delta > tolerance:
            raise ValueError(
                f"manual-chain parity failed: quality delta {delta:.6f} > {tolerance:.6f}"
            )
        summary["parity"] = {
            "quality": manual["quality"],
            "quality_delta": delta,
            "tolerance": tolerance,
            "manifest": manual["manifest"],
        }
    return summary


def _eval_summary(result: Any) -> dict[str, Any]:
    rows = result.get("rows") or []
    manifest = result.get("manifest") or {}
    if hasattr(manifest, "model_dump"):
        manifest = manifest.model_dump()
    paths = result.get("paths") or {}
    return {
        "quality": float(rows[0]["quality"]) if rows else 0.0,
        "n_cases": int(manifest.get("n_cases") or 0),
        "split": manifest.get("split"),
        "manifest": str(paths.get("manifest") or ""),
        "scores": str(paths.get("scores") or ""),
        "retrieval": result.get("retrieval") or {},
    }
