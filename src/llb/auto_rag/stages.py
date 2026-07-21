"""Production stage adapters for the auto-RAG orchestrator."""

from pathlib import Path
from typing import Any

from llb.auto_rag.evaluation import final_eval_stage
from llb.auto_rag.models import AutoRagSettings


def ingest_stage(settings: AutoRagSettings, _outputs: dict[str, Any]) -> dict[str, Any]:
    from llb.prep.corpus_ingest import ingest_corpus

    out = settings.run_dir / "stages" / "ingest" / "corpus"
    result = ingest_corpus(settings.corpus, out, min_chars=1)
    if result.n_docs < 1:
        raise ValueError("corpus ingestion produced no usable documents")
    return {
        "corpus": str(result.out_dir),
        "manifest": str(result.out_dir / "corpus_manifest.json"),
        "n_docs": result.n_docs,
        "n_skipped": result.n_skipped,
    }


def draft_stage(settings: AutoRagSettings, outputs: dict[str, Any]) -> dict[str, Any]:
    from llb.prep.ontology.endpoint_config import EndpointConfig, EndpointPlan
    from llb.prep.ontology.pipeline.run import draft_goldset

    stage_dir = settings.run_dir / "stages" / "draft" / "bundle"
    endpoint = EndpointConfig(
        kind="local",
        model=settings.draft_model,
        backend="ollama",
        temperature=0.0,
        max_tokens=settings.draft_max_tokens,
        timeout=300.0,
        think=False,
        num_ctx=settings.draft_num_ctx,
    )
    result = draft_goldset(
        outputs["ingest"]["corpus"],
        EndpointPlan.single(endpoint),
        max_items=settings.max_items,
        seed=settings.seed,
        out_dir=stage_dir,
        doc_limit=settings.doc_limit,
        extract_concurrency=settings.draft_concurrency,
        resume=(stage_dir / "extraction_journal.meta.json").is_file(),
    )
    return {
        "bundle": str(result.out_dir),
        "goldset": str(result.out_dir / "goldset.jsonl"),
        "ontology": str(result.out_dir / "ontology.json"),
        "n_items": len(result.items),
    }


def verification_stage(settings: AutoRagSettings, outputs: dict[str, Any]) -> dict[str, Any]:
    from llb.auto_rag.verification import verify_bundle
    from llb.core.config_validation import DEFAULT_OLLAMA_HOST

    policy = settings.gate_policy
    if policy == "auto":
        policy = (
            "frontier"
            if settings.egress_consent and (settings.max_usd or settings.max_calls)
            else "local"
        )
    return verify_bundle(
        Path(outputs["draft"]["bundle"]),
        settings.run_dir / "stages" / "verification",
        policy=policy,
        judge_model=settings.judge_model or settings.draft_model,
        judge_base_url=settings.judge_base_url or f"{DEFAULT_OLLAMA_HOST}/v1",
        threshold=settings.verify_threshold,
        min_accept_rate=settings.min_accept_rate,
        egress_consent=settings.egress_consent,
        max_usd=settings.max_usd,
        max_calls=settings.max_calls,
        scorer_ledger=settings.run_dir / "scorer_ledger.jsonl",
    )


def retrieval_stage(settings: AutoRagSettings, outputs: dict[str, Any]) -> dict[str, Any]:
    from llb.auto_rag.retrieval import validate_and_repair_retrieval

    return validate_and_repair_retrieval(
        Path(outputs["ingest"]["corpus"]),
        Path(outputs["verification"]["goldset"]),
        settings.run_dir / "stages" / "retrieval",
        k=settings.retrieval_k,
        recall_gate=settings.retrieval_recall_gate,
    )


def joint_search_stage(settings: AutoRagSettings, outputs: dict[str, Any]) -> dict[str, Any]:
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.backends.prepare.manifest import load_manifest
    from llb.core.config import RunConfig
    from llb.optimize.joint_search import run_joint_search

    models = load_manifest(settings.candidates)
    if settings.candidate_models:
        wanted = set(settings.candidate_models)
        models = [model for model in models if model["name"] in wanted]
        missing = sorted(wanted - {model["name"] for model in models})
        if missing:
            raise ValueError(f"candidate model names not found: {', '.join(missing)}")
    if not models:
        raise ValueError("auto-RAG needs at least one candidate model")
    runtime_data = settings.run_dir / "stages" / "joint_search" / "data"
    cfg = RunConfig(
        data_dir=runtime_data,
        corpus_root=outputs["ingest"]["corpus"],
        goldset_path=outputs["verification"]["goldset"],
        scorer_policy="human",
        seed=settings.seed,
    )
    result = run_joint_search(
        cfg,
        models,
        n_trials=settings.trials,
        run_id=settings.run_id,
        screen_limit=settings.screen_limit,
        min_finalists=min(settings.min_finalists, len(models)),
        objectives=settings.objectives,
        vram_mib=max_vram_mb(detect_gpus()),
        ram_mib=detect_ram_mb(),
        seed=settings.seed,
        max_model_len=settings.max_model_len,
        case_limit=settings.eval_limit,
    )
    if result.recommended is None:
        raise ValueError(f"joint search produced no recommendation; skipped={result.skipped}")
    return {
        "run_dir": str(result.run_dir),
        "scoreboard": str(result.scoreboard_paths["json"]),
        "recommended": result.recommended,
        "finalists": [item.name for item in result.finalists],
        "skipped": result.skipped,
    }


def prompt_system_stage(settings: AutoRagSettings, outputs: dict[str, Any]) -> dict[str, Any]:
    from llb.prompt_system.pipeline import CANDIDATES_FILE, prepare_prompt_system
    from llb.prompt_system.review import pin, save_candidates

    package = settings.run_dir / "stages" / "prompt_system" / "package"
    run = prepare_prompt_system(
        outputs["ingest"]["corpus"],
        out_dir=package,
        context_window=settings.max_model_len,
        ontology_bundle=outputs["draft"]["bundle"],
    )
    selected = min(run.candidates, key=_prompt_candidate_rank)
    pin(selected, "selected autonomously: least context loss, then strongest knowledge tree")
    save_candidates(run.candidates, package / CANDIDATES_FILE)
    return {
        "package": str(package),
        "prompt_system_id": selected.prompt_system_id,
        "used_tokens": selected.used_tokens,
        "knowledge_tree": selected.knowledge_tree,
    }


def _prompt_candidate_rank(candidate: Any) -> tuple[int, int, int, int, str]:
    dropped = sum(section["n_dropped"] for section in candidate.dropped_context["sections"])
    fields = candidate.fields
    return (
        -int(bool(candidate.knowledge_tree)),
        dropped,
        -fields.knowledge_tree_depth,
        -fields.knowledge_tree_budget,
        candidate.prompt_system_id,
    )


def recommendation_stage(settings: AutoRagSettings, outputs: dict[str, Any]) -> dict[str, Any]:
    from llb.auto_rag.recommendation import write_recommendation

    return write_recommendation(settings.run_dir, outputs)


DEFAULT_STAGES = {
    "ingest": ingest_stage,
    "draft": draft_stage,
    "verification": verification_stage,
    "retrieval": retrieval_stage,
    "joint_search": joint_search_stage,
    "prompt_system": prompt_system_stage,
    "final_eval": final_eval_stage,
    "recommendation": recommendation_stage,
}
