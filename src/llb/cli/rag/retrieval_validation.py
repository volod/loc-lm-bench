"""Execution helpers for the ``validate-retrieval`` CLI command."""

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from llb.cli.helpers import load_config

MODEL_QUERY_PREP_STEPS = frozenset({"rewrite", "hyde", "decompose"})
RETRIEVAL_RECALL_GATE = 0.8


@dataclass(frozen=True)
class RetrievalValidationRequest:
    """Resolved CLI inputs for one retrieval validation run."""

    config: Path | None
    goldset: Path | None
    k: int
    split: str | None
    retrieval_backend: str | None
    retrieval_strategy: str | None
    graph_weight: float | None
    query_prep: str | None
    query_glossary: Path | None
    query_prep_typo_guard: bool
    query_prep_ab: bool
    query_prep_model: str | None
    query_prep_backend: str | None
    out: Path | None


def _parse_query_prep_steps(value: str | None) -> list[str]:
    return [step.strip() for step in value.split(",") if step.strip()] if value else []


def _load_validation_inputs(
    request: RetrievalValidationRequest, steps: list[str]
) -> tuple[Any, Any, list[Any]]:
    from llb.executor.cases import spans_as_dicts
    from llb.executor.runner_retrieval import _load_store
    from llb.goldset.schema import load_goldset

    cfg = load_config(
        request.config,
        goldset_path=request.goldset,
        retrieval_backend=request.retrieval_backend,
        retrieval_strategy=request.retrieval_strategy,
        graph_weight=request.graph_weight,
        query_prep=steps or None,
        query_glossary_path=request.query_glossary,
        query_prep_typo_guard=request.query_prep_typo_guard or None,
    )
    store = _load_store(cfg)
    items = load_goldset(cfg.goldset_path)
    if request.split:
        items = [item for item in items if item.split == request.split]
    return cfg, store, [(item.question, spans_as_dicts(item)) for item in items]


def _model_endpoint(
    cfg: Any, request: RetrievalValidationRequest, steps: list[str]
) -> tuple[Any, Any, dict[str, str] | None]:
    from llb.executor.runner_backend import _make_launcher

    model_steps = MODEL_QUERY_PREP_STEPS.intersection(steps)
    if model_steps and request.query_prep_model is None:
        typer.echo("[error] model-backed query prep needs --query-prep-model", err=True)
        raise typer.Exit(code=2)
    endpoint_cfg = cfg.with_overrides(
        model=request.query_prep_model,
        backend=request.query_prep_backend or ("ollama" if request.query_prep_model else None),
    )
    launcher = _make_launcher(endpoint_cfg) if model_steps else None
    endpoint = (
        {"model": endpoint_cfg.model, "backend": endpoint_cfg.backend} if model_steps else None
    )
    return endpoint_cfg, launcher, endpoint


def _score_retrieval(
    ab_items: list[Any], store: Any, pipeline: Any, k: int, steps: list[str]
) -> None:
    from llb.rag import retrieval
    from llb.rag.query_prep.retrieval import retrieve_prepared

    pairs = [
        (retrieve_prepared(store, pipeline.process(question), k), spans)
        for question, spans in ab_items
    ]
    metrics = retrieval.evaluate_retrieval(pairs, k)
    gate = (
        "PASS"
        if metrics["recall_at_k"] >= RETRIEVAL_RECALL_GATE
        else f"BELOW {RETRIEVAL_RECALL_GATE} (retrieval is the bottleneck)"
    )
    lane = f" query-prep={','.join(steps)}" if steps else ""
    typer.echo(
        f"[validate-retrieval] n={metrics['n']} recall@{k}={metrics['recall_at_k']:.3f} "
        f"mrr={metrics['mrr']:.3f}{lane} -> {gate}"
    )


def run_retrieval_validation(request: RetrievalValidationRequest) -> None:
    """Build the configured pipeline and emit either a score line or an A/B report."""
    from llb.executor.runner_retrieval import build_query_prep
    from llb.rag.query_prep.pipeline import QueryPrep

    steps = _parse_query_prep_steps(request.query_prep)
    cfg, store, ab_items = _load_validation_inputs(request, steps)
    endpoint_cfg, launcher, endpoint = _model_endpoint(cfg, request, steps)
    launcher_context = launcher if launcher is not None else nullcontext(None)
    with launcher_context as active:
        pipeline = build_query_prep(endpoint_cfg, store, active) or QueryPrep.build(())
        if request.query_prep_ab:
            _emit_query_prep_ab_report(
                ab_items,
                store,
                request.k,
                steps,
                request.out,
                pipeline=pipeline,
                endpoint=endpoint,
            )
            return
        _score_retrieval(ab_items, store, pipeline, request.k, steps)


def _emit_query_prep_ab_report(
    ab_items: list[Any],
    store: Any,
    k: int,
    steps: list[str],
    out: Path | None,
    *,
    pipeline: Any,
    endpoint: dict[str, str] | None,
) -> None:
    """Print (and optionally write) the per-step cumulative query-prep A/B retrieval report."""
    from llb.rag.query_prep.report import (
        cumulative_pipelines,
        format_query_prep_ab,
        query_prep_ab_report,
    )
    from llb.rag.query_prep.retrieval import retrieve_prepared

    stages = cumulative_pipelines(
        steps,
        vocabulary=pipeline.vocabulary,
        glossary=pipeline.glossary,
        rewriter=pipeline.rewriter,
        hypothesizer=pipeline.hypothesizer,
        decomposer=pipeline.decomposer,
        known_word=pipeline.known_word,
        plausible=pipeline.plausible,
    )
    report = query_prep_ab_report(
        ab_items, lambda result, depth: retrieve_prepared(store, result, depth), k, stages
    )
    if endpoint is not None:
        report["endpoint"] = endpoint
    typer.echo(format_query_prep_ab(report))
    if out is not None:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"[validate-retrieval] wrote A/B report -> {out}")
