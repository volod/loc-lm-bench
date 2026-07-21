"""Retrieval validation + query-glossary commands."""

from pathlib import Path
from contextlib import nullcontext
from typing import Any, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config


@app.command("validate-retrieval")
def validate_retrieval(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    k: int = typer.Option(10, help="recall@k cutoff (Premise 4 gate is recall@10 >= 0.8)"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
    retrieval_backend: Optional[str] = typer.Option(
        None, help="faiss | graph | fused (vector + GraphRAG)"
    ),
    retrieval_strategy: Optional[str] = typer.Option(
        None, help="graph strategy: local_khop | global_community"
    ),
    graph_weight: Optional[float] = typer.Option(
        None, help="fused backend: graph share of weighted RRF, 0..1 (default 0.3)"
    ),
    query_prep: Optional[str] = typer.Option(
        None,
        "--query-prep",
        help="opt-in query-side lane (uk-query-processing): comma-separated deterministic steps "
        "normalize,typos,glossary plus model steps rewrite,hyde,decompose",
    ),
    query_glossary: Optional[Path] = typer.Option(
        None, help="query_glossary.json for the 'glossary' step (build-query-glossary)"
    ),
    query_prep_typo_guard: bool = typer.Option(
        False,
        "--query-prep-typo-guard",
        help="typos step: leave an OOV token pymorphy3 knows as a valid Ukrainian word form "
        "unchanged (an inflection is not a misspelling)",
    ),
    query_prep_ab: bool = typer.Option(
        False,
        "--query-prep-ab",
        help="A/B report: recall@k / MRR at baseline then each cumulative query-prep step, with "
        "per-step deltas (proves each step's retrieval effect before turning it on)",
    ),
    query_prep_model: Optional[str] = typer.Option(
        None, help="local model for rewrite/hyde/decompose query-prep steps"
    ),
    query_prep_backend: Optional[str] = typer.Option(
        None, help="local backend for model query prep: ollama | vllm | llamacpp"
    ),
    out: Optional[Path] = typer.Option(None, help="write the A/B JSON report here"),
) -> None:
    """Score the configured backend's retrieval over the gold set (does not rank models)."""
    from llb.executor.cases import spans_as_dicts
    from llb.executor.runner_backend import _make_launcher
    from llb.executor.runner_retrieval import _load_store, build_query_prep
    from llb.goldset.schema import load_goldset
    from llb.rag import retrieval
    from llb.rag.query_prep.pipeline import QueryPrep
    from llb.rag.query_prep.retrieval import retrieve_prepared

    steps = [s.strip() for s in query_prep.split(",") if s.strip()] if query_prep else []
    cfg = load_config(
        config,
        goldset_path=goldset,
        retrieval_backend=retrieval_backend,
        retrieval_strategy=retrieval_strategy,
        graph_weight=graph_weight,
        query_prep=steps or None,
        query_glossary_path=query_glossary,
        query_prep_typo_guard=query_prep_typo_guard or None,
    )
    store = _load_store(cfg)
    items = load_goldset(cfg.goldset_path)
    if split:
        items = [it for it in items if it.split == split]
    ab_items = [(it.question, spans_as_dicts(it)) for it in items]

    model_steps = {"rewrite", "hyde", "decompose"}.intersection(steps)
    if model_steps and query_prep_model is None:
        typer.echo(
            "[error] model-backed query prep needs --query-prep-model",
            err=True,
        )
        raise typer.Exit(code=2)
    endpoint_cfg = cfg.with_overrides(
        model=query_prep_model,
        backend=query_prep_backend or ("ollama" if query_prep_model else None),
    )
    launcher = _make_launcher(endpoint_cfg) if model_steps else None
    with launcher if launcher is not None else nullcontext(None) as active:
        pipeline = build_query_prep(endpoint_cfg, store, active) or QueryPrep.build(())
        if query_prep_ab:
            _emit_query_prep_ab_report(
                ab_items,
                store,
                k,
                steps,
                out,
                pipeline=pipeline,
                endpoint={"model": endpoint_cfg.model, "backend": endpoint_cfg.backend}
                if model_steps
                else None,
            )
            return

        pairs = [
            (retrieve_prepared(store, pipeline.process(question), k), spans)
            for question, spans in ab_items
        ]
        metrics = retrieval.evaluate_retrieval(pairs, k)
        gate = (
            "PASS" if metrics["recall_at_k"] >= 0.8 else "BELOW 0.8 (retrieval is the bottleneck)"
        )
        lane = f" query-prep={','.join(steps)}" if steps else ""
        typer.echo(
            f"[validate-retrieval] n={metrics['n']} recall@{k}={metrics['recall_at_k']:.3f} "
            f"mrr={metrics['mrr']:.3f}{lane} -> {gate}"
        )


def _emit_query_prep_ab_report(
    ab_items: list[Any],
    store: Any,
    k: int,
    steps: list[str],
    out: Optional[Path],
    *,
    pipeline: Any,
    endpoint: dict[str, str] | None,
) -> None:
    """Print (and optionally write) the per-step cumulative query-prep A/B retrieval report."""
    import json

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


@app.command("build-query-glossary")
def build_query_glossary_cmd(
    bundle: Optional[Path] = typer.Option(
        None, help="draft bundle dir with prompt_dictionary_candidates.jsonl"
    ),
    candidates: Optional[Path] = typer.Option(
        None, help="explicit prompt_dictionary_candidates.jsonl (overrides --bundle)"
    ),
    out: Path = typer.Option(..., help="write the query_glossary.json here"),
    no_transliterations: bool = typer.Option(
        False,
        "--no-transliterations",
        help="do not seed romanized Latin aliases from each Cyrillic term",
    ),
) -> None:
    """Build a query_glossary.json from a draft bundle's dictionary candidates (uk-query-processing).

    Each candidate term becomes a canonical entry with its recorded aliases plus (by default) a
    romanized Latin variant, so the query-prep 'glossary' step can expand transliterated or
    surzhyk spellings. Hand-add more surzhyk/transliteration aliases by editing the emitted JSON.
    """
    import json

    from llb.prep.ontology.constants import PROMPT_DICTIONARY_FILENAME
    from llb.rag.query_prep.glossary import build_glossary_from_candidates

    source = (
        candidates
        if candidates is not None
        else (bundle / PROMPT_DICTIONARY_FILENAME if bundle is not None else None)
    )
    if source is None:
        typer.echo("[error] build-query-glossary needs --bundle or --candidates", err=True)
        raise typer.Exit(code=2)
    if not source.is_file():
        typer.echo(f"[error] dictionary candidates not found: {source}", err=True)
        raise typer.Exit(code=2)
    rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    glossary = build_glossary_from_candidates(rows, add_transliterations=not no_transliterations)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(glossary.to_dict(source_bundle=str(source)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(f"[build-query-glossary] {len(glossary.entries)} entries -> {out}")
