"""Retrieval validation + query-glossary commands."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config

if TYPE_CHECKING:
    from llb.core.config import RunConfig


@app.command("validate-retrieval")
def validate_retrieval(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    k: int = typer.Option(10, help="recall@k cutoff (Premise 4 gate is recall@10 >= 0.8)"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
    retrieval_backend: Optional[str] = typer.Option(None, help="faiss | graph (GraphRAG backend)"),
    retrieval_strategy: Optional[str] = typer.Option(
        None, help="graph strategy: local_khop | global_community"
    ),
    query_prep: Optional[str] = typer.Option(
        None,
        "--query-prep",
        help="opt-in query-side lane (uk-query-processing): comma-separated deterministic steps "
        "normalize,typos,glossary (the 'rewrite' step needs a model -- use run-eval)",
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
    out: Optional[Path] = typer.Option(None, help="write the A/B JSON report here"),
) -> None:
    """Score the configured backend's retrieval over the gold set (does not rank models)."""
    from llb.executor.cases import spans_as_dicts
    from llb.executor.runner_retrieval import _load_store
    from llb.goldset.schema import load_goldset
    from llb.rag import retrieval
    from llb.rag.query_prep.base import STEP_REWRITE
    from llb.rag.query_prep.pipeline import QueryPrep

    steps = [s.strip() for s in query_prep.split(",") if s.strip()] if query_prep else []
    cfg = load_config(
        config,
        goldset_path=goldset,
        retrieval_backend=retrieval_backend,
        retrieval_strategy=retrieval_strategy,
        query_prep=steps or None,
        query_glossary_path=query_glossary,
        query_prep_typo_guard=query_prep_typo_guard or None,
    )
    store = _load_store(cfg)
    items = load_goldset(cfg.goldset_path)
    if split:
        items = [it for it in items if it.split == split]
    ab_items = [(it.question, spans_as_dicts(it)) for it in items]

    if STEP_REWRITE in steps:
        typer.echo(
            "[error] validate-retrieval does not run the 'rewrite' step (it needs a model); "
            "use run-eval --query-prep for the LLM rewrite",
            err=True,
        )
        raise typer.Exit(code=2)
    vocabulary, glossary, known_word = _resolve_query_prep_deps(cfg, store, steps)

    if query_prep_ab:
        _emit_query_prep_ab_report(
            ab_items,
            store,
            k,
            steps,
            out,
            vocabulary=vocabulary,
            glossary=glossary,
            known_word=known_word,
        )
        return

    pipeline = QueryPrep.build(
        steps, vocabulary=vocabulary, glossary=glossary, known_word=known_word
    )
    pairs = [
        (store.retrieve(pipeline.process(question).processed, k), spans)
        for question, spans in ab_items
    ]
    metrics = retrieval.evaluate_retrieval(pairs, k)
    gate = "PASS" if metrics["recall_at_k"] >= 0.8 else "BELOW 0.8 (retrieval is the bottleneck)"
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
    vocabulary: Any,
    glossary: Any,
    known_word: Any,
) -> None:
    """Print (and optionally write) the per-step cumulative query-prep A/B retrieval report."""
    import json

    from llb.rag.query_prep.report import (
        cumulative_pipelines,
        format_query_prep_ab,
        query_prep_ab_report,
    )

    stages = cumulative_pipelines(
        steps, vocabulary=vocabulary, glossary=glossary, known_word=known_word
    )
    report = query_prep_ab_report(ab_items, store.retrieve, k, stages)
    typer.echo(format_query_prep_ab(report))
    if out is not None:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"[validate-retrieval] wrote A/B report -> {out}")


def _resolve_query_prep_deps(
    cfg: "RunConfig", store: "Any", steps: list[str]
) -> "tuple[Any, Any, Any]":
    """Resolve the (vocabulary, glossary, known-word probe) the query-prep steps need."""
    from llb.rag.query_prep.base import STEP_GLOSSARY, STEP_TYPOS
    from llb.rag.query_prep.glossary import Glossary
    from llb.rag.query_prep.typos import build_vocabulary

    vocabulary = None
    glossary = None
    known_word = None
    if STEP_TYPOS in steps:
        chunks = getattr(store, "chunks", None) or []
        vocabulary = build_vocabulary(str(chunk.get("text", "")) for chunk in chunks)
        if cfg.query_prep_typo_guard:
            from llb.rag.lexical import load_uk_word_probe

            known_word = load_uk_word_probe()
    if STEP_GLOSSARY in steps:
        if cfg.query_glossary_path is None:
            typer.echo(
                "[error] the 'glossary' step needs --query-glossary (build-query-glossary)",
                err=True,
            )
            raise typer.Exit(code=2)
        glossary = Glossary.load(cfg.query_glossary_path)
    return vocabulary, glossary, known_word


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
