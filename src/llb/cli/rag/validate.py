"""Retrieval validation + query-glossary commands."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.rag.retrieval_validation import (
    RetrievalValidationRequest,
    run_retrieval_validation,
)


@app.command("validate-retrieval")
def validate_retrieval(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    corpus_root: Optional[Path] = typer.Option(
        None, help="corpus directory whose persisted index should be validated"
    ),
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
    request = RetrievalValidationRequest(
        config=config,
        corpus_root=corpus_root,
        goldset=goldset,
        k=k,
        split=split,
        retrieval_backend=retrieval_backend,
        retrieval_strategy=retrieval_strategy,
        graph_weight=graph_weight,
        query_prep=query_prep,
        query_glossary=query_glossary,
        query_prep_typo_guard=query_prep_typo_guard,
        query_prep_ab=query_prep_ab,
        query_prep_model=query_prep_model,
        query_prep_backend=query_prep_backend,
        out=out,
    )
    run_retrieval_validation(request)


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
