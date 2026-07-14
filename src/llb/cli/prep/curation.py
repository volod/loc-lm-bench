"""Draft curation + external-service source-prep commands."""

from pathlib import Path
from typing import Any, Optional

import typer

from llb.cli.app import app


@app.command("curate-drafts")
def curate_drafts_cmd(
    inputs: list[Path] = typer.Argument(
        ..., help="exported artifact files to merge (raw JSON, fenced blocks, or JSONL)"
    ),
    kind: str = typer.Option(
        ..., help="artifact kind: squad | grounded | security | chains | inventory"
    ),
    out: Path = typer.Option(..., help="merged curated artifact output path"),
    corpus_root: Optional[Path] = typer.Option(
        None,
        help="staged corpus dir (.md/.txt); enables verbatim-quote grounding and repair",
    ),
    dedup_threshold: Optional[float] = typer.Option(
        None, help="cosine threshold for near-duplicate questions (default 0.9)"
    ),
    semantic_dedup: bool = typer.Option(
        True,
        "--semantic-dedup/--no-semantic-dedup",
        help="use the pinned E5 embedder for near-duplicate detection (falls back to "
        "exact-only when the [rag] extra is unavailable)",
    ),
    dedup_against: list[Path] = typer.Option(
        [], help="prior draft bundle dir(s); drop questions near-duplicating their goldsets"
    ),
    min_context_chars: Optional[int] = typer.Option(
        None, help="squad: drop items whose context is shorter than this (default 80)"
    ),
    dedup_spans: bool = typer.Option(
        False, help="squad: also drop repeated (context, answer-span) pairs"
    ),
) -> None:
    """Merge, deduplicate, and filter externally drafted artifacts into ONE importable file."""
    from llb.prep.curation import dispatcher as curation
    from llb.prep.curation.common import resolve_embedder

    if kind not in curation.KINDS:
        raise SystemExit(f"[curate] unknown --kind {kind!r} (expected one of {curation.KINDS})")
    embedder = resolve_embedder(semantic_dedup) if kind != "inventory" else None
    prior = curation.load_prior_bundle_questions(list(dedup_against)) if dedup_against else None
    kwargs: dict[str, Any] = {}
    if dedup_threshold is not None:
        kwargs["dedup_threshold"] = dedup_threshold
    if kind == "squad":
        if min_context_chars is not None:
            kwargs["min_context_chars"] = min_context_chars
        kwargs["dedup_spans"] = dedup_spans
    payload, report = curation.curate(
        kind,
        list(inputs),
        corpus_root=corpus_root,
        embedder=embedder,
        prior_questions=prior,
        **kwargs,
    )
    report_path = curation.write_curated(kind, payload, out, report)
    counts = report.to_dict()["counts"]
    typer.echo(
        f"[curate] {kind}: kept {report.kept}/{report.loaded} "
        f"(invalid={counts['invalid']} flabby={counts['flabby']} "
        f"exact-dup={counts['exact_duplicates']} near-dup={counts['near_duplicates']} "
        f"repaired={counts['repaired']}) -> {out}\n"
        f"[curate] report -> {report_path}"
    )


@app.command("coverage-plan-text")
def coverage_plan_text_cmd(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="curated inventory coverage JSON slice"
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="output text path (default: input path with .txt suffix)"
    ),
) -> None:
    """Convert a prompt-01 inventory coverage JSON slice into a NotebookLM source text file."""
    from llb.prep.curation.coverage_text import write_coverage_plan_text

    result = write_coverage_plan_text(input_path, out)
    typer.echo(
        "[coverage-plan-text] "
        f"{result.documents} docs, {result.cross_document_links} cross-links -> {result.path}"
    )


@app.command("import-external-draft")
def import_external_draft_cmd(
    artifact: Path = typer.Option(
        ..., help="grounded-JSONL Artifact B export (curated or raw; quote + source_doc_id rows)"
    ),
    corpus_root: Path = typer.Option(
        ..., help="local corpus dir (.md/.txt) each quote is re-grounded against"
    ),
    sidecar: Path = typer.Option(
        ...,
        help="external_provenance.json data-classification sidecar (must declare open); "
        "a missing or non-open sidecar aborts before writing any bundle",
    ),
    out_dir: Optional[Path] = typer.Option(
        None, help="output bundle dir (default: $DATA_DIR/prepare-goldset/<timestamp>/)"
    ),
    seed: int = typer.Option(13, help="deterministic split-assignment seed"),
    retrieval_index_dir: Optional[Path] = typer.Option(
        None,
        "--retrieval-index-dir",
        help="full-corpus RAG index; annotates each imported item with its gold-span "
        "retrieval_rank in item provenance (needle parity with local drafts)",
    ),
    retrieval_k: int = typer.Option(
        10, "--retrieval-k", min=1, help="top-k window for the needle-rank annotation"
    ),
    drop_nonretrievable_needles: bool = typer.Option(
        False,
        "--drop-nonretrievable-needles",
        help="drop imported items whose gold span is not retrieved within top-k "
        "(requires --retrieval-index-dir)",
    ),
) -> None:
    """Import an external-service grounded goldset (Artifact B) into a canonical draft bundle.

    Re-grounds every quote against the local corpus, drops + counts non-verbatim rows, computes
    exact source_spans, stamps provenance=frontier-drafted / verified=false, records the external
    service/model/classification, and carries question_type/difficulty in item provenance. Route the
    emitted bundle through the usual validate-goldset -> cross-check-goldset -> verify-* chain.
    """
    from llb.prep.external_draft import import_external_draft
    from llb.prep.ontology.pipeline.journaling import default_out_dir

    if retrieval_index_dir is not None and not retrieval_index_dir.is_dir():
        typer.echo(f"[error] retrieval index dir not found: {retrieval_index_dir}", err=True)
        raise typer.Exit(code=2)
    resolved_out_dir = out_dir or default_out_dir()
    result = import_external_draft(
        artifact,
        corpus_root,
        sidecar,
        resolved_out_dir,
        seed=seed,
        retrieval_index_dir=retrieval_index_dir,
        retrieval_k=retrieval_k,
        drop_nonretrievable_needles=drop_nonretrievable_needles,
    )
    counts = result.report.to_dict()["counts"]
    typer.echo(
        f"[import-external-draft] imported {result.report.kept}/{result.report.loaded} items "
        f"(verified=false; dropped={counts['dropped']} repaired={counts['repaired']}) "
        f"-> {result.out_dir}"
    )
    if result.validation["errors"]:
        for err in result.validation["errors"][:20]:
            typer.echo(f"[import-external-draft] VALIDATION ERROR: {err}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"[import-external-draft] validation PASS (splits={result.validation['splits']})")
