"""Corpus ingestion commands: PDF and mixed txt/md/pdf -> canonical .md corpus."""

import json
from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app

REPEAT_BLOCKS_HELP = (
    "intra-document repeated-block handling: keep (default, unchanged) | drop (index the "
    "first copy of a repeated block, remove the rest) | anchor (keep every copy, prefixed "
    "with its enclosing-heading breadcrumb so each stays retrievable in its own section)"
)


@app.command("ingest-pdf-corpus")
def ingest_pdf_corpus_cmd(
    pdf_root: Path = typer.Option(..., help="directory of local PDF source documents"),
    out_dir: Optional[Path] = typer.Option(
        None, help="output corpus dir of extracted .md files (default: <pdf-root>/_md)"
    ),
    min_chars: int = typer.Option(
        500, min=1, help="skip PDFs whose extracted text is shorter than this"
    ),
    parser: str = typer.Option(
        "auto", help="PDF parser: auto | pymupdf4llm | docling | marker | unstructured | markitdown"
    ),
    limit: Optional[int] = typer.Option(None, help="cap the number of PDFs to ingest"),
    refresh: bool = typer.Option(
        False, "--refresh", help="reconvert every PDF even when the source is unchanged"
    ),
    repeat_blocks: str = typer.Option("keep", help=REPEAT_BLOCKS_HELP, metavar="keep|drop|anchor"),
) -> None:
    """Extract local PDFs into the `.md` corpus shape used by RAG, goldset, and GraphRAG commands."""
    _run_pdf_markdown_ingest(
        "ingest-pdf-corpus",
        pdf_root,
        out_dir,
        min_chars,
        parser,
        limit,
        refresh,
        repeat_blocks=repeat_blocks,
    )


@app.command("ingest-corpus")
def ingest_corpus_cmd(
    root: Path = typer.Option(..., help="directory of mixed .txt/.md/.pdf source documents"),
    out_dir: Optional[Path] = typer.Option(
        None, help="output corpus dir of .md/.txt files (default: <root>/_md)"
    ),
    min_chars: int = typer.Option(
        500, min=1, help="skip documents whose text is shorter than this"
    ),
    parser: str = typer.Option(
        "auto", help="PDF parser: auto | pymupdf4llm | docling | marker | unstructured | markitdown"
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="reconvert/re-copy every source even when it is unchanged"
    ),
    default_language: Optional[str] = typer.Option(
        None,
        "--default-language",
        help="language tag for sources that do not provide one (otherwise a cheap detector runs)",
    ),
    source_system: str = typer.Option(
        "local", help="default source-system tag recorded in corpus governance metadata"
    ),
    acl_label: Optional[str] = typer.Option(
        None, "--acl-label", help="default ACL label copied to manifest items and chunks"
    ),
) -> None:
    """Ingest a mixed txt/md/pdf directory into one canonical corpus (PDFs converted, text passed through)."""
    from llb.prep.corpus_ingest import ingest_corpus

    try:
        result = ingest_corpus(
            root,
            out_dir,
            min_chars=min_chars,
            parser=parser,
            refresh=refresh,
            default_language=default_language,
            source_system=source_system,
            acl_label=acl_label,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    reused_note = f", {result.n_reused} reused unchanged" if result.n_reused else ""
    removed_note = f", {result.n_removed_sources} removed" if result.n_removed_sources else ""
    typer.echo(
        f"[ingest-corpus] {result.n_docs}/{len(result.items)} documents ingested "
        f"({result.n_skipped} skipped{reused_note}{removed_note}) -> {result.out_dir}"
    )


@app.command("pdf-to-markdown")
def pdf_to_markdown_cmd(
    pdf_root: Path = typer.Argument(..., help="directory of local PDF source documents"),
    out_dir: Optional[Path] = typer.Argument(
        None, help="output dir of extracted .md files (default: <pdf-root>/_md)"
    ),
    min_chars: int = typer.Option(
        500, min=1, help="skip PDFs whose extracted text is shorter than this"
    ),
    parser: str = typer.Option(
        "auto", help="PDF parser: auto | pymupdf4llm | docling | marker | unstructured | markitdown"
    ),
    limit: Optional[int] = typer.Option(None, help="cap the number of PDFs to convert"),
    refresh: bool = typer.Option(
        False, "--refresh", help="reconvert every PDF even when the source is unchanged"
    ),
    repeat_blocks: str = typer.Option("keep", help=REPEAT_BLOCKS_HELP, metavar="keep|drop|anchor"),
) -> None:
    """Convert local PDFs into markdown files plus quality/citation sidecars."""
    _run_pdf_markdown_ingest(
        "pdf-to-markdown",
        pdf_root,
        out_dir,
        min_chars,
        parser,
        limit,
        refresh,
        repeat_blocks=repeat_blocks,
    )


@app.command("strip-corpus-repeats")
def strip_corpus_repeats_cmd(
    corpus: Path = typer.Option(
        ..., help="converted corpus root to census (never edited in place)"
    ),
    out: Optional[Path] = typer.Option(
        None, help="write the rewritten corpus here (required for --mode drop|anchor)"
    ),
    mode: str = typer.Option("keep", help=REPEAT_BLOCKS_HELP, metavar="keep|drop|anchor"),
    min_repeats: Optional[int] = typer.Option(
        None, help="occurrences inside ONE document before a block counts as repeated (default 3)"
    ),
    goldset: Optional[Path] = typer.Option(
        None, help="gold set whose span offsets should follow the rewrite"
    ),
    goldset_out: Optional[Path] = typer.Option(
        None, help="write the remapped gold set here (defaults to <out>/goldset.jsonl)"
    ),
    report: Optional[Path] = typer.Option(None, help="write the JSON census/rewrite report here"),
) -> None:
    """Census a converted corpus's intra-document repeated blocks, and optionally strip them.

    `--mode keep` measures only: it reports how much of the corpus repeats INSIDE one document
    (which index-time collapse hides but cannot fix at the source) and writes nothing. `drop` and
    `anchor` rewrite into `--out`, carrying the page-citation sidecars and, with `--goldset`, the
    gold spans onto the rewritten text so the same questions stay scoreable.
    """
    from llb.prep.pdf.repeats import DEFAULT_MIN_REPEATS, REPEAT_KEEP
    from llb.prep.pdf.repeat_corpus import format_repeat_report, strip_corpus_repeats

    if mode != REPEAT_KEEP and out is None:
        typer.echo(f"[error] --mode {mode} rewrites the corpus: pass --out <new-root>", err=True)
        raise typer.Exit(code=2)
    target_goldset = goldset_out
    if goldset is not None and target_goldset is None and out is not None:
        target_goldset = out / "goldset.jsonl"
    try:
        result = strip_corpus_repeats(
            corpus,
            out,
            mode=mode,
            min_repeats=min_repeats or DEFAULT_MIN_REPEATS,
            goldset=goldset,
            goldset_out=target_goldset,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    typer.echo(format_repeat_report(result))
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"[strip-corpus-repeats] wrote report -> {report}")


def _run_pdf_markdown_ingest(
    command: str,
    pdf_root: Path,
    out_dir: Optional[Path],
    min_chars: int,
    parser: str,
    limit: Optional[int],
    refresh: bool = False,
    repeat_blocks: str = "keep",
) -> None:
    from llb.prep.pdf.ingest import ingest_pdf_corpus

    try:
        result = ingest_pdf_corpus(
            pdf_root,
            out_dir,
            min_chars=min_chars,
            parser=parser,
            limit=limit,
            refresh=refresh,
            repeat_blocks=repeat_blocks,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    n_reused = sum(1 for item in result.items if item.reused)
    reused_note = f", {n_reused} reused unchanged" if n_reused else ""
    typer.echo(
        f"[{command}] {result.n_docs}/{len(result.items)} PDFs extracted "
        f"({result.n_skipped} skipped{reused_note}) -> {result.out_dir}"
    )
