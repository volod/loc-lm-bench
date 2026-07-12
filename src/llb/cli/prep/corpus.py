"""Corpus ingestion commands: PDF and mixed txt/md/pdf -> canonical .md corpus."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app


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
) -> None:
    """Extract local PDFs into the `.md` corpus shape used by RAG, goldset, and GraphRAG commands."""
    _run_pdf_markdown_ingest(
        "ingest-pdf-corpus", pdf_root, out_dir, min_chars, parser, limit, refresh
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
) -> None:
    """Convert local PDFs into markdown files plus quality/citation sidecars."""
    _run_pdf_markdown_ingest(
        "pdf-to-markdown", pdf_root, out_dir, min_chars, parser, limit, refresh
    )


def _run_pdf_markdown_ingest(
    command: str,
    pdf_root: Path,
    out_dir: Optional[Path],
    min_chars: int,
    parser: str,
    limit: Optional[int],
    refresh: bool = False,
) -> None:
    from llb.prep.pdf_corpus import ingest_pdf_corpus

    try:
        result = ingest_pdf_corpus(
            pdf_root,
            out_dir,
            min_chars=min_chars,
            parser=parser,
            limit=limit,
            refresh=refresh,
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
