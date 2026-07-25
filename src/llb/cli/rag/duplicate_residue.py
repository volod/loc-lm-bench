"""Post-collapse duplicate-residue measurement over a built RAG store."""

import json
from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app


@app.command("measure-duplicate-residue")
def measure_duplicate_residue_cmd(
    store: Optional[Path] = typer.Option(None, help="built store directory (chunks + vectors)"),
    config: Optional[Path] = typer.Option(None, help="YAML run config naming the store instead"),
    thresholds: str = typer.Option(
        "0.999,0.99,0.95", help="comma-separated cosine bands to report"
    ),
    examples: int = typer.Option(8, help="sample pairs to print per residue kind"),
    out: Optional[Path] = typer.Option(None, help="write the JSON residue report here"),
) -> None:
    """Measure what repetition a store still holds after its duplicate collapse.

    Reads the INDEXED chunks and their stored vectors -- no embedder is loaded, so the
    measurement is cheap and does not need a GPU -- and reports, per coarser duplicate tier, how
    much more it would collapse, plus the cosine bands where the ranking can barely tell two
    surviving chunks apart. The samples are what an adopt-or-reject verdict is read from: near
    neighbours no text tier reaches, and the pairs digit masking merges (a page footer, or two
    rows of a rate table -- only the corpus can say).
    """
    from llb.cli.helpers import load_config
    from llb.rag.duplicate_residue import format_residue_report, measure_duplicate_residue
    from llb.rag.store_build import CHUNKS_FILE, META_FILE
    from llb.rag.store_io import _read_jsonl
    from llb.rag.vector_index import RAG_BACKEND_FAISS, load_vector_index
    from llb.core.store_generations import resolve_store_dir

    if store is None and config is None:
        typer.echo("[error] pass --store <dir> or --config <yaml>", err=True)
        raise typer.Exit(code=2)
    store_dir = Path(store) if store is not None else load_config(config).index_dir()
    if not (store_dir / META_FILE).is_file() and not (store_dir / "generations").is_dir():
        typer.echo(f"[error] no built store at {store_dir} (run build-index first)", err=True)
        raise typer.Exit(code=2)
    store_dir = resolve_store_dir(store_dir, META_FILE)
    meta = json.loads((store_dir / META_FILE).read_text(encoding="utf-8"))
    chunks = _read_jsonl(store_dir / CHUNKS_FILE)
    index = load_vector_index(meta.get("backend", RAG_BACKEND_FAISS), store_dir)
    vectors = getattr(index, "vectors", None)
    if vectors is None:
        typer.echo(
            f"[error] the {meta.get('backend')} index at {store_dir} does not expose its stored "
            "vectors; rebuild once with `llb build-index` to measure the embedding residue",
            err=True,
        )
        raise typer.Exit(code=2)
    report = measure_duplicate_residue(
        chunks,
        vectors(),
        store_tier=str(meta.get("duplicate_tier", "exact")),
        thresholds=tuple(float(t) for t in thresholds.split(",") if t.strip()),
        examples=examples,
    )
    typer.echo(format_residue_report(report))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"[duplicate-residue] wrote report -> {out}")
