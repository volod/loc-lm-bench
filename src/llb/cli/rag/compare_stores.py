"""Vector-store backend comparison (`compare-vector-stores`) plus the shared corpus helpers.

The embedder bake-off command lives beside it in `compare_embeddings.py`; both read the same
goldset-sibling corpus rule and the same on-disk size helper from here.
"""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config


@app.command("compare-vector-stores")
def compare_vector_stores_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    corpus_root: Optional[Path] = typer.Option(
        None,
        help=(
            "corpus directory to build for each backend; defaults to the sibling corpus/ of "
            "--goldset when present, else the config corpus_root"
        ),
    ),
    backends: str = typer.Option(
        "faiss,chroma,qdrant,lancedb",
        help="comma-separated vector backends to compare (each over the SAME corpus + embedder)",
    ),
    k: int = typer.Option(10, help="recall@k / MRR cutoff"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
    noise_floor: bool = typer.Option(
        False,
        "--noise-floor",
        help="also measure the MEASUREMENT FLOOR per backend (see compare-retrieval), so a "
        "backend-ranking delta smaller than the floor reads as noise rather than as a winner",
    ),
    noise_floor_replicates: Optional[int] = typer.Option(
        None, help="--noise-floor: jitter replicates per backend (default 64)"
    ),
    out: Optional[Path] = typer.Option(None, help="write the JSON comparison report here"),
) -> None:
    """platform matrix: compare vector-store backends (FAISS vs Chroma/Qdrant/LanceDB) by the source-span metric.

    Builds the SAME corpus under each backend with the SAME chunking + pinned embedder, then scores
    recall@k / MRR on the gold set -- the model-independent retrieval gate before a backend's runs
    can be compared to FAISS. Each non-FAISS backend needs its optional extra installed."""
    import json

    from llb.executor.cases import spans_as_dicts
    from llb.goldset.schema import load_goldset
    from llb.rag.compare import compare_retrieval, format_comparison
    from llb.rag.comparison_builders import build_vector_store_comparison

    cfg = load_config(
        config,
        goldset_path=goldset,
        corpus_root=_compare_vector_corpus_root(goldset, corpus_root),
    )
    items = load_goldset(cfg.goldset_path)
    if split:
        items = [it for it in items if it.split == split]
    selected = [b.strip() for b in backends.split(",") if b.strip()]
    stores = build_vector_store_comparison(cfg, selected)
    compare_items = [(it.question, spans_as_dicts(it)) for it in items]
    report = compare_retrieval(stores, compare_items, k)
    if noise_floor:
        from llb.rag.noise_floor import DEFAULT_REPLICATES, measure_noise_floor

        report["noise_floor"] = measure_noise_floor(
            stores, compare_items, k, replicates=noise_floor_replicates or DEFAULT_REPLICATES
        )
    typer.echo(format_comparison(report))
    if out is not None:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"[compare-vector-stores] wrote report -> {out}")


def _compare_vector_corpus_root(
    goldset: Optional[Path], corpus_root: Optional[Path]
) -> Optional[Path]:
    """Resolve the corpus used by compare-vector-stores without surprising config overrides."""
    if corpus_root is not None:
        return corpus_root
    if goldset is None:
        return None
    sibling = goldset.parent / "corpus"
    return sibling if sibling.exists() else None


def _dir_size_bytes(path: Path) -> int:
    """Total bytes of every file under `path` (the persisted store's on-disk footprint)."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
