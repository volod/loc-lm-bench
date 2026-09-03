"""Vector-store backend comparison (`compare-vector-stores`) plus the shared comparison helpers.

The embedder bake-off command lives beside it in `compare_embeddings.py`; both read the same
goldset-sibling corpus rule and the same on-disk size helper from here, and `compare_retrieval.py`
reuses the paired-baseline resolver so every comparison lane names its incumbent the same way.
"""

from pathlib import Path
from typing import Any, Optional

import typer

from llb.cli.app import app
from llb.core.contracts.retrieval.comparison import SIDECAR_KIND_COMPARISON
from llb.rag.comparison.sidecar import write_sidecar
from llb.cli.helpers import load_config
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
)


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
    baseline: Optional[str] = typer.Option(
        None,
        help="incumbent backend every row is PAIRED against; defaults to faiss when it is in "
        "--backends, else the first selected backend",
    ),
    resamples: int = typer.Option(
        DEFAULT_RESAMPLES, min=0, help="paired percentile-bootstrap resamples for the deltas"
    ),
    confidence: float = typer.Option(
        DEFAULT_CONFIDENCE, min=0.5, max=0.999, help="paired bootstrap confidence level"
    ),
    seed: int = typer.Option(DEFAULT_SEED, help="seed for the shared bootstrap index sets"),
    out: Optional[Path] = typer.Option(None, help="write the JSON comparison report here"),
) -> None:
    """platform matrix: compare vector-store backends (FAISS vs Chroma/Qdrant/LanceDB) by the source-span metric.

    Builds the SAME corpus under each backend with the SAME chunking + pinned embedder, then scores
    recall@k / MRR on the gold set -- the model-independent retrieval gate before a backend's runs
    can be compared to FAISS. Every row carries a PAIRED delta interval against the baseline backend
    over shared resample index sets plus its win/loss/tie ledger, and the report ends in an explicit
    adopt-or-retain verdict rather than a point-estimate rank -- a backend swap is decided the same
    way an embedder swap is. Each non-FAISS backend needs its optional extra installed."""

    from llb.executor.cases import spans_as_dicts
    from llb.goldset.schema import load_goldset
    from llb.rag.comparison.run import compare_retrieval
    from llb.rag.comparison.rows import format_comparison
    from llb.rag.comparison.builders import build_vector_store_comparison
    from llb.rag.vector_store.vector_index import RAG_BACKEND_FAISS

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
    try:
        paired_baseline = resolve_paired_baseline(stores, baseline, (RAG_BACKEND_FAISS,))
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    report = compare_retrieval(
        stores,
        compare_items,
        k,
        item_ids=[it.id for it in items],
        baseline=paired_baseline,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    if noise_floor:
        from llb.rag.noise_floor.measure import DEFAULT_REPLICATES, measure_noise_floor

        report["noise_floor"] = measure_noise_floor(
            stores, compare_items, k, replicates=noise_floor_replicates or DEFAULT_REPLICATES
        )
    typer.echo(format_comparison(report))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        write_sidecar(out, SIDECAR_KIND_COMPARISON, "compare-stores", report)
        typer.echo(f"[compare-vector-stores] wrote report -> {out}")


def resolve_paired_baseline(
    stores: dict[str, Any], requested: str | None, preferred: tuple[str, ...]
) -> str:
    """Resolve the incumbent lane the paired deltas are measured against, before any retrieval.

    A named lane that was not scored is an operator error, not a fallback: silently pairing
    against a different row would answer a different question than the one that was asked.
    """
    if requested is not None:
        if requested not in stores:
            raise ValueError(
                f"paired baseline lane `{requested}` was not scored; choose one of "
                f"{', '.join(stores)}"
            )
        return requested
    return next((lane for lane in preferred if lane in stores), next(iter(stores)))


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
