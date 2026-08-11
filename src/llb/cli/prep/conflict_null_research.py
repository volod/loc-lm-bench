"""CLI for the corpus-conflict independent-null research matrix."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.conflicts.null_research_evaluation import (
    DEFAULT_MAX_GOODS_CANDIDATES,
    DEFAULT_RESEARCH_FPR,
    DEFAULT_RESEARCH_RANK_BUDGET,
    DEFAULT_TRANSFER_THRESHOLD,
)


@app.command("research-conflict-nulls")
def research_conflict_nulls_cmd(
    fixture_corpus: Path = typer.Option(..., help="planted conflict fixture corpus"),
    fixture_store: Path = typer.Option(..., help="real-embedder store for the planted fixture"),
    hr_corpus: Path = typer.Option(..., help="high-recall quickstart corpus"),
    hr_store: Path = typer.Option(..., help="store for the high-recall quickstart corpus"),
    goods_corpus: Path = typer.Option(..., help="goods quickstart corpus"),
    goods_store: Path = typer.Option(..., help="store for the goods quickstart corpus"),
    reference_corpus: Path = typer.Option(..., help="unrelated Ukrainian reference corpus"),
    reference_store: Path = typer.Option(..., help="store for the unrelated reference corpus"),
    out: Optional[Path] = typer.Option(
        None,
        help="artifact directory (default: $DATA_DIR/corpus-conflicts/null-research/<run>)",
    ),
    fpr: float = typer.Option(
        DEFAULT_RESEARCH_FPR,
        min=0.000001,
        max=0.5,
        help="nominal upper-tail probability resolved independently by each null",
    ),
    rank_budget: int = typer.Option(
        DEFAULT_RESEARCH_RANK_BUDGET,
        min=1,
        help="current candidate-budget baseline on the planted fixture",
    ),
    transfer_threshold: float = typer.Option(
        DEFAULT_TRANSFER_THRESHOLD,
        min=0.0,
        max=1.0,
        help="swept HR baseline cosine whose pairs must be recovered",
    ),
    max_goods_candidates: int = typer.Option(
        DEFAULT_MAX_GOODS_CANDIDATES,
        min=0,
        help="largest non-flooding candidate count on the goods corpus",
    ),
    permutations: int = typer.Option(
        3,
        min=1,
        help="minimum deterministic shuffles per chunk; small corpora repeat until the null "
        "tail is statistically resolved",
    ),
    seed: int = typer.Option(0, help="deterministic permutation seed"),
    embedding_device: Optional[str] = typer.Option(
        None, help="sentence-transformers device for permutation embeddings (for example cuda)"
    ),
) -> None:
    """Compare independent-null candidates against fixture and cross-corpus transfer gates."""
    from llb.conflicts.null_research import run_null_research
    from llb.conflicts.null_research_report import write_null_research
    from llb.conflicts.store_access import load_store_view
    from llb.core.paths import resolve_data_dir
    from llb.core.store_generations import generation_timestamp
    from llb.rag.embedding import Embedder

    fixture_view = load_store_view(fixture_store)
    hr_view = load_store_view(hr_store)
    goods_view = load_store_view(goods_store)
    reference_view = load_store_view(reference_store)
    models = {
        fixture_view.embedding_model,
        hr_view.embedding_model,
        goods_view.embedding_model,
        reference_view.embedding_model,
    }
    if len(models) != 1:
        raise typer.BadParameter(
            "all research stores must use the same embedding model; found "
            + ", ".join(sorted(models))
        )
    model = next(iter(models))
    embedder = Embedder(model, device=embedding_device)
    try:
        summary = run_null_research(
            fixture=(fixture_corpus, fixture_view),
            hr=(hr_corpus, hr_view),
            goods=(goods_corpus, goods_view),
            reference=(reference_corpus, reference_view),
            embed=embedder.encode_passages,
            fpr=fpr,
            rank_budget=rank_budget,
            transfer_threshold=transfer_threshold,
            max_goods_candidates=max_goods_candidates,
            permutations=permutations,
            seed=seed,
        )
    finally:
        embedder.release()
    out_dir = out or (
        resolve_data_dir() / "corpus-conflicts" / "null-research" / generation_timestamp()
    )
    paths = write_null_research(out_dir, summary)
    typer.echo(f"[conflict-null] verdict={summary['verdict']}")
    for method in summary["methods"]:
        typer.echo(
            f"[conflict-null] method={method['method']} accepted={method['gates']['accepted']}"
        )
    typer.echo(f"[conflict-null] report: {paths['report']}")
