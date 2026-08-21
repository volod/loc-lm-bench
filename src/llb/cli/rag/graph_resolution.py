"""Graph entity node resolution: propose an overlay and price it on the graph lane."""

from pathlib import Path
from typing import Any, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.cli.rag.compare_stores import _compare_vector_corpus_root
from llb.graph.constants import STRATEGY_GLOBAL_COMMUNITY, STRATEGY_LOCAL_KHOP
from llb.graph.resolution.constants import DEFAULT_THRESHOLDS
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
)

_DEFAULT_STRATEGIES = f"{STRATEGY_LOCAL_KHOP},{STRATEGY_GLOBAL_COMMUNITY}"
_DEFAULT_THRESHOLDS = ",".join(f"{threshold:g}" for threshold in DEFAULT_THRESHOLDS)


@app.command("resolve-graph-entities")
def resolve_graph_entities_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    k: int = typer.Option(10, help="recall@k / MRR cutoff for the paired lane rerun"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
    corpus_root: Optional[Path] = typer.Option(
        None,
        "--corpus-root",
        help="corpus the vector reference row was built over (default: the sibling corpus/ of "
        "--goldset); only read with --with-vector",
    ),
    thresholds: str = typer.Option(
        _DEFAULT_THRESHOLDS, help="comma-separated candidate match-probability cuts to price"
    ),
    strategies: str = typer.Option(
        _DEFAULT_STRATEGIES, help="comma-separated graph strategies to rerun the lane under"
    ),
    mention_embeddings: bool = typer.Option(
        True,
        "--mention-embeddings/--no-mention-embeddings",
        help="score the mention-embedding cosine with the pinned embedder (needs the [rag] extra)",
    ),
    with_vector: bool = typer.Option(
        False,
        "--with-vector",
        help="also score the built FAISS lane as a reference row (never eligible for the verdict)",
    ),
    resamples: int = typer.Option(
        DEFAULT_RESAMPLES, min=0, help="paired percentile-bootstrap resamples"
    ),
    confidence: float = typer.Option(
        DEFAULT_CONFIDENCE, min=0.5, max=0.999, help="paired bootstrap confidence level"
    ),
    seed: int = typer.Option(DEFAULT_SEED, help="paired bootstrap seed"),
    out_dir: Optional[Path] = typer.Option(
        None, help="artifact directory (default: $DATA_DIR/graph-entity-resolution/<run>/)"
    ),
) -> None:
    """Does entity fragmentation cost the graph lane recall, and would merging buy it back?

    Links the built graph's node table on name distance, surface-form intersection, entity type,
    co-occurring document ids, and mention-embedding cosine; clusters the survivors at every
    candidate cut; and reruns the graph lane over the SAME items at the same seed with and
    without each overlay. The overlay is written BESIDE the graph -- this command rewrites no
    stored graph, and a cut that lifts no lane metric is recorded as a negative result.
    """
    from llb.executor.cases import spans_as_dicts
    from llb.goldset.schema import load_goldset
    from llb.graph.resolution.artifacts import bundle_dir
    from llb.graph.resolution.compare import LaneItems
    from llb.graph.resolution.report import format_console_summary
    from llb.graph.resolution.run import resolve_graph_entities
    from llb.graph.store import GraphStore
    from llb.rag.question_types import aligned_question_types

    cfg = load_config(
        config,
        goldset_path=goldset,
        corpus_root=_compare_vector_corpus_root(goldset, corpus_root) if with_vector else None,
    )
    cuts = _parse_thresholds(thresholds)
    lanes = _parse_strategies(strategies)
    items = load_goldset(cfg.goldset_path)
    if split:
        items = [item for item in items if item.split == split]
    if not items:
        typer.echo("[error] the gold set selection is empty", err=True)
        raise typer.Exit(code=2)
    store = GraphStore.load(cfg.graph_dir())
    published = resolve_graph_entities(
        store.graph,
        LaneItems(
            items=[(item.question, spans_as_dicts(item)) for item in items],
            item_ids=[item.id for item in items],
            slice_labels=aligned_question_types(cfg.goldset_path, [item.id for item in items]),
        ),
        out_dir or bundle_dir(cfg.data_dir),
        k=k,
        strategies=lanes,
        khop_depth=cfg.graph_khop_depth,
        graph_meta=store.meta,
        thresholds=cuts,
        embedder=_embedder(cfg) if mention_embeddings else None,
        vector_store=_vector_store(cfg) if with_vector else None,
        graph_dir=cfg.graph_dir(),
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    typer.echo(format_console_summary(published.summary))
    typer.echo(f"[resolve-graph-entities] wrote bundle -> {published.out_dir}")


def _parse_thresholds(raw: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in raw.split(",") if part.strip())
    if not values or any(not 0.0 < value <= 1.0 for value in values):
        typer.echo("[error] --thresholds must be match probabilities in (0, 1]", err=True)
        raise typer.Exit(code=2)
    return values


def _parse_strategies(raw: str) -> tuple[str, ...]:
    from llb.graph.constants import STRATEGIES

    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = [value for value in values if value not in STRATEGIES]
    if not values or unknown:
        typer.echo(f"[error] --strategies must name graph strategies from {STRATEGIES}", err=True)
        raise typer.Exit(code=2)
    return values


def _embedder(cfg: Any) -> Any:
    """The pinned RAG embedder, adapted to the node-text seam (the same one dedup reuses)."""
    from llb.prep.ontology.extraction.dedup import E5QuestionEmbedder

    return E5QuestionEmbedder(cfg.embedding_model)


def _vector_store(cfg: Any) -> Any:
    """The built FAISS lane, or a hard error naming what to build first."""
    from llb.executor.runner_retrieval import _load_store

    return _load_store(cfg.with_overrides(retrieval_backend="faiss"))
