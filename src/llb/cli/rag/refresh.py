"""Dynamic-corpus refresh command: incremental store updates + the drift report."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config

if TYPE_CHECKING:
    from llb.core.config import RunConfig
    from llb.rag.refresh.drift import RetrievalDrift
    from llb.rag.refresh.store_refresh import VectorRefreshResult

# Drift reports land under `$DATA_DIR/<REFRESH_METHOD>/<run-timestamp>/`.
REFRESH_METHOD = "refresh"


@app.command("refresh-index")
def refresh_index(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    corpus_root: Optional[Path] = typer.Option(
        None, help="corpus directory to diff against the stores (overrides the config)"
    ),
    goldset: Optional[Path] = typer.Option(
        None, help="gold set JSONL for the drift report's retrieval validation (overrides config)"
    ),
    k: int = typer.Option(10, help="recall@k cutoff for the drift report"),
    retune_threshold: Optional[float] = typer.Option(
        None,
        help="absolute recall@k / MRR delta that triggers the re-tune recommendation "
        "(default 0.05)",
    ),
    skip_graph: bool = typer.Option(
        False, "--skip-graph", help="do not refresh an existing graph store"
    ),
    graph_extraction: Optional[Path] = typer.Option(
        None,
        help="extraction.jsonl with re-extracted rows for the changed documents "
        "(graph store refresh; deletion-only refreshes need none)",
    ),
) -> None:
    """Incrementally refresh the built stores after corpus edits and emit the drift report.

    Diffs the content-hash corpus manifest against each store's indexed state, re-chunks and
    re-embeds only added/modified documents (deletions propagate to the dense, lexical, and
    graph paths), and publishes the result as a new immutable store generation. The drift report
    re-runs retrieval validation on the gold set and recommends a re-tune when the recall/MRR
    delta crosses the threshold.
    """
    from llb.core.store_generations import generation_timestamp
    from llb.rag.refresh.drift import DEFAULT_RETUNE_THRESHOLD, write_drift_report
    from llb.rag.refresh.store_refresh import refresh_vector_store

    cfg = load_config(config, corpus_root=corpus_root, goldset_path=goldset)
    timestamp = generation_timestamp()
    result = refresh_vector_store(cfg.index_dir(), cfg.corpus_root, timestamp=timestamp)
    if result.refreshed:
        typer.echo(
            f"[refresh-index] vector store: {result.diff.summary()}; "
            f"{result.n_reused} rows reused, {result.n_embedded} embedded "
            f"-> {result.generation_dir}"
        )
    else:
        typer.echo(f"[refresh-index] corpus unchanged; store at {result.source_dir} is current")

    _refresh_graph_if_present(cfg, skip_graph, graph_extraction, timestamp)

    if not result.refreshed:
        return
    threshold = retune_threshold if retune_threshold is not None else DEFAULT_RETUNE_THRESHOLD
    drift = _measure_drift_if_goldset(cfg, result, k, threshold)
    report_dir = cfg.data_dir / REFRESH_METHOD / timestamp
    _json_path, md_path = write_drift_report(report_dir, result.diff, drift, result.generation_dir)
    if drift is not None:
        typer.echo(
            f"[refresh-index] drift over {drift.n_items} gold items: "
            f"recall@{drift.k} {drift.old_recall:.3f} -> {drift.new_recall:.3f} "
            f"({drift.delta_recall:+.3f}), mrr {drift.old_mrr:.3f} -> {drift.new_mrr:.3f} "
            f"({drift.delta_mrr:+.3f})"
        )
        typer.echo(
            "[refresh-index] RE-TUNE RECOMMENDED: the delta crosses the threshold "
            f"({threshold}); re-run the tuner over the refreshed store"
            if drift.retune_recommended
            else f"[refresh-index] no re-tune needed (deltas under {threshold})"
        )
    typer.echo(f"[refresh-index] drift report -> {md_path}")


def _measure_drift_if_goldset(
    cfg: "RunConfig", result: "VectorRefreshResult", k: int, threshold: float
) -> "RetrievalDrift | None":
    """Run the old-vs-new retrieval validation when the configured gold set exists."""
    from llb.goldset.schema import load_goldset
    from llb.rag.refresh.drift import measure_drift

    goldset_path = Path(cfg.goldset_path)
    if not goldset_path.is_file():
        typer.echo(
            f"[refresh-index] gold set not found at {goldset_path}; "
            "retrieval validation skipped in the drift report"
        )
        return None
    items = load_goldset(goldset_path)
    return measure_drift(result.old_store, result.new_store, items, k=k, threshold=threshold)


def _refresh_graph_if_present(
    cfg: "RunConfig", skip_graph: bool, graph_extraction: Optional[Path], timestamp: str
) -> None:
    """Refresh the graph store when one exists (`--skip-graph` opts out)."""
    from llb.core.store_generations import resolve_store_dir
    from llb.graph.constants import META_FILE as GRAPH_META_FILE

    graph_dir = cfg.graph_dir()
    graph_live = resolve_store_dir(graph_dir, GRAPH_META_FILE)
    if skip_graph or not (graph_live / GRAPH_META_FILE).is_file():
        return
    from llb.graph.ingest import load_extractions
    from llb.graph.refresh import refresh_graph_store

    update = load_extractions(graph_extraction) if graph_extraction is not None else None
    graph_result = refresh_graph_store(
        graph_dir,
        cfg.corpus_root,
        extraction_update=update,
        timestamp=timestamp,
    )
    if graph_result.refreshed:
        store = graph_result.store
        assert store is not None
        typer.echo(
            f"[refresh-index] graph store: {graph_result.diff.summary()}; "
            f"{store.meta['n_nodes']} nodes, {store.meta['n_edges']} edges "
            f"-> {graph_result.generation_dir}"
        )
    else:
        typer.echo(f"[refresh-index] graph store at {graph_result.source_dir} is current")
