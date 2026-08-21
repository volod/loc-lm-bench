"""Production wiring: fit the node linkage once, price every candidate cut, publish the bundle."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.graph.model import KnowledgeGraph
from llb.graph.resolution.artifacts import write_declined, write_resolution_artifacts
from llb.graph.resolution.compare import LaneItems, build_strategy_lanes, compare_strategy
from llb.graph.resolution.constants import (
    DEFAULT_THRESHOLDS,
    MAX_RESOLUTION_NODES,
    MIN_RESOLUTION_NODES,
    RESOLUTION_MODE,
)
from llb.graph.resolution.overlay import NodeOverlay, overlay_from_clusters
from llb.graph.resolution.records import build_node_spec, build_records, node_text
from llb.graph.resolution.verdict import baseline_rows, decide, threshold_rows
from llb.linkage.clustering import cluster_pairs
from llb.linkage.model import LinkageResult
from llb.rag.comparison.models import ComparisonReport, Retriever

_LOG = logging.getLogger(__name__)

_REQUIRED_PACKAGES = ("splink", "duckdb")


class NodeEmbedder:
    """Minimal embedder seam: map node texts to vectors (order-preserving)."""

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - protocol shape
        raise NotImplementedError


@dataclass(frozen=True)
class ResolutionRun:
    """One published resolution reading: what was fitted, what was priced, what it decided."""

    out_dir: Path
    summary: JsonObject
    reports: dict[str, ComparisonReport] = field(default_factory=dict)
    result: LinkageResult | None = None
    overlays: tuple[NodeOverlay, ...] = ()

    @property
    def declined(self) -> bool:
        return bool(self.summary.get("declined"))


def declined_reason(graph: KnowledgeGraph) -> str | None:
    """Why the lane is not running, in the words the summary will carry (None = it runs)."""
    missing = [name for name in _REQUIRED_PACKAGES if find_spec(name) is None]
    if missing:
        return f"the linkage extra is not installed (missing: {', '.join(missing)})"
    if len(graph.nodes) < MIN_RESOLUTION_NODES:
        return (
            f"{len(graph.nodes)} nodes are below the {MIN_RESOLUTION_NODES}-node floor a "
            "Fellegi-Sunter fit needs"
        )
    if len(graph.nodes) > MAX_RESOLUTION_NODES:
        return (
            f"{len(graph.nodes)} nodes exceed the {MAX_RESOLUTION_NODES}-node cap of the "
            "entity-type blocking rule"
        )
    return None


def _mention_vectors(
    graph: KnowledgeGraph, embedder: NodeEmbedder | None
) -> list[list[float]] | None:
    if embedder is None:
        return None
    vectors = embedder.embed([node_text(node) for node in graph.nodes])
    if not vectors or not vectors[0]:
        _LOG.warning("[graph-resolution] the embedder returned no width; scoring without cosine")
        return None
    return [list(map(float, vector)) for vector in vectors]


def _overlays(
    result: LinkageResult, record_ids: list[str], graph: KnowledgeGraph, thresholds: Sequence[float]
) -> tuple[NodeOverlay, ...]:
    """One overlay per candidate cut, re-clustered from the SAME fit (see `linkage.clustering`)."""
    return tuple(
        overlay_from_clusters(cluster_pairs(record_ids, result.pairs, threshold), graph, threshold)
        for threshold in thresholds
    )


def resolve_graph_entities(
    graph: KnowledgeGraph,
    lane_items: LaneItems,
    out_dir: Path,
    *,
    k: int,
    strategies: Sequence[str],
    khop_depth: int,
    graph_meta: JsonObject,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    embedder: NodeEmbedder | None = None,
    vector_store: Retriever | None = None,
    graph_dir: Path | None = None,
    resamples: int | None = None,
    confidence: float | None = None,
    seed: int | None = None,
) -> ResolutionRun:
    """Fit the node linkage, build an overlay per cut, and score the graph lane with and without."""
    reason = declined_reason(graph)
    if reason is not None:
        _LOG.info("[graph-resolution] not run: %s", reason)
        return _publish_declined(out_dir, graph, reason)

    from llb.linkage.engine import run_linkage

    cuts = tuple(sorted(set(thresholds), reverse=True))
    vectors = _mention_vectors(graph, embedder)
    records = build_records(graph, vectors)
    spec = build_node_spec(len(vectors[0]) if vectors else 0, min(cuts))
    result = run_linkage(records, spec)
    record_ids = [str(record["unique_id"]) for record in records]
    overlays = _overlays(result, record_ids, graph, cuts)

    reports: dict[str, ComparisonReport] = {}
    for strategy in strategies:
        lanes = build_strategy_lanes(
            graph,
            overlays,
            strategy,
            khop_depth=khop_depth,
            meta=graph_meta,
            vector_store=vector_store,
        )
        reports[strategy] = compare_strategy(
            lanes,
            lane_items,
            strategy,
            k,
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        )
    summary = _summary(graph, lane_items, k, result, overlays, reports)
    return ResolutionRun(
        out_dir=write_resolution_artifacts(
            out_dir, summary, reports, result, overlays, graph, records, graph_dir
        ),
        summary=summary,
        reports=reports,
        result=result,
        overlays=overlays,
    )


def _summary(
    graph: KnowledgeGraph,
    lane_items: LaneItems,
    k: int,
    result: LinkageResult,
    overlays: Sequence[NodeOverlay],
    reports: dict[str, ComparisonReport],
) -> JsonObject:
    return {
        "mode": RESOLUTION_MODE,
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
        "n_items": len(lane_items.items),
        "k": k,
        "linkage": {
            **result.summary(),
            # Named, not just counted: the exact-match level on `name` is UNREACHABLE by
            # construction -- the builder already keys a node on its normalized name, so two
            # records with an equal name would be one node -- and a reader should be able to see
            # that the untrained level is that one rather than a signal the fit missed.
            "untrained_levels": [
                f"{level.comparison}/{level.level}" for level in result.untrained_levels
            ],
        },
        "baseline": baseline_rows(reports),
        "thresholds": threshold_rows(reports, overlays),
        "verdict": decide(reports),
    }


def _publish_declined(out_dir: Path, graph: KnowledgeGraph, reason: str) -> ResolutionRun:
    summary: JsonObject = {
        "mode": RESOLUTION_MODE,
        "declined": True,
        "reason": reason,
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
    }
    return ResolutionRun(out_dir=write_declined(out_dir, summary), summary=summary)
