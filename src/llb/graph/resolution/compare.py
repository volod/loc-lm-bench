"""The paired graph-lane rerun: the same items, the same seed, with and without each overlay.

One `compare_retrieval` call per graph STRATEGY, each baselined on that strategy's pre-overlay
lane, so a paired reading always compares a lane against the lane it would replace. The vector
row rides along when it is built, as the reference the graph lane is read against -- never as a
lane the verdict could adopt, because this run decides an overlay, not a backend.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from llb.core.contracts.common import JsonObject
from llb.graph.model import KnowledgeGraph
from llb.graph.resolution.constants import (
    LANE_BASE_TEMPLATE,
    LANE_OVERLAY_TEMPLATE,
    LANE_VECTOR,
)
from llb.graph.resolution.overlay import NodeOverlay, apply_overlay
from llb.rag.comparison.models import CompareItem, ComparisonReport, Retriever

_LOG = logging.getLogger(__name__)


def threshold_label(threshold: float) -> str:
    """A stable, filename-safe rendering of a candidate cut (`0.95` -> `0.95`)."""
    return f"{threshold:g}"


def base_lane(strategy: str) -> str:
    return LANE_BASE_TEMPLATE.format(strategy=strategy)


def overlay_lane(strategy: str, threshold: float) -> str:
    return LANE_OVERLAY_TEMPLATE.format(strategy=strategy, threshold=threshold_label(threshold))


@dataclass(frozen=True)
class LaneItems:
    """The item set every lane is scored over -- identical across lanes by construction."""

    items: list[CompareItem]
    item_ids: list[str]
    slice_labels: list[str | None] | None = None


def build_strategy_lanes(
    graph: KnowledgeGraph,
    overlays: Sequence[NodeOverlay],
    strategy: str,
    *,
    khop_depth: int,
    meta: JsonObject,
    vector_store: Retriever | None = None,
) -> dict[str, Retriever]:
    """The pre-overlay lane, one lane per overlay, and the optional vector reference row."""
    from llb.graph.store import GraphStore

    def store(source: KnowledgeGraph) -> Retriever:
        return GraphStore(source, dict(meta), strategy=strategy, khop_depth=khop_depth)

    lanes: dict[str, Retriever] = {base_lane(strategy): store(graph)}
    for overlay in overlays:
        lanes[overlay_lane(strategy, overlay.threshold)] = store(apply_overlay(graph, overlay))
    if vector_store is not None:
        lanes[LANE_VECTOR] = vector_store
    return lanes


def compare_strategy(
    lanes: dict[str, Retriever],
    lane_items: LaneItems,
    strategy: str,
    k: int,
    *,
    resamples: int | None = None,
    confidence: float | None = None,
    seed: int | None = None,
) -> ComparisonReport:
    """Score every lane of one strategy against its pre-overlay baseline."""
    from llb.rag.comparison.run import compare_retrieval

    baseline = base_lane(strategy)
    report = compare_retrieval(
        lanes,
        lane_items.items,
        k,
        slice_labels=lane_items.slice_labels,
        item_ids=lane_items.item_ids,
        baseline=baseline,
        eligible_lanes=[lane for lane in lanes if lane != LANE_VECTOR],
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    _LOG.info(
        "[graph-resolution] %s: %d lanes over %d items -> %s",
        strategy,
        len(lanes),
        len(lane_items.items),
        report["verdict"]["decision"],
    )
    return report
