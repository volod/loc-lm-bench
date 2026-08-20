"""Select ordinary or relation/document-stratified multi-hop graph paths."""

from collections.abc import Sequence

from llb.graph.model import KnowledgeGraph
from llb.prep.ontology.drafting.graph_paths import (
    iter_bridge_pair_seeds,
    iter_two_hop_seeds,
    walk_chain_paths,
    walk_two_hop_paths,
)
from llb.prep.ontology.models import MultiHopSeed
from llb.prep.ontology.drafting.path_strata import PathStratumTargets, select_stratified_paths

SpanPair = tuple[tuple[str, int, int], tuple[str, int, int]]


def _path_candidates(
    graph: KnowledgeGraph,
    *,
    bridge_fill: bool,
    excluded_span_pairs: set[SpanPair] | None,
) -> list[MultiHopSeed]:
    by_id = graph.node_by_id()
    seen_pairs = set(excluded_span_pairs or ())
    directed = list(iter_two_hop_seeds(graph, by_id, seen_pairs))
    if not bridge_fill:
        return directed
    return [*directed, *iter_bridge_pair_seeds(graph, by_id, seen_pairs)]


def select_multi_hop_paths(
    graph: KnowledgeGraph,
    *,
    max_paths: int,
    seed: int,
    bridge_fill: bool,
    stratified: bool,
    targets: PathStratumTargets,
    source_documents: Sequence[str],
    excluded_span_pairs: set[SpanPair] | None,
) -> tuple[list[MultiHopSeed], dict[str, object] | None]:
    """Select paths before drafting and return the optional strata report."""
    if not stratified:
        walk = walk_chain_paths if bridge_fill else walk_two_hop_paths
        return (
            walk(
                graph,
                max_paths=max_paths,
                seed=seed,
                excluded_span_pairs=excluded_span_pairs,
            ),
            None,
        )
    candidates = _path_candidates(
        graph,
        bridge_fill=bridge_fill,
        excluded_span_pairs=excluded_span_pairs,
    )
    return select_stratified_paths(
        candidates,
        max_paths=max_paths,
        targets=targets,
        source_documents=source_documents,
    )
