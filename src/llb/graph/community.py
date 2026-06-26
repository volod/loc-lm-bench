"""Deterministic community detection for the narrative (global_community) layer.

The narrative layer needs communities, but pinning the benchmark to an external graph-analytics
dependency (igraph/leidenalg) or an abandoned graph DB breaks the "single desktop, reproducible,
minimal deps" ethos. Community detection is inherently an OFFLINE, build-time step over a static
graph, so we run it ONCE here and persist the result as a `community_id` column -- exactly the
condition under which DuckDB "covers narratives" (then `global_community` retrieval is just a
`WHERE community_id = ?`, with no graph-analytics dep at query time).

The algorithm is asynchronous label propagation made fully deterministic: nodes are processed in
sorted id order, each adopts the most frequent label among its neighbors, and ties break to the
smallest label. Async updates (a node sees already-updated neighbor labels in the same pass) damp
the oscillation plain LPA can show, the pass count is capped, and labels are finally compacted to
contiguous ids in first-appearance order -- so the same corpus always partitions identically.
"""

import logging
from collections import Counter

from llb.graph.constants import COMMUNITY_MAX_ITERS
from llb.graph.model import KnowledgeGraph

_LOG = logging.getLogger(__name__)


def detect_communities(
    adjacency: dict[int, set[int]],
    *,
    max_iters: int = COMMUNITY_MAX_ITERS,
) -> dict[int, int]:
    """Partition the nodes of `adjacency` into communities (node_id -> community_id).

    Pure + deterministic. An isolated node (no neighbors) stays its own community.
    """
    node_ids = sorted(adjacency)
    label = {nid: nid for nid in node_ids}  # each node starts in its own community

    for _ in range(max_iters):
        changed = False
        for nid in node_ids:
            neighbors = adjacency.get(nid, ())
            if not neighbors:
                continue
            counts = Counter(label[n] for n in neighbors)
            best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]  # most common, smallest
            if best != label[nid]:
                label[nid] = best
                changed = True
        if not changed:
            break

    return _compact(node_ids, label)


def _compact(node_ids: list[int], label: dict[int, int]) -> dict[int, int]:
    """Renumber labels to contiguous community ids in first-appearance order."""
    remap: dict[int, int] = {}
    out: dict[int, int] = {}
    for nid in node_ids:
        raw = label[nid]
        if raw not in remap:
            remap[raw] = len(remap)
        out[nid] = remap[raw]
    return out


def assign_communities(graph: KnowledgeGraph, *, max_iters: int = COMMUNITY_MAX_ITERS) -> int:
    """Detect communities and write `community_id` onto each node in place. Returns the count."""
    communities = detect_communities(graph.adjacency(), max_iters=max_iters)
    for node in graph.nodes:
        node.community_id = communities.get(node.node_id, node.node_id)
    n_communities = len(set(communities.values()))
    _LOG.info("[graph] detected %d communities over %d nodes", n_communities, len(graph.nodes))
    return n_communities
