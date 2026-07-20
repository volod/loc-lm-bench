"""Incremental semantic-prefix-tree update driven by the manifest diff.

A corpus edit touches a handful of documents, so rebuilding the whole tree wastes the work already
paid for. This applies a `ManifestDiff` in place: chunks of deleted and modified documents are
removed, chunks of added and modified documents are inserted at their nearest leaf, and centroids
and radii are recomputed ONLY along the affected root-to-leaf paths. Nodes off those paths keep
their exact geometry, so their pruning bounds stay valid without being touched.

The result is identical in behavior to a rebuild -- `matching_pairs` returns the same set either
way -- but the node ids and the exact tree shape can differ, since insertion order is not the same
as bisecting from scratch. `rebuild_recommended` flags when enough of the tree has drifted that a
fresh build is the better trade.
"""

import math
from dataclasses import dataclass

from llb.conflicts.tree import SemanticPrefixTree
from llb.conflicts.tree_build import bisect
from llb.conflicts.tree_node import TREE_VERSION, TreeNode, node_bounds, node_geometry
from llb.conflicts.vectorops import VectorSet
from llb.core.contracts.rag import ChunkRecord
from llb.rag.refresh.diff import ManifestDiff

# Rebuild instead of patching once this share of the tree's chunks has changed.
REBUILD_FRACTION = 0.5


@dataclass
class TreeRefreshResult:
    """What an incremental refresh actually did."""

    inserted: int
    removed: int
    touched_nodes: int
    rebuilt: bool

    def payload(self) -> dict[str, object]:
        return {
            "inserted": self.inserted,
            "removed": self.removed,
            "touched_nodes": self.touched_nodes,
            "rebuilt": self.rebuilt,
        }


def _parent_map(tree: SemanticPrefixTree) -> dict[int, int]:
    return {child: node.node_id for node in tree.nodes.values() for child in node.children}


def _path_to_root(node_id: int, parents: dict[int, int]) -> list[int]:
    path = [node_id]
    while node_id in parents:
        node_id = parents[node_id]
        path.append(node_id)
    return path


def _descend_to_leaf(tree: SemanticPrefixTree, vectors: VectorSet, ordinal: int) -> int:
    """The leaf whose centroid is closest to `ordinal`, following the tree's own prefix path."""
    node = tree.nodes[tree.root_id]
    while not node.is_leaf:
        children = [tree.nodes[child] for child in node.children]
        populated = [child for child in children if child.members]
        if not populated:
            break
        distances = [vectors.distances_to(child.centroid, [ordinal])[0] for child in populated]
        node = populated[min(range(len(populated)), key=lambda i: distances[i])]
    return node.node_id


def _recompute(tree: SemanticPrefixTree, node_ids: list[int], vectors: VectorSet) -> None:
    """Recompute centroid + radius for each node, and re-derive internal members from children."""
    for node_id in node_ids:
        node = tree.nodes[node_id]
        if not node.is_leaf:
            members: list[int] = []
            for child in node.children:
                members.extend(tree.nodes[child].members)
            node.members = sorted(members)
        if not node.members:
            node.centroid, node.radius = [], 0.0
            node.lower_bounds, node.upper_bounds = [], []
            continue
        node.centroid, node.radius = node_geometry(node.members, vectors)
        node.lower_bounds, node.upper_bounds = node_bounds(node.members, vectors)


def _split_if_needed(
    tree: SemanticPrefixTree, leaf_id: int, vectors: VectorSet, counter_start: int
) -> int:
    """Split an over-capacity leaf once; returns the next free node id."""
    leaf = tree.nodes[leaf_id]
    if len(leaf.members) <= tree.leaf_size:
        return counter_start
    left, right = bisect(leaf.members, vectors)
    if not left or not right:
        return counter_start
    next_id = counter_start
    for members in (left, right):
        centroid, radius = node_geometry(members, vectors)
        lower, upper = node_bounds(members, vectors)
        child = TreeNode(
            node_id=next_id,
            members=members,
            centroid=centroid,
            radius=radius,
            lower_bounds=lower,
            upper_bounds=upper,
        )
        tree.nodes[child.node_id] = child
        leaf.children.append(child.node_id)
        next_id += 1
    return next_id


def refresh_tree(
    tree: SemanticPrefixTree,
    diff: ManifestDiff,
    chunks: list[ChunkRecord],
    vectors: VectorSet,
) -> tuple[SemanticPrefixTree, TreeRefreshResult]:
    """Apply `diff` to `tree` over the CURRENT `chunks` / `vectors`, rebuilding when cheaper.

    `chunks` and `vectors` describe the refreshed store, so a chunk's ordinal is its position in
    the new build order. Documents in `diff.changed` are re-inserted from scratch; everything else
    keeps its place.
    """
    changed_docs = diff.changed
    affected = [ordinal for ordinal, chunk in enumerate(chunks) if chunk["doc_id"] in changed_docs]
    total = len(chunks)
    if not diff.has_changes:
        return tree, TreeRefreshResult(0, 0, 0, rebuilt=False)
    if total == 0 or len(affected) >= REBUILD_FRACTION * total:
        rebuilt = SemanticPrefixTree.build(vectors, leaf_size=tree.leaf_size)
        return rebuilt, TreeRefreshResult(total, 0, len(rebuilt.nodes), rebuilt=True)

    stale = {
        ordinal
        for ordinal in tree.nodes[tree.root_id].members
        if ordinal >= total or chunks[ordinal]["doc_id"] in changed_docs
    }
    parents = _parent_map(tree)
    touched: set[int] = set()

    removed = _remove_stale(tree, stale, parents, touched)
    _insert_affected(tree, affected, vectors, parents, touched)

    # Deepest first, so an internal node re-derives its members from already-updated children.
    ordered = sorted(touched, key=lambda node_id: -len(_path_to_root(node_id, parents)))
    _recompute(tree, ordered, vectors)
    return tree, TreeRefreshResult(len(affected), removed, len(touched), rebuilt=False)


def _remove_stale(
    tree: SemanticPrefixTree,
    stale: set[int],
    parents: dict[int, int],
    touched: set[int],
) -> int:
    """Drop stale ordinals from every leaf; returns how many were removed."""
    removed = 0
    for node in tree.nodes.values():
        if not node.is_leaf:
            continue
        keep = [ordinal for ordinal in node.members if ordinal not in stale]
        if len(keep) != len(node.members):
            removed += len(node.members) - len(keep)
            node.members = keep
            touched.update(_path_to_root(node.node_id, parents))
    return removed


def _insert_affected(
    tree: SemanticPrefixTree,
    affected: list[int],
    vectors: VectorSet,
    parents: dict[int, int],
    touched: set[int],
) -> None:
    """Insert each changed chunk at its nearest leaf, splitting leaves that overflow."""
    next_id = max(tree.nodes) + 1
    for ordinal in affected:
        leaf_id = _descend_to_leaf(tree, vectors, ordinal)
        leaf = tree.nodes[leaf_id]
        leaf.members = sorted([*leaf.members, ordinal])
        touched.update(_path_to_root(leaf_id, parents))
        next_id = _split_if_needed(tree, leaf_id, vectors, next_id)


def tree_meta(
    tree: SemanticPrefixTree,
    *,
    embedding_model: str,
    dim: int,
    corpus_fingerprint: str,
    doc_fingerprints: dict[str, str],
    cos_threshold: float,
) -> dict[str, object]:
    """The persisted tree sidecar: geometry stats plus the fingerprints that make it reusable.

    The embedder fingerprint is what stops a tree built under one encoder from being queried under
    another -- centroids and radii are only meaningful in the space they were computed in.
    """
    maximum_radius = max((node.radius for node in tree.nodes.values()), default=0.0)
    radius_payload: dict[str, object]
    if tree.metric == "angular":
        radius_payload = {"max_radius_rad": round(maximum_radius, 6)}
    else:
        radius_payload = {"max_radius_euclidean": round(maximum_radius, 6)}
    return {
        **tree.stats(),
        "embedding_model": embedding_model,
        "dim": dim,
        "cos_threshold": cos_threshold,
        **radius_payload,
        "corpus_fingerprint": corpus_fingerprint,
        "doc_fingerprints": dict(doc_fingerprints),
    }


def tree_is_reusable(meta: dict[str, object], embedding_model: str, dim: int) -> bool:
    """A persisted tree may only be reused under the encoder and dimension that built it.

    Centroids and radii are only meaningful in the space they were computed in, so a store
    re-embedded with a different encoder must rebuild rather than patch.
    """
    return (
        meta.get("version") == TREE_VERSION
        and meta.get("embedding_model") == embedding_model
        and _as_int(meta.get("dim")) == dim
    )


def _as_int(value: object) -> int:
    return value if isinstance(value, int) else -1


def radius_degrees(radians: float) -> float:
    """Angular radius in degrees (report-friendly)."""
    return round(math.degrees(radians), 2)
