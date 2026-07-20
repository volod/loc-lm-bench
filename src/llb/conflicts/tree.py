"""The semantic prefix tree: a centroid tree that blocks conflict candidates without an O(n^2) scan.

Chunks that mean similar things share a prefix path from the root, so candidate conflict pairs
come from descending the tree instead of comparing every chunk to every other chunk.

Construction is deterministic bisecting 2-means on the unit sphere (farthest-first seeding, no
RNG), splitting a node until it holds at most `leaf_size` chunks. Every node records the centroid
of its subtree and the radius: the maximum ANGULAR distance from that centroid to any member.

Pruning is exact, not heuristic. Angular distance is a metric, so for nodes N1, N2 with centroids
c1, c2 and radii r1, r2, every pair (x, y) in N1 x N2 satisfies

    theta(x, y) >= theta(c1, c2) - r1 - r2

Whenever that lower bound exceeds the query threshold, no pair under those nodes can match and the
whole cross-product is skipped. Nothing above the threshold is ever missed, so the tree returns
exactly the pairs an exhaustive scan would -- which is why `matching_pairs` can be checked against
a brute-force scan for equality rather than for approximate recall.

WHEN THAT PRUNING ACTUALLY PAYS, measured rather than assumed: at low intrinsic dimension it skips
55-92% of the pair space. At real encoder dimensionality it skips NOTHING. Over 2578 chunks of
768-dim multilingual-E5 vectors the median leaf radius is 63 degrees, so the bound only fires when
two nodes are more than 70 degrees apart -- and no two chunks in a single-language corpus ever
are. That is the curse of dimensionality, not a defect in the bound: metric-tree pruning degrades
to a full scan above roughly 30 intrinsic dimensions. The semantic tier therefore searches pairs
with `VectorSet.pairs_above` (an exact blocked matrix product, 1800x faster on that corpus) and
keeps this tree for what it is genuinely good at: a persisted, inspectable semantic hierarchy over
the corpus and the unit of incremental refresh. `matching_pairs` remains exact and tested, and is
the right path for reduced-dimension spaces.
"""

import json
from pathlib import Path
from typing import Any

from llb.conflicts.constants import DEFAULT_LEAF_SIZE
from llb.conflicts.tree_build import NodeCounter, build_node
from llb.conflicts.tree_node import TREE_VERSION, TreeNode, dot
from llb.conflicts.vectorops import VectorSet, angular_distance


class SemanticPrefixTree:
    """A persisted centroid tree over store chunk vectors."""

    def __init__(self, nodes: dict[int, TreeNode], root_id: int, leaf_size: int):
        self.nodes = nodes
        self.root_id = root_id
        self.leaf_size = leaf_size

    # --- construction -------------------------------------------------------------------------

    @classmethod
    def build(
        cls, vectors: VectorSet, *, leaf_size: int = DEFAULT_LEAF_SIZE
    ) -> "SemanticPrefixTree":
        """Bisect the whole vector set into a tree of at-most-`leaf_size` leaves."""
        if leaf_size < 1:
            raise ValueError("leaf_size must be >= 1")
        nodes: dict[int, TreeNode] = {}
        counter = NodeCounter()
        if len(vectors) == 0:
            root = TreeNode(node_id=counter.next(), members=[], centroid=[], radius=0.0)
            nodes[root.node_id] = root
            return cls(nodes, root.node_id, leaf_size)
        root_id = build_node(list(range(len(vectors))), vectors, leaf_size, nodes, counter)
        return cls(nodes, root_id, leaf_size)

    # --- querying -----------------------------------------------------------------------------

    def candidate_pairs(self, cos_threshold: float) -> list[tuple[int, int]]:
        """Every pair the tree cannot rule out at `cos_threshold`, as sorted `(low, high)`.

        A superset of the true matches: leaf cross-products are emitted whole, without checking
        each pair's own similarity. `matching_pairs` applies that final filter. Needs no vectors
        -- the stored centroids and radii carry everything the pruning bound uses.
        """
        theta = angular_distance(cos_threshold)
        pairs: set[tuple[int, int]] = set()
        stack: list[tuple[int, int]] = [(self.root_id, self.root_id)]
        while stack:
            left_id, right_id = stack.pop()
            left, right = self.nodes[left_id], self.nodes[right_id]
            if not left.members or not right.members:
                continue
            if left_id != right_id and self._prune(left, right, theta):
                continue
            if left.is_leaf and right.is_leaf:
                pairs.update(_leaf_pairs(left, right, same_node=left_id == right_id))
                continue
            stack.extend(self._descend(left, right))
        return sorted(pairs)

    def matching_pairs(
        self, vectors: VectorSet, cos_threshold: float
    ) -> list[tuple[int, int, float]]:
        """Every pair at or above `cos_threshold`, as sorted `(low, high, similarity)`.

        Identical to what a brute-force all-pairs scan returns; the tree only avoids the work.
        """
        out: list[tuple[int, int, float]] = []
        for left, right in self.candidate_pairs(cos_threshold):
            similarity = vectors.similarity(left, right)
            if similarity >= cos_threshold:
                out.append((left, right, similarity))
        return sorted(out)

    def _prune(self, left: TreeNode, right: TreeNode, theta: float) -> bool:
        """True when no member pair across these nodes can be within `theta`."""
        separation = angular_distance(dot(left.centroid, right.centroid))
        return separation - left.radius - right.radius > theta

    def _descend(self, left: TreeNode, right: TreeNode) -> list[tuple[int, int]]:
        """Split the larger node (or both, when the pair is one node against itself)."""
        if left.node_id == right.node_id:
            children = left.children
            return [
                (children[i], children[j])
                for i in range(len(children))
                for j in range(i, len(children))
            ]
        if left.is_leaf or (not right.is_leaf and len(right.members) > len(left.members)):
            return [(left.node_id, child) for child in right.children]
        return [(child, right.node_id) for child in left.children]

    # --- statistics and persistence -------------------------------------------------------------

    def leaves(self) -> list[TreeNode]:
        return [node for node in self.nodes.values() if node.is_leaf and node.members]

    def depth(self) -> int:
        return self._depth(self.root_id)

    def _depth(self, node_id: int) -> int:
        node = self.nodes[node_id]
        if node.is_leaf:
            return 1
        return 1 + max(self._depth(child) for child in node.children)

    def stats(self) -> dict[str, Any]:
        leaves = self.leaves()
        sizes = [len(leaf.members) for leaf in leaves]
        return {
            "version": TREE_VERSION,
            "n_vectors": len(self.nodes[self.root_id].members),
            "n_nodes": len(self.nodes),
            "n_leaves": len(leaves),
            "depth": self.depth() if self.nodes[self.root_id].members else 0,
            "leaf_size": self.leaf_size,
            "max_leaf": max(sizes) if sizes else 0,
        }

    def payload(self) -> dict[str, Any]:
        return {
            "version": TREE_VERSION,
            "root_id": self.root_id,
            "leaf_size": self.leaf_size,
            "nodes": [self.nodes[key].payload() for key in sorted(self.nodes)],
        }

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.payload()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "SemanticPrefixTree":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != TREE_VERSION:
            raise SystemExit(
                f"[conflicts] semantic tree at {path} has version "
                f"{payload.get('version')!r}, expected {TREE_VERSION!r}; rebuild it."
            )
        nodes = {
            int(row["node_id"]): TreeNode(
                node_id=int(row["node_id"]),
                members=[int(value) for value in row["members"]],
                centroid=[float(value) for value in row["centroid"]],
                radius=float(row["radius"]),
                children=[int(value) for value in row["children"]],
            )
            for row in payload["nodes"]
        }
        return cls(nodes, int(payload["root_id"]), int(payload["leaf_size"]))


def _leaf_pairs(left: TreeNode, right: TreeNode, *, same_node: bool) -> set[tuple[int, int]]:
    """Every unordered member pair across two leaves (or within one)."""
    if same_node:
        members = left.members
        return {
            (min(members[i], members[j]), max(members[i], members[j]))
            for i in range(len(members))
            for j in range(i + 1, len(members))
        }
    return {(min(a, b), max(a, b)) for a in left.members for b in right.members if a != b}
