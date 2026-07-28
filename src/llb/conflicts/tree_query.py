"""Leaf-pair generation and exact filtering for semantic-tree queries."""

from llb.conflicts.constants import TREE_BOUND_EPSILON
from llb.conflicts.tree_node import TreeNode
from llb.conflicts.vectorops import VectorSet


def leaf_pairs(left: TreeNode, right: TreeNode, *, same_node: bool) -> set[tuple[int, int]]:
    """Every unordered member pair across two leaves (or within one)."""
    if same_node:
        members = left.members
        return {
            (min(members[i], members[j]), max(members[i], members[j]))
            for i in range(len(members))
            for j in range(i + 1, len(members))
        }
    return {(min(a, b), max(a, b)) for a in left.members for b in right.members if a != b}


def filtered_leaf_pairs(
    left: TreeNode,
    right: TreeNode,
    *,
    same_node: bool,
    vectors: VectorSet | None,
    distance_threshold: float,
) -> set[tuple[int, int]]:
    """Leaf candidates, optionally filtered by their exact vector distance."""
    pairs = leaf_pairs(left, right, same_node=same_node)
    if vectors is None:
        return pairs
    return {
        pair for pair in pairs if vectors.distance(*pair) <= distance_threshold + TREE_BOUND_EPSILON
    }
