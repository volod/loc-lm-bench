"""Deterministic construction of the semantic prefix tree: bisecting 2-means, no RNG.

Split from `tree.py` along the build-versus-query seam: this module only creates nodes, `tree.py`
only queries and persists them. Seeding is farthest-first -- the member furthest from the node
centroid, then the member furthest from that one -- so the same vectors always produce the same
tree, which a persisted structure has to guarantee.
"""

from llb.conflicts.constants import SPLIT_ITERATIONS
from llb.conflicts.tree_node import TreeNode, node_geometry
from llb.conflicts.vectorops import Vector, VectorSet


class NodeCounter:
    """Hands out sequential node ids so a tree's ids are dense and build-order stable."""

    def __init__(self) -> None:
        self._value = -1

    def next(self) -> int:
        self._value += 1
        return self._value


def build_node(
    members: list[int],
    vectors: VectorSet,
    leaf_size: int,
    nodes: dict[int, TreeNode],
    counter: NodeCounter,
) -> int:
    """Create the node for `members`, splitting recursively until leaves fit `leaf_size`."""
    centroid, radius = node_geometry(members, vectors)
    node = TreeNode(node_id=counter.next(), members=members, centroid=centroid, radius=radius)
    nodes[node.node_id] = node
    if len(members) <= leaf_size:
        return node.node_id
    left, right = bisect(members, vectors)
    if not left or not right:
        # Every member sits at the same point: splitting cannot separate them, so stop here
        # rather than recursing forever. The leaf is oversized but correct.
        return node.node_id
    node.children = [
        build_node(left, vectors, leaf_size, nodes, counter),
        build_node(right, vectors, leaf_size, nodes, counter),
    ]
    return node.node_id


def bisect(members: list[int], vectors: VectorSet) -> tuple[list[int], list[int]]:
    """Deterministic 2-means split: farthest-first seeding, then `SPLIT_ITERATIONS` refinements."""
    centroid = vectors.centroid(members)
    first = members[_argmin(vectors.similarity_to(centroid, members))]
    second = members[_argmin(vectors.similarity_to(vectors.row(first), members))]
    if first == second:
        return members, []
    left_seed, right_seed = vectors.row(first), vectors.row(second)
    best: tuple[list[int], list[int]] = ([], [])
    for _ in range(SPLIT_ITERATIONS):
        left, right = _assign(members, vectors, left_seed, right_seed)
        if not left or not right:
            # A refinement pass collapsed the split; keep the last usable one.
            break
        best = (left, right)
        new_left, new_right = vectors.centroid(left), vectors.centroid(right)
        if new_left == left_seed and new_right == right_seed:
            break
        left_seed, right_seed = new_left, new_right
    return best


def _assign(
    members: list[int], vectors: VectorSet, left_seed: Vector, right_seed: Vector
) -> tuple[list[int], list[int]]:
    """Assign each member to the nearer seed; exact ties go left, keeping the split stable."""
    left_similarity = vectors.similarity_to(left_seed, members)
    right_similarity = vectors.similarity_to(right_seed, members)
    left: list[int] = []
    right: list[int] = []
    for member, left_value, right_value in zip(members, left_similarity, right_similarity):
        (left if left_value >= right_value else right).append(member)
    return left, right


def _argmin(values: list[float]) -> int:
    return min(range(len(values)), key=lambda index: values[index])
