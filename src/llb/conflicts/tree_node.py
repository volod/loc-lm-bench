"""The semantic prefix tree's node record and the geometry both the build and query sides need.

Kept separate from `tree.py` and `tree_build.py` so construction and querying can each import the
node without importing each other.
"""

from dataclasses import dataclass, field
from typing import Any

from llb.conflicts.vectorops import Vector, VectorSet

TREE_VERSION = "semantic-prefix-tree-v3"


@dataclass
class TreeNode:
    """One node: its subtree members, centroid, metric radius, and child node ids."""

    node_id: int
    members: list[int]
    centroid: Vector
    radius: float
    children: list[int] = field(default_factory=list)
    lower_bounds: Vector = field(default_factory=list)
    upper_bounds: Vector = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "members": self.members,
            "centroid": self.centroid,
            "radius": self.radius,
            "children": self.children,
            "lower_bounds": self.lower_bounds,
            "upper_bounds": self.upper_bounds,
        }


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def node_geometry(members: list[int], vectors: VectorSet) -> tuple[Vector, float]:
    """The centroid of `members` and maximum metric distance from it to a member.

    The radius is what makes the pruning bound sound: every member is guaranteed to lie within it
    of the centroid, so two nodes can be ruled out from their centroids and radii alone.
    """
    centroid = vectors.centroid(members)
    radius = max(vectors.distances_to(centroid, members), default=0.0)
    return centroid, radius


def node_bounds(members: list[int], vectors: VectorSet) -> tuple[Vector, Vector]:
    """Axis-aligned bounds used by the projected Euclidean tree."""
    if not members:
        return [], []
    rows = [vectors.row(member) for member in members]
    return (
        [min(column) for column in zip(*rows)],
        [max(column) for column in zip(*rows)],
    )
