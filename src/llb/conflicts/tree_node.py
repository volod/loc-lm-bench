"""The semantic prefix tree's node record and the geometry both the build and query sides need.

Kept separate from `tree.py` and `tree_build.py` so construction and querying can each import the
node without importing each other.
"""

from dataclasses import dataclass, field
from typing import Any

from llb.conflicts.vectorops import Vector, VectorSet, angular_distance

TREE_VERSION = "semantic-prefix-tree-v1"


@dataclass
class TreeNode:
    """One node: its subtree members, centroid, angular radius, and child node ids."""

    node_id: int
    members: list[int]
    centroid: Vector
    radius: float
    children: list[int] = field(default_factory=list)

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
        }


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def node_geometry(members: list[int], vectors: VectorSet) -> tuple[Vector, float]:
    """The centroid of `members` and the maximum angular distance from it to any member.

    The radius is what makes the pruning bound sound: every member is guaranteed to lie within it
    of the centroid, so two nodes can be ruled out from their centroids and radii alone.
    """
    centroid = vectors.centroid(members)
    similarities = vectors.similarity_to(centroid, members)
    radius = max((angular_distance(value) for value in similarities), default=0.0)
    return centroid, radius
