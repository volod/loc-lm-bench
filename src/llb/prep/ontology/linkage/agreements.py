"""Reading a scored pair back in words: which agreements the probability was built from.

A pair's raw agreement vector is one integer per comparison, and an integer is not an explanation.
The fitted model carries the level labels those integers index, so the drop report can name the
agreements that drove a rejection instead of printing `{"question_vector": 3}`.
"""

from collections.abc import Sequence

from llb.core.contracts.common import JsonObject
from llb.linkage.model import LinkagePair

# A near-certain pair's probability is within a rounding error of 1.0; the weight is what still
# separates two of them, so the probability keeps enough decimals to stay readable beside it.
_PROBABILITY_ROUND = 12
_WEIGHT_ROUND = 4

_NULL_LEVEL = "is_null_level"
_LEVELS = "comparison_levels"
_OUTPUT_COLUMN = "output_column_name"
_LABEL = "label_for_charts"


def level_labels(model: JsonObject) -> dict[str, dict[int, str]]:
    """`{comparison: {agreement value: level label}}` read off the trained model.

    Splink orders a comparison's levels from most to least specific after the null level, and the
    agreement value counts UP from the catch-all, so the mapping is the reversed level order.
    """
    labels: dict[str, dict[int, str]] = {}
    for comparison in model.get("comparisons", ()):
        levels = [
            level for level in comparison.get(_LEVELS, ()) if not level.get(_NULL_LEVEL, False)
        ]
        top = len(levels) - 1
        labels[str(comparison.get(_OUTPUT_COLUMN, ""))] = {
            top - index: str(level.get(_LABEL, "")) for index, level in enumerate(levels)
        }
    return labels


def describe_agreement(pair: LinkagePair, labels: dict[str, dict[int, str]]) -> dict[str, str]:
    """One pair's agreement vector as `{comparison: level label}`, in comparison order."""
    return {
        comparison: labels.get(comparison, {}).get(value, str(value))
        for comparison, value in pair.agreement.items()
    }


def pair_payload(pair: LinkagePair, labels: dict[str, dict[int, str]]) -> JsonObject:
    """The per-pair block a drop row and a disagreement row both carry."""
    return {
        "match_probability": round(pair.match_probability, _PROBABILITY_ROUND),
        "match_weight": round(pair.match_weight, _WEIGHT_ROUND),
        "agreements": describe_agreement(pair, labels),
    }


def adjacency(pairs: Sequence[LinkagePair]) -> dict[str, list[LinkagePair]]:
    """Every scored pair indexed by both of its record ids."""
    index: dict[str, list[LinkagePair]] = {}
    for pair in pairs:
        index.setdefault(pair.left_id, []).append(pair)
        index.setdefault(pair.right_id, []).append(pair)
    return index


def pair_index(pairs: Sequence[LinkagePair]) -> dict[tuple[str, str], LinkagePair]:
    """Scored pairs keyed by their unordered record-id pair."""
    return {pair_key(pair.left_id, pair.right_id): pair for pair in pairs}


def pair_key(left: str, right: str) -> tuple[str, str]:
    """Pair identity is unordered, the same way the linkage seam reads a reviewer label."""
    return (left, right) if left <= right else (right, left)


def other_id(pair: LinkagePair, record_id: str) -> str:
    return pair.right_id if pair.left_id == record_id else pair.left_id


def best_parent(
    record_id: str, neighbours: dict[str, list[LinkagePair]], positions: dict[str, int]
) -> LinkagePair | None:
    """The highest-scoring pair against a record that PRECEDES this one in the table.

    Direction matters: the constant keeps the first of a repeated pair and drops the later one, so
    a model verdict on the same decision may only look backwards -- at the prior bundles, and at
    the earlier drafts of the same batch.
    """
    position = positions[record_id]
    earlier = [
        pair
        for pair in neighbours.get(record_id, ())
        if positions[other_id(pair, record_id)] < position
    ]
    return max(earlier, key=lambda pair: pair.match_probability, default=None)
