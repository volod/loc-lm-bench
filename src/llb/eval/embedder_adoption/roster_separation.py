"""Property-separation statistics for model rosters."""

import math
from collections.abc import Mapping, Sequence

from llb.eval.embedder_adoption.roster_models import (
    NUMERIC_PROPERTIES,
    ModelProfile,
    PropertySeparation,
)


def property_separation(
    name: str,
    answer_models: Sequence[str],
    other_models: Sequence[str],
    profiles: Mapping[str, ModelProfile],
) -> PropertySeparation:
    """Test one declared property for a clean split between outcome groups."""
    missing = sorted(
        model
        for model in (*answer_models, *other_models)
        if profiles.get(model, {}).get(name) is None
    )
    inside = [
        profiles[model][name]  # type: ignore[literal-required]
        for model in answer_models
        if profiles.get(model, {}).get(name) is not None
    ]
    outside = [
        profiles[model][name]  # type: ignore[literal-required]
        for model in other_models
        if profiles.get(model, {}).get(name) is not None
    ]
    entry: PropertySeparation = {
        "property": name,
        "separates": False,
        "missing": missing,
        "reason": "",
    }
    if not inside or not outside:
        entry["reason"] = (
            f"`{name}` is undeclared for every model on one side of the split, so it cannot be "
            "tested"
        )
        return entry
    if name in NUMERIC_PROPERTIES:
        return _numeric(
            entry, [float(value) for value in inside], [float(value) for value in outside]
        )
    return _categorical(entry, [str(value) for value in inside], [str(value) for value in outside])


def _numeric(
    entry: PropertySeparation, inside: list[float], outside: list[float]
) -> PropertySeparation:
    name = entry["property"]
    entry["chance_probability"] = _threshold_chance(len(inside), len(outside))
    if max(outside) < min(inside):
        entry["separates"] = True
        entry["reason"] = (
            f"`{name}` separates upward: every capturing model is above {max(outside):g} and every "
            f"other model at or below it (capturing {_span(inside)}, other {_span(outside)}); "
            "a clean threshold split would arise by chance with probability "
            f"{entry['chance_probability']:.2f} at this roster size"
        )
    elif max(inside) < min(outside):
        entry["separates"] = True
        entry["reason"] = (
            f"`{name}` separates downward: every capturing model is below {min(outside):g} and "
            f"every other model at or above it (capturing {_span(inside)}, other "
            f"{_span(outside)}); a clean threshold split would arise by chance with probability "
            f"{entry['chance_probability']:.2f} at this roster size"
        )
    else:
        entry["reason"] = (
            f"`{name}` overlaps across the split (capturing {_span(inside)}, other "
            f"{_span(outside)}), so no threshold predicts the outcome"
        )
    return entry


def _categorical(
    entry: PropertySeparation, inside: list[str], outside: list[str]
) -> PropertySeparation:
    name = entry["property"]
    shared = sorted(set(inside) & set(outside))
    if shared:
        entry["reason"] = (
            f"`{name}` is shared across the split ({', '.join(shared)}), so it does not predict "
            "the outcome"
        )
    elif len(set(inside)) == len(inside) and len(set(outside)) == len(outside):
        entry["reason"] = (
            f"`{name}` takes a distinct value for every model ({', '.join(sorted(set(inside)))} vs "
            f"{', '.join(sorted(set(outside)))}), so the split only restates the model list and "
            "predicts nothing about a model outside the roster"
        )
    else:
        entry["separates"] = True
        entry["reason"] = (
            f"`{name}` separates: capturing models are {', '.join(sorted(set(inside)))} and every "
            f"other model is {', '.join(sorted(set(outside)))}"
        )
    return entry


def _threshold_chance(inside: int, outside: int) -> float:
    if inside <= 0 or outside <= 0:
        return 1.0
    return min(1.0, 2.0 / math.comb(inside + outside, inside))


def _span(values: Sequence[float]) -> str:
    low, high = min(values), max(values)
    return f"{low:g}" if low == high else f"{low:g}-{high:g}"
