"""Does the reranker cell's answer gain track anything an operator can read off IN ADVANCE?

Two models were enough to show that whether `bge-m3`'s first-hit-rank gain reaches the answer under
a cross-encoder reranker is MODEL-dependent -- one captured it, the other did not, on identical
reranked contexts. That leaves the operator with a coin flip. This module reads a ROSTER of finished
sweeps (three or more) and asks the next question: do the models that capture the gain share a
property the operator already knows before spending a run -- parameter count, or model family?

The test is deliberately a SEPARATION test, not a fit: a property predicts the outcome only if it
splits the capturing models from the rest with no overlap. And because a roster is a handful of
models, a clean split is not automatically evidence -- with `k` of `n` models capturing, exactly two
of the `C(n, k)` possible labelings are cleanly threshold-separable on any numeric property, so the
report quotes that chance probability beside the split rather than letting a 5-model coincidence
read as a law.

Pure and file-driven: the input is N `AdoptionBarReport`s plus a declared profile per model, so the
whole reading is unit-tested with dict reports -- no backend, no store, no GPU. It runs AFTER the
heavy sweeps, never during one.
"""

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from llb.eval.embedder_adoption.stability import RowStability

from llb.eval.embedder_adoption.cross_model import (
    READING_ANSWER,
    assert_comparable,
    cell_reading,
    model_id,
)
from llb.eval.embedder_adoption.models import DEFAULT_FOCUS_CELL, AdoptionBarReport


# Pre-readable model properties: known from the model card before any run is spent. Numeric
# properties are tested for a threshold split, categorical ones for disjoint value sets.
PROPERTY_PARAMS = "params_b"
PROPERTY_FAMILY = "family"
NUMERIC_PROPERTIES = (PROPERTY_PARAMS,)
CATEGORICAL_PROPERTIES = (PROPERTY_FAMILY,)
PROPERTIES = NUMERIC_PROPERTIES + CATEGORICAL_PROPERTIES

# A declared property separates the capturing models from the rest with no overlap.
DECISION_PROPERTY_PREDICTS = "property_predicts"
# Every declared property has a capturing and a non-capturing model sharing its value/range, so
# nothing an operator can read in advance tells them which side they are on.
DECISION_NO_PROPERTY_PREDICTS = "no_property_predicts"
# Every model in the roster reads the focus cell the same way, so there is no split to explain --
# the model-dependence the roster was assembled to explain did not reproduce.
DECISION_INSUFFICIENT_VARIATION = "insufficient_variation"


class ModelProfile(TypedDict):
    """What an operator knows about a model before spending a run on it."""

    params_b: NotRequired[float]
    family: NotRequired[str]


class RosterCell(TypedDict):
    """One cell's reading under every model in the roster."""

    label: str
    readings: dict[str, str]  # model id -> READING_*
    unanimous: bool
    answer_models: list[str]
    # How settled each reading is (`llb.eval.embedder_adoption.stability`). Measured from the run
    # bundles the sweep names, so it is absent when those bundles are no longer on disk -- the
    # reading itself never depends on it.
    stability: NotRequired[dict[str, "RowStability"]]


class PropertySeparation(TypedDict):
    """Whether ONE declared property splits the capturing models from the rest."""

    property: str
    separates: bool
    # Probability that a random labeling of the profiled models would split this cleanly; only
    # defined for numeric properties, where "cleanly" means a threshold exists.
    chance_probability: NotRequired[float]
    # Models with no declared value for this property, excluded from its test.
    missing: list[str]
    reason: str


class RosterVerdict(TypedDict):
    """Is the reranker benefit predictable in advance, and from what?"""

    decision: str
    focus_cell: str
    answer_models: list[str]
    other_models: list[str]
    separations: list[PropertySeparation]
    reason: str


class RosterReport(TypedDict):
    """The roster artifact: every model's per-cell reading plus the property reading."""

    models: list[str]
    profiles: dict[str, ModelProfile]
    baseline: str
    candidate: str
    n: int
    focus_cell: str
    verdicts: dict[str, str]  # model id -> that sweep's BarVerdict decision
    cells: list[RosterCell]
    verdict: RosterVerdict
    metadata: NotRequired[dict[str, object]]


def compare_roster(
    reports: Sequence[AdoptionBarReport],
    profiles: Mapping[str, ModelProfile] | None = None,
    *,
    focus_cell: str = DEFAULT_FOCUS_CELL,
    measure_stability: bool = True,
) -> RosterReport:
    """Read N finished sweeps and state whether a declared property predicts the focus-cell gain.

    Every report is checked against the FIRST one for comparability, so a roster mixing item sets,
    seeds, cell grids, or encoder pairs fails loudly rather than producing a meaningless split.

    `measure_stability` additionally re-reads each sweep's run bundles to mark readings that a
    looser confidence level would change. It never alters a reading or the verdict, and it degrades
    silently when the bundles are gone, so an archived roster still reports.
    """
    if len(reports) < 3:
        raise ValueError(
            "a roster reading needs at least three sweeps; use compare-adoption-models for two"
        )
    reference = reports[0]
    for other in reports[1:]:
        assert_comparable(reference, other)
    models = [model_id(report) for report in reports]
    duplicates = sorted({model for model in models if models.count(model) > 1})
    if duplicates:
        raise ValueError(
            f"the roster scored the same model more than once: {', '.join(duplicates)}"
        )
    declared = {model: dict(profiles.get(model, {})) for model in models} if profiles else {}
    cells = [
        _roster_cell(label, reports, models, measure_stability)
        for label in (cell["label"] for cell in reference["cells"])
    ]
    if not any(cell["label"] == focus_cell for cell in cells):
        raise ValueError(
            f"focus cell {focus_cell!r} is not in the swept grid "
            f"({', '.join(cell['label'] for cell in cells)})"
        )
    return {
        "models": models,
        "profiles": declared,  # type: ignore[typeddict-item]
        "baseline": reference["baseline"],
        "candidate": reference["candidate"],
        "n": len(reference["item_ids"]),
        "focus_cell": focus_cell,
        "verdicts": {
            model: report["verdict"]["decision"] for model, report in zip(models, reports)
        },
        "cells": cells,
        "verdict": decide_roster(cells, declared, focus_cell=focus_cell),  # type: ignore[arg-type]
    }


def _roster_cell(
    label: str,
    reports: Sequence[AdoptionBarReport],
    models: Sequence[str],
    measure_stability: bool = False,
) -> RosterCell:
    readings: dict[str, str] = {}
    stability: dict[str, "RowStability"] = {}
    for model, report in zip(models, reports):
        cell = next(entry for entry in report["cells"] if entry["label"] == label)
        readings[model] = cell_reading(cell)
        if measure_stability:
            measured = _measure(report, label)
            if measured is not None:
                stability[model] = measured
    answer_models = [model for model in models if readings[model] == READING_ANSWER]
    return {
        "label": label,
        "readings": readings,
        **({"stability": stability} if stability else {}),  # type: ignore[typeddict-item]
        "unanimous": len(set(readings.values())) == 1,
        "answer_models": answer_models,
    }


def _measure(report: AdoptionBarReport, label: str) -> "RowStability | None":
    """This sweep's stability for one cell, or `None` when it cannot be obtained.

    Prefers the value the SWEEP itself persisted -- it was measured from the same deltas and the
    same resample draw the published intervals came from, so it needs no bundles at all. Sweeps
    recorded before the sweep carried the annotation fall back to re-deriving it from the run
    bundles they name, at that sweep's own confidence so the two paths cannot disagree.

    Imported lazily and failure-tolerant on purpose: stability is an additive annotation, so a
    roster over archived artifacts must still produce its table and verdict.
    """
    from llb.eval.embedder_adoption.screen import cell_item_deltas
    from llb.eval.embedder_adoption.stability import row_stability

    cell = next((entry for entry in report["cells"] if entry["label"] == label), None)
    if cell is None:
        return None
    persisted = cell.get("stability")
    if persisted is not None:
        return persisted
    try:
        return row_stability(
            cell_item_deltas(report, label),
            resamples=report["resamples"],
            confidence=report["confidence"],
            seed=report["seed"],
        )
    except (ValueError, OSError):
        return None


def decide_roster(
    cells: Sequence[RosterCell],
    profiles: Mapping[str, ModelProfile],
    *,
    focus_cell: str = DEFAULT_FOCUS_CELL,
) -> RosterVerdict:
    """Name the property that predicts the focus-cell answer gain, or record that none does.

    A unanimous focus cell is reported as `insufficient_variation` rather than as a prediction:
    with nothing to separate, any property would "separate" it vacuously.
    """
    focus = next(cell for cell in cells if cell["label"] == focus_cell)
    answer_models = list(focus["answer_models"])
    other_models = [model for model in focus["readings"] if model not in answer_models]
    verdict: RosterVerdict = {
        "decision": DECISION_INSUFFICIENT_VARIATION,
        "focus_cell": focus_cell,
        "answer_models": answer_models,
        "other_models": other_models,
        "separations": [],
        "reason": "",
    }
    if not answer_models or not other_models:
        shared = READING_ANSWER if answer_models else "no answer gain"
        verdict["reason"] = (
            f"every model in the roster reads `{focus_cell}` the same way ({shared}), so there is "
            "no split for a model property to explain"
        )
        return verdict
    separations = [_separation(name, answer_models, other_models, profiles) for name in PROPERTIES]
    verdict["separations"] = separations
    predicting = [entry for entry in separations if entry["separates"]]
    if not predicting:
        verdict["decision"] = DECISION_NO_PROPERTY_PREDICTS
        verdict["reason"] = (
            f"{len(answer_models)} of {len(focus['readings'])} models turn the rank gain into an "
            f"answer on `{focus_cell}` ({', '.join(f'`{m}`' for m in answer_models)}), but no "
            f"declared property ({', '.join(PROPERTIES)}) separates them from the rest: "
            + "; ".join(entry["reason"] for entry in separations)
        )
        return verdict
    verdict["decision"] = DECISION_PROPERTY_PREDICTS
    verdict["reason"] = (
        f"{', '.join(entry['property'] for entry in predicting)} separates the models that turn "
        f"the rank gain into an answer on `{focus_cell}` from the rest: "
        + "; ".join(entry["reason"] for entry in predicting)
    )
    return verdict


def _separation(
    name: str,
    answer_models: Sequence[str],
    other_models: Sequence[str],
    profiles: Mapping[str, ModelProfile],
) -> PropertySeparation:
    """Test ONE property for a clean split between the capturing models and the rest."""
    missing = sorted(
        model
        for model in (*answer_models, *other_models)
        if profiles.get(model, {}).get(name) is None
    )
    inside = [profiles[m][name] for m in answer_models if profiles.get(m, {}).get(name) is not None]  # type: ignore[literal-required]
    outside = [
        profiles[m][name]  # type: ignore[literal-required]
        for m in other_models
        if profiles.get(m, {}).get(name) is not None
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
        return _numeric_separation(entry, [float(v) for v in inside], [float(v) for v in outside])
    return _categorical_separation(entry, [str(v) for v in inside], [str(v) for v in outside])


def _numeric_separation(
    entry: PropertySeparation, inside: list[float], outside: list[float]
) -> PropertySeparation:
    """A numeric property separates when a THRESHOLD splits the two groups without overlap."""
    name = entry["property"]
    entry["chance_probability"] = _threshold_chance(len(inside), len(outside))
    if max(outside) < min(inside):
        entry["separates"] = True
        entry["reason"] = (
            f"`{name}` separates upward: every capturing model is above {max(outside):g} and every "
            f"other model at or below it (capturing {_span(inside)}, other {_span(outside)}); "
            f"a clean threshold split would arise by chance with probability "
            f"{entry['chance_probability']:.2f} at this roster size"
        )
        return entry
    if max(inside) < min(outside):
        entry["separates"] = True
        entry["reason"] = (
            f"`{name}` separates downward: every capturing model is below {min(outside):g} and "
            f"every other model at or above it (capturing {_span(inside)}, other "
            f"{_span(outside)}); a clean threshold split would arise by chance with probability "
            f"{entry['chance_probability']:.2f} at this roster size"
        )
        return entry
    entry["reason"] = (
        f"`{name}` overlaps across the split (capturing {_span(inside)}, other {_span(outside)}), "
        "so no threshold predicts the outcome"
    )
    return entry


def _categorical_separation(
    entry: PropertySeparation, inside: list[str], outside: list[str]
) -> PropertySeparation:
    """A categorical property separates when the two groups share no value.

    With one model per value this is vacuously true, so a split that merely re-states the model
    list is refused: a property has to group models to predict anything about a NEW one.
    """
    name = entry["property"]
    shared = sorted(set(inside) & set(outside))
    if shared:
        entry["reason"] = (
            f"`{name}` is shared across the split ({', '.join(shared)}), so it does not predict "
            "the outcome"
        )
        return entry
    # No two models anywhere in the roster share a value: the "split" is just the model list
    # relabelled, and it predicts nothing about a model outside the roster.
    if len(set(inside)) == len(inside) and len(set(outside)) == len(outside):
        entry["reason"] = (
            f"`{name}` takes a distinct value for every model ({', '.join(sorted(set(inside)))} vs "
            f"{', '.join(sorted(set(outside)))}), so the split only restates the model list and "
            "predicts nothing about a model outside the roster"
        )
        return entry
    entry["separates"] = True
    entry["reason"] = (
        f"`{name}` separates: capturing models are {', '.join(sorted(set(inside)))} and every "
        f"other model is {', '.join(sorted(set(outside)))}"
    )
    return entry


def _threshold_chance(inside: int, outside: int) -> float:
    """Probability a random labeling of `inside + outside` distinct values splits on a threshold.

    Exactly two labelings are cleanly threshold-separable -- the `inside` largest values, or the
    `inside` smallest -- out of `C(n, inside)` equally likely ones. This is the number that keeps a
    five-model coincidence from reading as a law.
    """
    total = inside + outside
    if inside <= 0 or outside <= 0:
        return 1.0
    return min(1.0, 2.0 / math.comb(total, inside))


def _span(values: Sequence[float]) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}"
