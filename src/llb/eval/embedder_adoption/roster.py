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

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llb.rag.fusion_evidence.stability import (
        ReadingStability as RowStability,
    )

from llb.eval.embedder_adoption.cross_model import (
    READING_ANSWER,
    assert_comparable,
    cell_reading,
    model_id,
)
from llb.eval.embedder_adoption.models import DEFAULT_FOCUS_CELL, AdoptionBarReport
from llb.eval.embedder_adoption.roster_models import (
    DECISION_INSUFFICIENT_VARIATION,
    DECISION_NO_PROPERTY_PREDICTS,
    DECISION_PROPERTY_PREDICTS,
    PROPERTIES,
    ModelProfile,
    RosterCell,
    RosterReport,
    RosterVerdict,
)
from llb.eval.embedder_adoption.roster_separation import property_separation


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
        readings[model] = cell_reading(cell, report["confidence"])
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
    from llb.eval.embedder_adoption.screen_data import cell_item_deltas
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
    separations = [
        property_separation(name, answer_models, other_models, profiles) for name in PROPERTIES
    ]
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
