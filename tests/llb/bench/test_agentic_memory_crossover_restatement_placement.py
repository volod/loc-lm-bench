"""The fold-step annotation on every published crossover, checked against its own geometry.

The annotation is what the restatement compares a restated guard against, so a mis-transcribed one
is not a cosmetic defect: it reads as `a_published_crossover_moves_under_the_shipped_cap` the first
time a cell at that depth becomes bound-sensitive, blaming a bound change for a number that was
mis-stated before the bound moved. These place each committed annotation on the ladder its study
measured, and then drive each form's own rule off it.
"""

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from llb.bench.agentic_memory_crossover_restatement_design import (
    audited_designs,
    load_restatement_design,
    published_crossovers,
    validate_restatement_design,
)
from llb.bench.agentic_memory_crossover_restatement_placement import (
    prompt_sequence,
    validate_derived_placements,
)
from llb.bench.agentic_memory_crossover_restatement_reading import (
    FORM_FOLD_STEP,
    FORM_INTERPOLATED,
    FORM_PORTABLE_RATIO,
)
from llb.bench.agentic_published_value_derivation import DERIVED_FROM, derived_source_of_form
from llb.bench.agentic_memory_fold_step_ladder import fold_step_guard_interval
from llb.bench.agentic_policy_change_audit import KIND_COLLAPSE, KIND_FOLD_STEP, KIND_SURFACE

ROOT = Path(__file__).resolve().parents[3]
DESIGN_PATH = ROOT / "samples/benchmarks/agentic_compact_crossover_restatement_design.json"


def _crossovers(design: dict[str, object]) -> dict[tuple[str, int], dict[str, object]]:
    return {
        (cast(str, row["study_kind"]), int(cast(int, row["depth"]))): row
        for row in published_crossovers(design)
    }


def _mutate(kind: str, depth: int, **fields: object) -> dict[str, object]:
    """The committed design with one published crossover's fields overridden."""
    design = deepcopy(load_restatement_design(DESIGN_PATH))
    for study in cast(list[dict[str, object]], design["audited_studies"]):
        if study["study_kind"] != kind:
            continue
        for crossover in cast(list[dict[str, object]], study["published_crossovers"]):
            if int(cast(int, crossover["depth"])) == depth:
                crossover.update(fields)
    return design


def test_every_committed_annotation_is_the_step_its_own_geometry_places_it_in():
    """Each form places differently, and all six committed rows satisfy their own rule."""
    design = load_restatement_design(DESIGN_PATH)
    designs = audited_designs(design, root=ROOT)
    rows = _crossovers(design)
    assert len(rows) == 6

    for (kind, depth), row in rows.items():
        sequence = prompt_sequence(designs[kind], depth)
        low, high = fold_step_guard_interval(
            sequence, int(cast(int, row["fold_step"])), float(cast(float, row["compact_share"]))
        )
        if row["form"] == FORM_INTERPOLATED:
            # A guard sits INSIDE its step. Depth 6 is the row this check was built for: its 14160
            # lands in step 6's [13136, 14912), which is not the step it used to be annotated with.
            assert low <= cast(float, row["value"]) < high
        elif row["form"] == FORM_FOLD_STEP:
            # A boundary is the interval's exclusive upper EDGE -- the first guard folding later.
            assert cast(float, row["value"]) == high
        else:
            # A derived ratio has no guard; it names the step of the guard it DECLARES it is
            # derived from, which is read off the design rather than off a constant in the checker.
            source = derived_source_of_form(row, FORM_INTERPOLATED)
            assert row["fold_step"] == rows[(source.study_kind, source.depth)]["fold_step"]

    assert rows[(KIND_SURFACE, 6)]["fold_step"] == 6
    validate_restatement_design(design, root=ROOT)


def test_an_interpolated_guard_annotated_with_the_wrong_step_is_refused():
    """The exact defect the check exists for: the annotation that used to say step 7."""
    with pytest.raises(ValueError, match="the guard folds at step 6 instead"):
        validate_restatement_design(_mutate(KIND_SURFACE, 6, fold_step=7), root=ROOT)


def test_a_fold_step_boundary_annotated_as_a_guard_inside_its_step_is_refused():
    """A boundary and a guard are placed by opposite rules, so one cannot be checked as the other."""
    with pytest.raises(ValueError, match="EXCLUSIVE upper edge"):
        validate_restatement_design(_mutate(KIND_FOLD_STEP, 6, value=14000), root=ROOT)


def test_a_step_the_geometry_cannot_fold_at_is_refused():
    """Step 1 is reachable by a small enough guard and folds nothing, so no row may claim it."""
    with pytest.raises(ValueError, match="cannot fold at"):
        validate_restatement_design(_mutate(KIND_COLLAPSE, 6, fold_step=1), root=ROOT)


def test_a_derived_ratio_must_name_the_step_of_the_guard_it_is_derived_from():
    """Two studies' annotations are tied together, so a disagreement is refused either way round."""
    with pytest.raises(ValueError, match="one of the two annotations is wrong"):
        validate_restatement_design(_mutate(KIND_COLLAPSE, 10, fold_step=9), root=ROOT)


def test_a_ratio_declaring_the_guard_of_another_depth_is_caught_by_the_annotation_it_disagrees_with():
    """The declaration is free to name another depth, and the two studies' steps then disagree.

    Which is the check earning its keep on the new seam: an edge the design states wrongly is not
    silently followed, because the fold step the ratio publishes is the source guard's own.
    """
    mis_declared = _mutate(
        KIND_COLLAPSE,
        6,
        derived_from=[{"study_kind": KIND_SURFACE, "depth": 10, "form": FORM_INTERPOLATED}],
    )
    with pytest.raises(ValueError, match="one of the two annotations is wrong"):
        validate_restatement_design(mis_declared, root=ROOT)


def test_a_ratio_declaring_a_source_the_design_does_not_publish_is_refused():
    """The declaration is checkable, so a source nobody publishes is refused before any value is read.

    Driven by dropping the guard the committed ratio names rather than by editing the declaration:
    that is how the edge actually breaks -- a design retires a published number and the value derived
    from it keeps pointing at it.
    """
    orphan = [
        row
        for row in published_crossovers(load_restatement_design(DESIGN_PATH))
        if not (row["study_kind"] == KIND_SURFACE and row["depth"] == 6)
    ]
    with pytest.raises(ValueError, match="which this design does not publish"):
        validate_derived_placements(orphan)
    assert any(row["form"] == FORM_PORTABLE_RATIO and row["depth"] == 6 for row in orphan)


def test_a_derived_ratio_that_declares_no_source_guard_is_refused():
    """A ratio is a quotient of one guard, so a design that names none states an unrestatable number."""
    undeclared = [
        {key: value for key, value in row.items() if key != DERIVED_FROM}
        for row in published_crossovers(load_restatement_design(DESIGN_PATH))
    ]
    with pytest.raises(ValueError, match="nothing in the run can restate it"):
        validate_derived_placements(undeclared)
