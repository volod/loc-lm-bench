"""The compound policy-change guarantee, tested against the case it exists for -- no GPU.

The audit replays two WHOLE policies, so a commit that moves several constants gets one verdict
computed between the two configurations that actually existed. On the published cells nothing tests
that, because there the compound answer and the union of the per-field ones coincide. The committed
interaction fixture is the geometry where they do NOT, and this module is the CI assertion that
keeps them apart: collapse the replay back to a field-at-a-time loop and the separating cells here
go quiet, which is exactly what fails.
"""

import json
from pathlib import Path

import pytest

from llb.bench.agentic.context_policy import (
    POLICY_COMPACT,
    ContextPolicy,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_memory_transcript import build_memory_dependent_tasks
from llb.bench.agentic_policy_change_audit import (
    AUDITED_DESIGN_PATHS,
    AUDITED_KINDS,
    KIND_INTERACTION,
    VERDICT_CHANGED,
    VERDICT_INVARIANT,
)
from llb.bench.agentic_policy_change_interaction import (
    READING_COMPOUND,
    READING_PER_FIELD,
    SEPARATES_ON_VERDICT,
    audit_interaction_design,
    separation_summary,
)
from llb.bench.agentic_policy_change_interaction_fixture import (
    INTERACTION_DESIGN_PATH,
    declared_expectations,
    interaction_cells,
    interaction_change,
    load_interaction_design,
    probe_cell_geometry,
    validate_interaction_design,
)
from llb.bench.agentic_policy_change_replay import prompt_sequence_digest, replay_episode

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def design() -> dict[str, object]:
    return load_interaction_design()


@pytest.fixture(scope="module")
def rows(design: dict[str, object]) -> list[dict[str, object]]:
    return audit_interaction_design(design)


def _cells(design: dict[str, object]) -> dict[str, dict[str, object]]:
    return {cell["cell_id"]: cell for cell in interaction_cells(design)}


def _whole_policy_digest(
    cell: dict[str, object], held: dict[str, object], settings: dict[str, object]
) -> str:
    """What the compact arm of one cell sends under one WHOLE policy, end to end."""
    policy = ContextPolicy(
        name=POLICY_COMPACT,
        observation_cap_chars=held["observation_cap_chars"],
        observation_head_share=held["observation_head_share"],
        **settings,
    )
    return prompt_sequence_digest(
        [
            prompt
            for record in build_memory_dependent_tasks(
                n_tasks=held["n_tasks"], depth=cell["depth"], pad_chars=held["pad_chars"]
            )
            for prompt in replay_episode(
                policy,
                task=AgenticTask.from_record(record),
                max_prompt_chars=cell["max_prompt_chars"],
                max_steps=cell["depth"] + held["max_steps_margin"],
            ).prompts
        ]
    )


# --- the separation -------------------------------------------------------------------------


def test_the_committed_geometry_holds_the_two_readings_apart(rows: list[dict[str, object]]):
    """The assertion the fixture exists for: a collapsed per-field audit reads this wrong."""
    summary = separation_summary(rows)
    assert summary["separates"] and summary["n_separating"] >= 1
    for row in rows:
        if not row["separates"]:
            continue
        assert SEPARATES_ON_VERDICT in row["separates_on"]
        assert row[READING_COMPOUND]["verdict"] == VERDICT_CHANGED
        assert row[READING_PER_FIELD]["verdict"] == VERDICT_INVARIANT
        # ... and neither field on its own sees anything, which is why the union misses it.
        assert all(
            field["verdict"] == VERDICT_INVARIANT
            for field in row[READING_PER_FIELD]["per_field"].values()
        )


def test_the_compound_reading_is_the_true_one_on_the_separating_cells(
    design: dict[str, object], rows: list[dict[str, object]]
):
    """Not just "the two differ": the commit really does move the prompts, and only one says so."""
    change = interaction_change(design)
    cells, held = _cells(design), design["held_fixed"]
    separating = [row for row in rows if row["separates"]]
    assert separating
    for row in separating:
        cell = cells[row["cell_id"]]
        baseline = _whole_policy_digest(
            cell, held, {"compact_share": cell["compact_share"], **change.baseline}
        )
        candidate = _whole_policy_digest(
            cell, held, {"compact_share": cell["compact_share"], **change.candidate}
        )
        assert baseline != candidate  # the per-field union's "invalidates nothing" is FALSE here


def test_the_fixture_also_carries_cells_where_the_two_readings_agree(
    rows: list[dict[str, object]],
):
    """A fixture that disagreed everywhere would be testing the change, not the interaction."""
    agreeing = [row for row in rows if not row["separates"]]
    assert agreeing
    assert {row[READING_COMPOUND]["verdict"] for row in agreeing} == {
        VERDICT_INVARIANT,
        VERDICT_CHANGED,
    }
    for row in agreeing:
        assert row[READING_COMPOUND]["verdict"] == row[READING_PER_FIELD]["verdict"]
        assert (
            row[READING_COMPOUND]["first_divergent_step"]
            == row[READING_PER_FIELD]["first_divergent_step"]
        )


def test_every_cell_reads_exactly_as_the_fixture_predeclared_it(
    design: dict[str, object], rows: list[dict[str, object]]
):
    """The design states each cell's reading up front, so a drift is a finding not a new baseline."""
    expected = declared_expectations(design)
    assert set(expected) == {row["cell_id"] for row in rows}
    for row in rows:
        want = expected[row["cell_id"]]
        compound, union = row[READING_COMPOUND], row[READING_PER_FIELD]
        assert row["separates"] is want["separates"], row["cell_id"]
        assert row["separates_on"] == want["separates_on"], row["cell_id"]
        assert compound["verdict"] == want["compound_verdict"], row["cell_id"]
        assert compound["first_divergent_step"] == want["compound_first_divergent_step"]
        assert union["verdict"] == want["per_field_union_verdict"], row["cell_id"]
        assert union["first_divergent_step"] == want["per_field_union_first_divergent_step"]
        assert {field: reading["verdict"] for field, reading in union["per_field"].items()} == want[
            "per_field_verdicts"
        ], row["cell_id"]


# --- the geometry the separation rests on ----------------------------------------------------


def test_the_predeclared_geometry_is_what_the_probe_measures(design: dict[str, object]):
    """Three numbers decide the separation, and the fixture states all three with no model."""
    change, held = interaction_change(design), design["held_fixed"]
    cells = _cells(design)
    for declared in design["cells"]:
        measured = probe_cell_geometry(cells[declared["cell_id"]], held, change)
        assert measured == declared["predeclared"], declared["cell_id"]


def test_a_cell_separates_exactly_when_the_offered_transcript_sits_between_the_two_triggers(
    design: dict[str, object], rows: list[dict[str, object]]
):
    """The mechanism, stated as an inequality: the bound's own value moves WITH the share."""
    expected = declared_expectations(design)
    for declared in design["cells"]:
        geometry = declared["predeclared"]
        offered = geometry["summary_input_chars"]
        in_band = (
            geometry["trigger_chars_at_candidate_share"]
            < offered
            <= geometry["trigger_chars_at_baseline_share"]
        )
        assert in_band is expected[declared["cell_id"]]["separates"], declared["cell_id"]
        # Both shares must fold at the SAME step, or the share alone would already read as changed.
        assert geometry["fold_step"] is not None, declared["cell_id"]


# --- the fixture contract --------------------------------------------------------------------


def test_the_fixture_is_a_geometry_and_not_a_published_study(design: dict[str, object]):
    """Nothing here is evidence, so no constant change can be asked to re-run it."""
    assert (ROOT / INTERACTION_DESIGN_PATH).is_file()
    assert INTERACTION_DESIGN_PATH not in AUDITED_DESIGN_PATHS.values()
    assert KIND_INTERACTION not in AUDITED_KINDS
    assert design["publishes_numbers"] is False


def test_a_fixture_that_cannot_separate_is_refused(design: dict[str, object]):
    """Every clause is a way the fixture could go quiet without the audit changing."""
    validate_interaction_design(design)
    with pytest.raises(ValueError, match="study_kind"):
        validate_interaction_design({**design, "study_kind": "compact_fold_step_crossover"})
    with pytest.raises(ValueError, match="must move exactly"):
        validate_interaction_design(
            {
                **design,
                "change": {"baseline": {"keep_last_n": 3}, "candidate": {"keep_last_n": 1}},
            }
        )
    with pytest.raises(ValueError, match="baselines it at"):
        validate_interaction_design(
            {**design, "held_fixed": {**design["held_fixed"], "compact_share": 0.45}}
        )
    pinned = json.loads(json.dumps(design["cells"]))
    pinned[0]["compact_share"] = 0.5
    with pytest.raises(ValueError, match="cannot audit it"):
        validate_interaction_design({**design, "cells": pinned})
    undeclared = json.loads(json.dumps(design["cells"]))
    undeclared[0].pop("predeclared")
    with pytest.raises(ValueError, match="declares no"):
        validate_interaction_design({**design, "cells": undeclared})
    agreeing = [cell for cell in design["cells"] if not cell["expected"]["separates"]]
    with pytest.raises(ValueError, match="at least one SEPARATING cell"):
        validate_interaction_design({**design, "cells": agreeing})
