"""Policy-change evidence audit: prompt-sequence invariance over every published cell, no GPU."""

import json
from pathlib import Path

import pytest

from llb.bench.agentic.context import (
    POLICY_COMPACT,
    POLICY_OBSERVATION_CAP,
    SUMMARY_INPUT_CAP_TRIGGER,
    SUMMARY_INPUT_CAP_WINDOW,
    ContextPolicy,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_memory_cap_audit import (
    VERDICT_INVARIANT as BOUND_INVARIANT,
    VERDICT_SENSITIVE as BOUND_SENSITIVE,
    audit_design,
)
from llb.bench.agentic_memory_transcript import build_memory_dependent_tasks
from llb.bench.agentic_policy_change_audit import (
    AUDITABLE_FIELDS,
    KIND_COLLAPSE,
    KIND_FOLD_STEP,
    KIND_SURFACE,
    VERDICT_CHANGED,
    VERDICT_INVARIANT,
    VERDICT_NOT_APPLICABLE,
    audit_policy_change,
    coerce_policy_value,
    declared_geometry,
    load_audited_design,
)
from llb.bench.agentic_policy_change_replay import (
    AUDITED_POLICIES,
    prompt_sequence_digest,
    replay_prompts,
)
from llb.bench.agentic_policy_change_audit_report import (
    format_policy_change_table,
    persist_policy_change_audit,
    policy_change_summary,
)

ROOT = Path(__file__).resolve().parents[3]
DESIGNS = {
    KIND_SURFACE: ROOT / "samples/benchmarks/agentic_compact_memory_boundary_surface_design.json",
    KIND_COLLAPSE: ROOT / "samples/benchmarks/agentic_compact_trigger_guard_collapse_design.json",
    KIND_FOLD_STEP: ROOT / "samples/benchmarks/agentic_compact_fold_step_crossover_design.json",
}


def _designs() -> dict[str, dict[str, object]]:
    return {kind: load_audited_design(path) for kind, path in DESIGNS.items()}


def _task() -> AgenticTask:
    return AgenticTask.from_record(
        build_memory_dependent_tasks(n_tasks=1, depth=6, pad_chars=1200)[0]
    )


# --- the replay mechanism -------------------------------------------------------------------


def test_a_replay_is_deterministic_and_records_both_kinds_of_model_call():
    """The recorded sequence is the whole model-facing history: step prompts and summarize calls."""
    policy = ContextPolicy(name=POLICY_COMPACT, compact_share=0.5)
    geometry = {"task": _task(), "max_prompt_chars": 13136, "max_steps": 10}
    first = replay_prompts(policy, **geometry)
    assert first and prompt_sequence_digest(first) == prompt_sequence_digest(
        replay_prompts(policy, **geometry)
    )
    assert any("Стисло підсумуй" in prompt for prompt in first)  # the summarize call is recorded
    assert any("Стисло підсумуй" not in prompt for prompt in first)  # so are the step prompts


def test_the_digest_separates_sequences_that_differ_only_in_where_a_boundary_falls():
    """Length-prefixing keeps `['ab','c']` from digesting the same as `['a','bc']`."""
    assert prompt_sequence_digest(["ab", "c"]) != prompt_sequence_digest(["a", "bc"])
    assert prompt_sequence_digest([]) == prompt_sequence_digest([])


def test_an_unknown_field_and_a_no_op_change_are_both_refused():
    with pytest.raises(ValueError, match="not an auditable policy field"):
        coerce_policy_value("temperature", "0.5")
    with pytest.raises(ValueError, match="two different values"):
        audit_policy_change(_designs(), field="keep_last_n", baseline=3, candidate=3)
    assert coerce_policy_value("observation_cap_chars", "800") == 800
    assert coerce_policy_value("observation_head_share", "0.6") == 0.6
    assert coerce_policy_value("summary_input_cap", "window") == "window"
    assert set(AUDITABLE_FIELDS) <= set(ContextPolicy.__dataclass_fields__)


def test_both_arms_of_a_cell_are_replayed():
    """A published number is a compact-minus-cap delta, so a change to either arm moves it."""
    assert set(AUDITED_POLICIES) == {POLICY_OBSERVATION_CAP, POLICY_COMPACT}
    audits = audit_policy_change(
        {KIND_SURFACE: load_audited_design(DESIGNS[KIND_SURFACE])},
        field="observation_cap_chars",
        baseline=800,
        candidate=1600,
    )
    for row in audits[KIND_SURFACE]:
        assert set(row["arms"]) == set(AUDITED_POLICIES)
        assert all(arm["n_tasks"] > 0 for arm in row["arms"].values())


# --- the verdicts ---------------------------------------------------------------------------


def test_a_field_no_audited_arm_reads_invalidates_nothing():
    """`keep_last_n` only steers its own policy, which no cap-fitting cell runs."""
    audits = audit_policy_change(_designs(), field="keep_last_n", baseline=3, candidate=1)
    rows = [row for study in audits.values() for row in study]
    assert rows and all(row["verdict"] == VERDICT_INVARIANT for row in rows)
    summary = policy_change_summary(audits, field="keep_last_n", baseline=3, candidate=1)
    assert summary["n_invalidated"] == 0 and summary["studies_invalidated"] == []


def test_a_field_every_trimming_policy_reads_invalidates_every_cell():
    """The memory tasks pad past the cap, so a wider cap changes both arms from the first hit."""
    audits = audit_policy_change(
        _designs(), field="observation_cap_chars", baseline=800, candidate=1600
    )
    rows = [row for study in audits.values() for row in study]
    assert rows and all(row["verdict"] == VERDICT_CHANGED for row in rows)
    assert all(set(row["changed_arms"]) == set(AUDITED_POLICIES) for row in rows)
    # Step 1 carries no observation yet; the first trimmed one reaches the prompt at step 2.
    assert all(row["first_divergent_step"] == 2 for row in rows)


def test_a_cell_that_pins_the_field_as_its_own_axis_is_not_described_by_the_change():
    """The collapse study sweeps `compact_share`; replaying those cells elsewhere is a other cell."""
    audits = audit_policy_change(_designs(), field="compact_share", baseline=0.5, candidate=0.45)
    pinned = [row for row in audits[KIND_COLLAPSE] if row["verdict"] == VERDICT_NOT_APPLICABLE]
    assert len(pinned) == len(audits[KIND_COLLAPSE])
    assert all(row["arms"] == {} and row["first_divergent_step"] is None for row in pinned)
    # A study that inherits the field from held_fixed IS described by the change.
    assert any(row["verdict"] == VERDICT_CHANGED for row in audits[KIND_SURFACE])
    summary = policy_change_summary(audits, field="compact_share", baseline=0.5, candidate=0.45)
    assert summary["n_not_applicable"] == len(pinned)
    assert KIND_COLLAPSE not in summary["studies_invalidated"]


def test_the_geometry_records_which_fields_a_cell_pins_itself():
    collapse = declared_geometry(load_audited_design(DESIGNS[KIND_COLLAPSE]), KIND_COLLAPSE)
    surface = declared_geometry(load_audited_design(DESIGNS[KIND_SURFACE]), KIND_SURFACE)
    assert all("compact_share" in cell["pinned_fields"] for cell in collapse)
    assert all(cell["pinned_fields"] == [] for cell in surface)


def test_the_general_audit_and_the_summarize_bound_view_agree_cell_for_cell():
    """The bound audit is one USE of this mechanism, so the two must never disagree."""
    audits = audit_policy_change(
        _designs(),
        field="summary_input_cap",
        baseline=SUMMARY_INPUT_CAP_TRIGGER,
        candidate=SUMMARY_INPUT_CAP_WINDOW,
    )
    expected = {VERDICT_INVARIANT: BOUND_INVARIANT, VERDICT_CHANGED: BOUND_SENSITIVE}
    for kind, path in DESIGNS.items():
        bound_rows = audit_design(load_audited_design(path), study_kind=kind)
        assert [row["cell_id"] for row in bound_rows] == [row["cell_id"] for row in audits[kind]]
        for bound, general in zip(bound_rows, audits[kind]):
            assert bound["verdict"] == expected[general["verdict"]], bound["cell_id"]
            # And the elision diagnostic explains the verdict rather than deciding it.
            assert (bound["trigger_elided_chars"] > 0) is (general["verdict"] == VERDICT_CHANGED)


# --- the report -----------------------------------------------------------------------------


def test_the_table_names_the_re_run_scope_and_persists(tmp_path: Path):
    audits = audit_policy_change(
        _designs(),
        field="summary_input_cap",
        baseline=SUMMARY_INPUT_CAP_TRIGGER,
        candidate=SUMMARY_INPUT_CAP_WINDOW,
    )
    summary = policy_change_summary(
        audits,
        field="summary_input_cap",
        baseline=SUMMARY_INPUT_CAP_TRIGGER,
        candidate=SUMMARY_INPUT_CAP_WINDOW,
    )
    assert summary["n_invalidated"] == 4 and summary["n_prompt_invariant"] == 18
    table = format_policy_change_table(audits, summary)
    assert "re-run scope" in table and "surface-d10-g23000" in table
    # An invariant cell is never in the scope list, which is the point of the audit.
    assert "- compact_memory_boundary_surface surface-d6-g12000:" not in table

    paths = persist_policy_change_audit(
        audits, summary, data_dir=tmp_path, table=table, mirror=lambda *_: None
    )
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["config"]["summary"]["policy_field"] == "summary_input_cap"
    assert manifest["metrics"]["objective_score"] == 0.0  # something was invalidated


def test_a_change_that_invalidates_nothing_scores_a_clean_audit(tmp_path: Path):
    audits = audit_policy_change(_designs(), field="keep_last_n", baseline=3, candidate=1)
    summary = policy_change_summary(audits, field="keep_last_n", baseline=3, candidate=1)
    table = format_policy_change_table(audits, summary)
    assert "invalidates NO published number" in table and "re-run scope" not in table
    paths = persist_policy_change_audit(
        audits, summary, data_dir=tmp_path, table=table, mirror=lambda *_: None
    )
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["metrics"]["objective_score"] == 1.0
