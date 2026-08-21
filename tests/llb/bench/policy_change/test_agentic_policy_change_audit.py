"""Policy-change evidence audit: prompt-sequence invariance over every published cell, no GPU."""

import json
from pathlib import Path
from typing import Any, cast

import pytest

from llb.bench.agentic.context_policy import (
    POLICY_COMPACT,
    POLICY_OBSERVATION_CAP,
    SUMMARY_INPUT_CAP_TRIGGER,
    SUMMARY_INPUT_CAP_WINDOW,
    ContextPolicy,
)
from llb.bench.agentic.model import STATUS_CONTEXT_OVERFLOW, AgenticTask
from llb.bench.memory.cap_audit import (
    VERDICT_INVARIANT as BOUND_INVARIANT,
    VERDICT_SENSITIVE as BOUND_SENSITIVE,
    audit_design,
)
from llb.bench.memory.transcript import build_memory_dependent_tasks
from llb.bench.policy_change.audit import (
    AUDITABLE_FIELDS,
    KIND_COLLAPSE,
    KIND_FOLD_STEP,
    KIND_SURFACE,
    VERDICT_CHANGED,
    VERDICT_INVARIANT,
    VERDICT_NOT_APPLICABLE,
    PolicyChange,
    audit_cell_prompts,
    audit_policy_change,
    coerce_policy_value,
)
from llb.bench.policy_change.geometry import declared_geometry, load_audited_design
from llb.bench.agentic.context_budget import fixed_budget
from llb.bench.agentic.episode import run_episode
from llb.bench.memory.boundary.probe import oracle_compacting_controller
from llb.bench.policy_change.replay import AUDITED_POLICIES, arm_comparison
from llb.bench.policy_change.replay_episode import (
    ReplayedEpisode,
    prompt_sequence_digest,
    replay_digest,
    replay_episode,
    replay_sequence_digest,
)
from llb.bench.policy_change.audit_report import (
    REFUSED_BYTES_NOTE,
    REFUSED_PROMPT_NOTE,
    format_invalidated_cells,
    format_policy_change_table,
    persist_policy_change_audit,
    policy_change_summary,
)

ROOT = Path(__file__).resolve().parents[4]
# The changes the tests below audit, each stated once as the whole change it is.
KEEP_CHANGE = PolicyChange.of("keep_last_n", 3, 1)
CAP_CHANGE = PolicyChange.of("observation_cap_chars", 800, 1600)
SHARE_CHANGE = PolicyChange.of("compact_share", 0.5, 0.45)
HEAD_SHARE_CHANGE = PolicyChange.of("observation_head_share", 0.6, 0.5)
BOUND_CHANGE = PolicyChange.of(
    "summary_input_cap", SUMMARY_INPUT_CAP_TRIGGER, SUMMARY_INPUT_CAP_WINDOW
)
# A guard between the step-1 prompt (3000 chars, no observation yet) and the step-2 prompt under
# either cap (3904 at 800, 4333 at 1600): every arm ends on a refused prompt the caps disagree on.
OVERFLOW_GUARD = 3500
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
    first = replay_episode(policy, **geometry)
    assert first.prompts and replay_digest(first) == replay_digest(
        replay_episode(policy, **geometry)
    )
    assert not first.refused and first.refused_prompt_chars is None  # nothing was refused here
    assert any("Стисло підсумуй" in p for p in first.prompts)  # the summarize call is recorded
    assert any("Стисло підсумуй" not in p for p in first.prompts)  # so are the step prompts


def test_the_digest_separates_sequences_that_differ_only_in_where_a_boundary_falls():
    """Length-prefixing keeps `['ab','c']` from digesting the same as `['a','bc']`."""
    assert prompt_sequence_digest(["ab", "c"]) != prompt_sequence_digest(["a", "bc"])
    assert prompt_sequence_digest([]) == prompt_sequence_digest([])


def _refused_at(**policy_fields: Any) -> ReplayedEpisode:
    """One episode behind the overflow guard, under an `observation_cap` policy stated by field."""
    return replay_episode(
        ContextPolicy(name=POLICY_OBSERVATION_CAP, **policy_fields),
        task=_task(),
        max_prompt_chars=OVERFLOW_GUARD,
        max_steps=10,
    )


def test_a_replay_records_the_prompt_the_guard_refused():
    """The loop prices the overflowing prompt and never sends it, so no other seam can see it."""
    # 3500 fits step 1 (3000 chars) and refuses step 2, the first prompt carrying an observation.
    record = _refused_at(observation_cap_chars=800)
    assert record.status == STATUS_CONTEXT_OVERFLOW and record.refused
    assert [len(prompt) for prompt in record.prompts] == [3000]
    assert record.refused_prompt_chars == 3904 > OVERFLOW_GUARD
    # The observer hands over the prompt itself, and the priced size is that prompt's own length
    # because this replay goes through `complete` rather than a serialized controller channel.
    assert record.refused_prompt is not None
    assert len(record.refused_prompt) == record.refused_prompt_chars
    assert record.refused_prompt not in record.prompts  # refused means never sent

    # Same episode under a wider cap: identical sent prompts, a bigger refusal, another digest.
    wider = _refused_at(observation_cap_chars=1600)
    assert prompt_sequence_digest(wider.prompts) == prompt_sequence_digest(record.prompts)
    assert wider.refused_prompt_chars == 4333
    assert replay_digest(wider) != replay_digest(record)


def test_the_refused_prompt_is_compared_by_bytes_and_not_by_size():
    """`observation_head_share` re-splits a trimmed observation without changing its length.

    So a size-only refusal record is blind to exactly the field whose whole effect is where the
    bytes went, and these two episodes -- same sent prompts, same 3904-char refusal -- would read
    as one.
    """
    sides = [_refused_at(observation_cap_chars=800, observation_head_share=h) for h in (0.6, 0.5)]
    assert prompt_sequence_digest(sides[0].prompts) == prompt_sequence_digest(sides[1].prompts)
    assert sides[0].refused_prompt_chars == sides[1].refused_prompt_chars == 3904
    assert sides[0].refused_prompt != sides[1].refused_prompt
    assert replay_digest(sides[0]) != replay_digest(sides[1])


def test_an_episode_that_sends_everything_it_builds_never_fires_the_refusal_observer():
    """The seam is inert on a fitting run, so it cannot perturb any measured episode."""
    seen: list[str] = []
    episode = run_episode(
        _task(),
        oracle_compacting_controller,
        max_steps=10,
        policy=ContextPolicy(name=POLICY_COMPACT, compact_share=0.5),
        budget=fixed_budget(13136),
        on_refused_prompt=seen.append,
    )
    assert episode.status != STATUS_CONTEXT_OVERFLOW and seen == []


def test_a_change_that_moves_only_an_overflowing_prompt_no_longer_reads_as_invariant():
    """The audit's own blind spot, closed: an episode that ends on the prompt the change moved.

    Both arms overflow at step 2 under both caps, so every prompt a model saw is byte-identical
    and the pre-refusal audit called the cell prompt-invariant -- for a change that moved the very
    prompt that ended the run. The compact arm here really is invariant (its fold at this guard
    discards the whole transcript, which is cap-independent), so the refusal is the ONLY signal.
    """
    cell = {
        "cell_id": "overflow-at-step-2",
        "depth": 6,
        "compact_share": 0.5,
        "max_prompt_chars": OVERFLOW_GUARD,
        "pinned_fields": [],
    }
    _, held = _surface_cell()
    row = audit_cell_prompts(cell, held, CAP_CHANGE)
    assert row["changed_arms"] == [POLICY_OBSERVATION_CAP]
    assert row["verdict"] == VERDICT_CHANGED and row["refused_prompt_only"]

    cap_arm = row["arms"][POLICY_OBSERVATION_CAP]
    assert cap_arm["sent_identical"] and cap_arm["refused_prompt_moved"]
    assert cap_arm["baseline_refused_tasks"] == cap_arm["candidate_refused_tasks"]
    assert cap_arm["baseline_refused_tasks"] == cap_arm["n_tasks"] == held["n_tasks"]
    # The refused prompt sits one call past the last one either replay made.
    assert cap_arm["first_divergent_step"] == row["first_divergent_step"] == 2

    # The compact arm neither refuses nor moves, which is what makes this cell the sharp case.
    compact_arm = row["arms"][POLICY_COMPACT]
    assert compact_arm["identical"] and compact_arm["baseline_refused_tasks"] == 0

    # And the re-run scope says so rather than pointing at a prompt nobody sent. The two caps
    # price the refusal differently, so this is the plain note.
    summary = policy_change_summary({KIND_SURFACE: [row]}, CAP_CHANGE)
    assert not row["refused_prompt_bytes_only"]
    assert REFUSED_PROMPT_NOTE in "\n".join(format_invalidated_cells(summary))


def test_a_change_that_moves_an_overflowing_prompt_without_resizing_it_is_still_read():
    """The same cell under the one audited field that moves bytes and never a prompt LENGTH.

    Both head shares send `[3000]` and are refused a 3904-char prompt, so every size the audit
    could compare is equal; only the refused prompt's own bytes separate them.
    """
    cell = {
        "cell_id": "overflow-at-step-2",
        "depth": 6,
        "compact_share": 0.5,
        "max_prompt_chars": OVERFLOW_GUARD,
        "pinned_fields": [],
    }
    _, held = _surface_cell()
    row = audit_cell_prompts(cell, held, HEAD_SHARE_CHANGE)
    assert row["verdict"] == VERDICT_CHANGED and row["changed_arms"] == [POLICY_OBSERVATION_CAP]
    assert row["refused_prompt_only"] and row["refused_prompt_bytes_only"]

    cap_arm = row["arms"][POLICY_OBSERVATION_CAP]
    assert cap_arm["sent_identical"] and cap_arm["refused_prompt_moved_bytes_only"]
    assert cap_arm["baseline_refused_tasks"] == cap_arm["candidate_refused_tasks"] == 7

    # The scope line says the sizes agree, so nobody reads the equal counts as an equal prompt.
    summary = policy_change_summary({KIND_SURFACE: [row]}, HEAD_SHARE_CHANGE)
    assert REFUSED_BYTES_NOTE in "\n".join(format_invalidated_cells(summary))


def test_no_published_cell_ends_on_a_refused_prompt_under_the_measured_policy():
    """Cap-fitting cells are CHOSEN not to overflow -- now asserted rather than assumed.

    Asserted on the BASELINE side, which is the configuration the published numbers were measured
    under: with no refusal there, every recorded verdict is decided by sent prompts alone, which is
    why the recorded table is unchanged by the refusal recording.
    """
    for change in (CAP_CHANGE, BOUND_CHANGE, KEEP_CHANGE):
        rows = [row for study in audit_policy_change(_designs(), change).values() for row in study]
        arms = [arm for row in rows for arm in cast(dict, row["arms"]).values()]
        assert arms and all(arm["baseline_refused_tasks"] == 0 for arm in arms), change.label
        assert not any(row["refused_prompt_only"] for row in rows), change.label


def test_a_candidate_value_that_stops_a_published_cell_fitting_its_guard_is_recorded():
    """A widened cap does not merely move `surface-d10-g14000`'s cap arm -- it overflows it.

    At guard 14000 the cap arm peaks at 11926 chars under the pinned 800 and is refused a 14621-
    char prompt at step 9 under 1600. The cell already read `prompts_change` from its sent prompts
    (the trim reaches the prompt at model call 2), so no recorded verdict moves; what is new is
    that the audit can now SAY the candidate leaves the cell un-runnable at its published guard.
    """
    rows = audit_policy_change(
        {KIND_SURFACE: load_audited_design(DESIGNS[KIND_SURFACE])}, CAP_CHANGE
    )
    refused = {
        cast(str, row["cell_id"]): arm
        for row in rows[KIND_SURFACE]
        for arm in cast(dict, row["arms"]).values()
        if arm["candidate_refused_tasks"]
    }
    assert list(refused) == ["surface-d10-g14000"]
    arm = refused["surface-d10-g14000"]
    assert arm["policy"] == POLICY_OBSERVATION_CAP
    assert arm["baseline_refused_tasks"] == 0
    assert arm["candidate_refused_tasks"] == arm["n_tasks"]  # every task, not one unlucky one

    # And the re-run scope says re-measuring it needs a new guard, not a re-run at the old one.
    summary = policy_change_summary(rows, CAP_CHANGE)
    assert summary["n_candidate_overflow"] == 1
    scope = format_invalidated_cells(summary)
    named = [line for line in scope if "no longer fits this guard" in line]
    assert len(named) == 1 and "surface-d10-g14000" in named[0]

    # Still reported at the sent-prompt divergence, which comes first.
    assert not arm["sent_identical"] and arm["first_divergent_step"] == 2


def test_an_unknown_field_and_a_no_op_change_are_both_refused():
    with pytest.raises(ValueError, match="not an auditable policy field"):
        coerce_policy_value("temperature", "0.5")
    with pytest.raises(ValueError, match="not an auditable policy field"):
        PolicyChange.of("temperature", 0.0, 0.5)
    with pytest.raises(ValueError, match="two different values"):
        PolicyChange.of("keep_last_n", 3, 3)
    with pytest.raises(ValueError, match="at least one auditable field"):
        PolicyChange(baseline={}, candidate={})
    with pytest.raises(ValueError, match="the same fields"):
        PolicyChange(baseline={"keep_last_n": 3}, candidate={"compact_keep_recent": 2})
    assert coerce_policy_value("observation_cap_chars", "800") == 800
    assert coerce_policy_value("observation_head_share", "0.6") == 0.6
    assert coerce_policy_value("summary_input_cap", "window") == "window"
    assert set(AUDITABLE_FIELDS) <= set(ContextPolicy.__dataclass_fields__)


def test_both_arms_of_a_cell_are_replayed():
    """A published number is a compact-minus-cap delta, so a change to either arm moves it."""
    assert set(AUDITED_POLICIES) == {POLICY_OBSERVATION_CAP, POLICY_COMPACT}
    audits = audit_policy_change(
        {KIND_SURFACE: load_audited_design(DESIGNS[KIND_SURFACE])},
        PolicyChange.of("observation_cap_chars", 800, 1600),
    )
    for row in audits[KIND_SURFACE]:
        assert set(row["arms"]) == set(AUDITED_POLICIES)
        assert all(arm["n_tasks"] > 0 for arm in row["arms"].values())


# --- the verdicts ---------------------------------------------------------------------------


def test_a_field_no_audited_arm_reads_invalidates_nothing_on_cap_fitting_cells():
    """`keep_last_n` only steers its own policy, which no cap-fitting cell runs."""
    audits = audit_policy_change(_designs(), KEEP_CHANGE)
    rows = [row for study in audits.values() for row in study]
    assert rows and all(row["verdict"] == VERDICT_INVARIANT for row in rows)
    summary = policy_change_summary(audits, KEEP_CHANGE)
    assert summary["n_invalidated"] == 0 and summary["studies_invalidated"] == []


def test_keep_last_n_invalidates_the_lanes_that_actually_run_that_policy():
    """The gap the wider registry closes: keep=1 looked free only while the audit skipped keep cells."""
    from llb.bench.policy_change.audit import KIND_CONSTANT_SWEEP, KIND_KEEP_LONG
    from llb.bench.policy_change.geometry import load_audited_designs

    audits = audit_policy_change(load_audited_designs(), KEEP_CHANGE)
    summary = policy_change_summary(audits, KEEP_CHANGE)
    assert summary["n_cells"] == 27 and summary["n_invalidated"] == 2
    assert summary["studies_invalidated"] == [KIND_CONSTANT_SWEEP, KIND_KEEP_LONG]
    changed = {
        (cast(str, row["study_kind"]), cast(str, row["cell_id"]))
        for row in cast(list[dict[str, object]], summary["invalidated"])
    }
    assert changed == {
        (KIND_CONSTANT_SWEEP, "sweep-keep-shipped"),
        (KIND_KEEP_LONG, "keep-long-shipped"),
    }
    # Cap-fitting and harness seed rows stay invariant -- they never apply keep_last_n.
    assert summary["n_prompt_invariant"] == 25


def test_a_field_every_trimming_policy_reads_invalidates_every_cell():
    """The memory tasks pad past the cap, so a wider cap changes both arms from the first hit."""
    audits = audit_policy_change(_designs(), CAP_CHANGE)
    rows = [row for study in audits.values() for row in study]
    assert rows and all(row["verdict"] == VERDICT_CHANGED for row in rows)
    assert all(set(row["changed_arms"]) == set(AUDITED_POLICIES) for row in rows)
    # Step 1 carries no observation yet; the first trimmed one reaches the prompt at step 2.
    assert all(row["first_divergent_step"] == 2 for row in rows)


def test_a_cell_that_pins_the_field_as_its_own_axis_is_not_described_by_the_change():
    """The collapse study sweeps `compact_share`; replaying those cells elsewhere is a other cell."""
    audits = audit_policy_change(_designs(), SHARE_CHANGE)
    pinned = [row for row in audits[KIND_COLLAPSE] if row["verdict"] == VERDICT_NOT_APPLICABLE]
    assert len(pinned) == len(audits[KIND_COLLAPSE])
    assert all(row["arms"] == {} and row["first_divergent_step"] is None for row in pinned)
    # A study that inherits the field from held_fixed IS described by the change.
    assert any(row["verdict"] == VERDICT_CHANGED for row in audits[KIND_SURFACE])
    summary = policy_change_summary(audits, SHARE_CHANGE)
    assert summary["n_not_applicable"] == len(pinned)
    assert KIND_COLLAPSE not in summary["studies_invalidated"]


# --- a compound change is ONE change ---------------------------------------------------------


def _surface_cell() -> tuple[dict[str, object], dict[str, object]]:
    """One published cell plus its study's held settings -- the smallest replayable geometry."""
    design = load_audited_design(DESIGNS[KIND_SURFACE])
    return declared_geometry(design, KIND_SURFACE)[0], design["held_fixed"]


def test_a_compound_arm_replays_a_whole_policy_rather_than_one_overridden_field():
    """The candidate arm IS the shipped configuration, not "new cap + whatever ships elsewhere"."""
    cell, held = _surface_cell()
    compound = arm_comparison(
        POLICY_COMPACT,
        cell,
        held,
        {"observation_cap_chars": 800, "compact_keep_recent": 1},
        {"observation_cap_chars": 1600, "compact_keep_recent": 2},
    )
    tasks = [
        AgenticTask.from_record(record)
        for record in build_memory_dependent_tasks(
            n_tasks=held["n_tasks"], depth=cell["depth"], pad_chars=held["pad_chars"]
        )
    ]
    expected = replay_sequence_digest(
        [
            replay_episode(
                ContextPolicy(
                    name=POLICY_COMPACT,
                    observation_cap_chars=1600,
                    compact_keep_recent=2,
                    observation_head_share=held["observation_head_share"],
                    compact_share=cell["compact_share"],
                ),
                task=task,
                max_prompt_chars=cell["max_prompt_chars"],
                max_steps=cell["depth"] + held["max_steps_margin"],
            )
            for task in tasks
        ]
    )
    assert compound["candidate_digest"] == expected
    # And auditing the cap ALONE compares a different configuration -- the bug this closes.
    single = arm_comparison(
        POLICY_COMPACT, cell, held, {"observation_cap_chars": 800}, {"observation_cap_chars": 1600}
    )
    assert single["candidate_digest"] != compound["candidate_digest"]


def test_a_restated_pin_on_a_held_field_feeds_the_baseline_arm():
    """A restated pin on observation_cap_chars beats the design's stale held value on untouched fields.

    The change moves only compact_keep_recent, so the cap is untouched. Without pins the baseline
    arm would replay the design's stale 400; with pins it replays the pinned 800 -- the same class
    of bug the compound audit closed, one level down.
    """
    cell, held = _surface_cell()
    stale = {**cast(dict[str, object], held), "observation_cap_chars": 400}
    pins = {
        "observation_cap_chars": 800,
        "observation_head_share": held["observation_head_share"],
        "keep_last_n": 3,
        "compact_share": 0.5,
        "compact_keep_recent": 1,
        "summary_input_cap": "window",
    }
    baseline, candidate = {"compact_keep_recent": 1}, {"compact_keep_recent": 2}
    from_design = arm_comparison(POLICY_COMPACT, cell, stale, baseline, candidate)
    from_pins = arm_comparison(POLICY_COMPACT, cell, stale, baseline, candidate, pinned=pins)
    assert from_design["baseline_digest"] != from_pins["baseline_digest"]
    # The pin-fed baseline is exactly the episode under the pinned cap, not the stale held one.
    tasks = [
        AgenticTask.from_record(record)
        for record in build_memory_dependent_tasks(
            n_tasks=held["n_tasks"], depth=cell["depth"], pad_chars=held["pad_chars"]
        )
    ]
    expected = replay_sequence_digest(
        [
            replay_episode(
                ContextPolicy(
                    name=POLICY_COMPACT,
                    observation_cap_chars=800,
                    compact_keep_recent=1,
                    observation_head_share=held["observation_head_share"],
                    compact_share=cell["compact_share"],
                ),
                task=task,
                max_prompt_chars=cell["max_prompt_chars"],
                max_steps=cell["depth"] + held["max_steps_margin"],
            )
            for task in tasks
        ]
    )
    assert from_pins["baseline_digest"] == expected
    # A moved field still beats a conflicting pin: settings win over the restated map.
    moved = arm_comparison(
        POLICY_COMPACT,
        cell,
        held,
        {"observation_cap_chars": 800},
        {"observation_cap_chars": 1600},
        pinned={**pins, "observation_cap_chars": 400},
    )
    assert (
        moved["baseline_digest"]
        == arm_comparison(
            POLICY_COMPACT,
            cell,
            held,
            {"observation_cap_chars": 800},
            {"observation_cap_chars": 1600},
        )["baseline_digest"]
    )


def test_two_constants_that_move_together_get_one_verdict_and_one_re_run_scope():
    change = PolicyChange(
        baseline={"observation_cap_chars": 800, "keep_last_n": 3},
        candidate={"observation_cap_chars": 1600, "keep_last_n": 1},
    )
    audits = audit_policy_change(_designs(), change)
    rows = [row for study in audits.values() for row in study]
    assert all(row["policy_fields"] == ["observation_cap_chars", "keep_last_n"] for row in rows)
    assert all(row["verdict"] == VERDICT_CHANGED for row in rows)

    summary = policy_change_summary(audits, change)
    assert summary["compound"] and summary["n_invalidated"] == len(rows) == 22
    assert summary["change_label"] == "observation_cap_chars 800 -> 1600, keep_last_n 3 -> 1"
    table = format_policy_change_table(audits, summary)
    assert "audited as ONE change" in table and "keep_last_n 3 -> 1" in table


def test_a_cell_that_pins_part_of_a_compound_change_is_audited_on_the_rest():
    """A collapse cell owns `compact_share`, so it keeps its own -- and still reads the cap move."""
    change = PolicyChange(
        baseline={"compact_share": 0.5, "observation_cap_chars": 800},
        candidate={"compact_share": 0.45, "observation_cap_chars": 1600},
    )
    audits = audit_policy_change(_designs(), change)
    collapse = audits[KIND_COLLAPSE]
    assert all(row["not_applicable_fields"] == ["compact_share"] for row in collapse)
    assert all(row["verdict"] == VERDICT_CHANGED for row in collapse)

    summary = policy_change_summary(audits, change)
    assert summary["n_not_applicable"] == 0 and summary["n_partially_applicable"] == len(collapse)
    assert KIND_COLLAPSE in summary["studies_invalidated"]
    assert "declare part of this change" in format_policy_change_table(audits, summary)


def test_the_geometry_records_which_fields_a_cell_pins_itself():
    collapse = declared_geometry(load_audited_design(DESIGNS[KIND_COLLAPSE]), KIND_COLLAPSE)
    surface = declared_geometry(load_audited_design(DESIGNS[KIND_SURFACE]), KIND_SURFACE)
    assert all("compact_share" in cell["pinned_fields"] for cell in collapse)
    assert all(cell["pinned_fields"] == [] for cell in surface)


def test_the_general_audit_and_the_summarize_bound_view_agree_cell_for_cell():
    """The bound audit is one USE of this mechanism, so the two must never disagree."""
    audits = audit_policy_change(_designs(), BOUND_CHANGE)
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
    audits = audit_policy_change(_designs(), BOUND_CHANGE)
    summary = policy_change_summary(audits, BOUND_CHANGE)
    assert summary["n_invalidated"] == 4 and summary["n_prompt_invariant"] == 18
    table = format_policy_change_table(audits, summary)
    assert "re-run scope" in table and "surface-d10-g23000" in table
    # An invariant cell is never in the scope list, which is the point of the audit.
    assert "- compact_memory_boundary_surface surface-d6-g12000:" not in table

    paths = persist_policy_change_audit(
        audits, summary, data_dir=tmp_path, table=table, mirror=lambda *_: None
    )
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["config"]["summary"]["policy_fields"] == ["summary_input_cap"]
    assert manifest["metrics"]["objective_score"] == 0.0  # something was invalidated


def test_a_change_that_invalidates_nothing_scores_a_clean_audit(tmp_path: Path):
    audits = audit_policy_change(_designs(), KEEP_CHANGE)
    summary = policy_change_summary(audits, KEEP_CHANGE)
    table = format_policy_change_table(audits, summary)
    assert "invalidates NO published number" in table and "re-run scope" not in table
    paths = persist_policy_change_audit(
        audits, summary, data_dir=tmp_path, table=table, mirror=lambda *_: None
    )
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["metrics"]["objective_score"] == 1.0
