"""The imperfect-play safety margin: the worst-case probe, the band it narrows, and the evidence.

Perfect play is the SHORTEST walk that finishes. Everything here is about the gap between that walk
and the longest one a cell's own step budget admits -- what the gap measures, how design validation
spends it, what the run bundles say a served model actually used of it, and how far a bound
invariance verdict read on the short walk carries to the long one.
"""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from llb.bench.memory.boundary.probe import (
    cap_peak_prompt_chars,
    cap_prompt_sequence,
    compact_fold_input_probe,
)
from llb.bench.memory.boundary.surface import (
    load_surface_design,
    surface_cap_peak_margins,
    surface_cap_peaks,
    validate_surface_design,
)
from llb.bench.memory.boundary.surface_cells import depth_cap_peak_margin
from llb.bench.memory.cap_audit import (
    VERDICT_INVARIANT,
    VERDICT_SENSITIVE,
    audit_design,
    audit_summary,
)
from llb.bench.memory.extra_steps import (
    UNREAD_MISSING,
    UNREAD_NO_PATH,
    UNREAD_NO_STEPS,
    bundle_step_counts,
    cell_observed_extra_steps,
    margin_is_covered,
    observed_extra_steps,
)
from llb.bench.memory.fold_step.ladder import (
    guard_is_cap_fitting,
    guard_is_cap_fitting_under_imperfect_play,
    imperfect_play_guard_band,
    usable_guard_band,
)
from llb.bench.memory.worst_case_probe import (
    MIN_BUDGETED_EXTRA_STEPS,
    cap_peak_margin,
    margin_peaks,
    stalling_controller,
    worst_case_cap_prompt_sequence,
    worst_case_fold_input_probe,
)
from llb.bench.policy_change.audit import KIND_SURFACE
from llb.bench.policy_change.geometry import load_audited_design
from llb.bench.policy_change.tasks import (
    TASK_BUILDER_LONG_TRANSCRIPT,
    worst_case_replay_controller,
)

ROOT = Path(__file__).resolve().parents[4]
DESIGN_PATH = ROOT / "samples/benchmarks/agentic_compact_memory_boundary_surface_design.json"

GEOMETRY = {"depth": 6, "n_tasks": 7}


def test_the_stalling_walk_spends_the_whole_step_budget_and_grows_the_peak():
    """The margin exists because the budget, not the workflow, decides how long a transcript gets."""
    perfect = cap_prompt_sequence(**GEOMETRY)
    worst = worst_case_cap_prompt_sequence(**GEOMETRY)
    # Perfect play finishes the step after the workflow completes; the stalling walk never finishes.
    assert len(perfect) == GEOMETRY["depth"] + 1
    assert len(worst) == GEOMETRY["depth"] + 4
    # The stalling walk IS the perfect one up to the point perfect play stops, so the extra prompts
    # are extra transcript rather than a different geometry.
    assert worst[: len(perfect)] == perfect
    assert max(worst) > max(perfect)


def test_the_margin_prices_both_peaks_and_the_steps_the_budget_left_over():
    margin = cap_peak_margin(**GEOMETRY)
    assert margin["perfect_play_peak_chars"] == cap_peak_prompt_chars(**GEOMETRY)
    assert margin["worst_case_peak_chars"] > margin["perfect_play_peak_chars"]
    assert (
        margin["margin_chars"]
        == margin["worst_case_peak_chars"] - margin["perfect_play_peak_chars"]
    )
    assert margin["margin_ratio"] == pytest.approx(
        margin["worst_case_peak_chars"] / margin["perfect_play_peak_chars"]
    )
    assert (
        margin["budgeted_extra_steps"] == margin["worst_case_steps"] - margin["perfect_play_steps"]
    )
    assert margin["budgeted_extra_steps"] >= MIN_BUDGETED_EXTRA_STEPS
    assert margin_peaks(margin) == (
        margin["worst_case_peak_chars"],
        margin["perfect_play_peak_chars"],
    )


def test_a_budget_with_no_room_to_misbehave_has_no_margin_to_certify_against():
    """A zero margin published as a margin would state a safety property the run cannot have."""
    with pytest.raises(ValueError, match="admits no imperfect play"):
        cap_peak_margin(**GEOMETRY, max_steps_margin=1)


def test_the_margin_narrows_the_band_from_below_and_never_widens_it_from_above():
    """Cap must fit the worst case; compact must fire on the SHORTEST transcript, not the longest."""
    margin = cap_peak_margin(**GEOMETRY)
    worst, perfect = margin_peaks(margin)
    share = 0.5
    plain = usable_guard_band(perfect, share)
    guarded = imperfect_play_guard_band(worst, perfect, share)
    assert guarded[0] > plain[0]
    assert guarded[1] == plain[1]

    # A guard between the two peaks is cap-fitting for a perfect controller and for nobody else.
    between = (perfect + worst) // 2
    assert guard_is_cap_fitting(between, perfect, share) is True
    assert guard_is_cap_fitting_under_imperfect_play(between, worst, perfect, share) is False

    with pytest.raises(ValueError, match="at least the perfect-play peak"):
        imperfect_play_guard_band(perfect - 1, perfect, share)


def test_a_guard_only_imperfect_play_reaches_the_trigger_of_never_activates_compact():
    """Why the upper bound keeps the perfect-play peak: above it, only the wasted steps fold.

    At this guard the oracle transcript never crosses the compact trigger, so the compact arm makes
    no summary call at all -- while the stalling walk does. A cell placed there measures compaction
    only when the controller misbehaves, which is the opposite of a held-fixed activation floor.
    """
    margin = cap_peak_margin(**GEOMETRY)
    worst, perfect = margin_peaks(margin)
    share = 0.5
    guard = int((perfect + worst) / (2 * share))
    assert not guard_is_cap_fitting(guard, perfect, share)

    probe = {"max_prompt_chars": guard, "compact_share": share, **GEOMETRY}
    assert compact_fold_input_probe(**probe)["n_compactions"] == 0
    assert worst_case_fold_input_probe(**probe)["n_compactions"] == 1


def test_the_committed_surface_is_cap_fitting_for_the_controller_that_runs_it():
    """The published grid keeps every guard clear of the worst case, not only of perfect play."""
    design = load_surface_design(DESIGN_PATH)
    validate_surface_design(design)
    margins = surface_cap_peak_margins(design)
    share = design["held_fixed"]["compact_share"]
    assert surface_cap_peaks(design) == {
        depth: margin["perfect_play_peak_chars"] for depth, margin in margins.items()
    }
    for cell in design["surface"]["cells"]:
        worst, perfect = margin_peaks(margins[cell["depth"]])
        assert guard_is_cap_fitting_under_imperfect_play(
            cell["max_prompt_chars"], worst, perfect, share
        )


def test_validation_refuses_a_guard_that_only_perfect_play_fits():
    """The margin is a REFUSAL, not a column: a guard inside it stops the design being read."""
    design = load_surface_design(DESIGN_PATH)
    held = design["held_fixed"]
    shallowest = min(cell["depth"] for cell in design["surface"]["cells"])
    worst, perfect = margin_peaks(depth_cap_peak_margin(shallowest, held))

    anchor = design["reference"]["max_prompt_chars"]
    perfect_play_only = deepcopy(design)
    for cell in perfect_play_only["surface"]["cells"]:
        # Not the anchor geometry, which the grid must keep re-running whatever else moves.
        if cell["depth"] == shallowest and cell["max_prompt_chars"] != anchor:
            cell["max_prompt_chars"] = (perfect + worst) // 2
            break
    with pytest.raises(ValueError, match="cap fits imperfect play"):
        validate_surface_design(perfect_play_only)


def test_the_bound_invariance_verdict_is_stated_for_the_worst_case_as_well():
    """Every published surface cell keeps its verdict on the longest transcript its budget allows.

    The mechanism is visible in the geometry: a cap-fitting guard puts the compact trigger inside
    the prefix the two walks SHARE, so the first fold offers the summarizer the same bytes under
    both. That is what makes the published invariance a statement about a real controller, and the
    check is what would catch a future cell where it stops holding.
    """
    rows = audit_design(load_audited_design(DESIGN_PATH), study_kind=KIND_SURFACE)
    assert rows
    for row in rows:
        assert row["worst_case_verdict"] in (VERDICT_INVARIANT, VERDICT_SENSITIVE)
        assert row["worst_case_verdict"] == row["verdict"]
        assert row["worst_case_n_compactions"] >= row["n_compactions"]
        assert row["worst_case_trigger_elided_chars"] >= row["window_elided_chars"]

    summary = audit_summary({KIND_SURFACE: rows})
    assert summary["n_worst_case_bound_invariant"] + summary["n_worst_case_bound_sensitive"] == len(
        rows
    )
    assert summary["worst_case_only_sensitive"] == []


def test_worst_case_replay_is_refused_for_a_shape_that_has_no_longer_walk():
    """Stalling a pipeline task is a different task, not a longer walk of the same one."""
    design = load_audited_design(DESIGN_PATH)
    held = design["held_fixed"]
    assert worst_case_replay_controller(None, held) is not None
    with pytest.raises(ValueError, match="defined only for"):
        worst_case_replay_controller(None, {**held, "task_builder": TASK_BUILDER_LONG_TRANSCRIPT})


def test_the_stalling_controller_only_ever_drives_the_workflow_tool():
    """The margin prices imperfect play of the task, never adversarial use of the sandbox."""
    call = json.loads(stalling_controller('Почни з advance із токеном "wf-000-0".'))
    assert call == {"name": "advance", "arguments": {"token": "wf-000-0"}}
    assert json.loads(stalling_controller("[next token: wf-000-3]"))["arguments"]["token"] == (
        "wf-000-3"
    )


def _bundle(directory: Path, steps: list[int]) -> Path:
    """A minimal compact-versus-cap bundle: the manifest, and one scored row per episode."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"run_name": "agentic-compact-vs-cap-compact"}), "utf-8")
    (directory / "scores.jsonl").write_text(
        "".join(
            json.dumps({"item_id": f"t{index}", "n_steps": value}) + "\n"
            for index, value in enumerate(steps)
        ),
        "utf-8",
    )
    return manifest


def test_the_extra_steps_a_real_controller_spent_are_read_back_out_of_its_bundle(tmp_path):
    manifest = _bundle(tmp_path / "compact", [7, 8, 10, 7])
    assert bundle_step_counts(manifest) == [7, 8, 10, 7]

    observed = observed_extra_steps(manifest, perfect_play_steps=7)
    assert observed["read"] is True
    assert observed["extra_steps"] == [0, 1, 3, 0]
    assert observed["max_extra_steps"] == 3
    assert observed["mean_extra_steps"] == pytest.approx(1.0)
    assert observed["n_episodes_beyond_perfect_play"] == 2


def test_an_episode_that_ended_early_is_kept_as_measured_rather_than_floored(tmp_path):
    """A refused prompt ends an episode BEFORE the oracle walk ends; clamping would hide it."""
    manifest = _bundle(tmp_path / "cap", [3, 4])
    observed = observed_extra_steps(manifest, perfect_play_steps=7)
    assert observed["extra_steps"] == [-4, -3]
    assert observed["max_extra_steps"] == -3


def test_a_bundle_this_host_does_not_have_reports_itself_unread(tmp_path):
    """The audit runs where the runs do not, so an absent bundle is a fact and not a failure."""
    assert observed_extra_steps(None, perfect_play_steps=7)["unread_reason"] == UNREAD_NO_PATH
    absent = observed_extra_steps(tmp_path / "gone" / "manifest.json", perfect_play_steps=7)
    assert (absent["read"], absent["unread_reason"]) == (False, UNREAD_MISSING)
    assert absent["max_extra_steps"] is None

    empty = tmp_path / "stepless"
    empty.mkdir()
    (empty / "manifest.json").write_text("{}", "utf-8")
    (empty / "scores.jsonl").write_text(json.dumps({"item_id": "t0"}) + "\n", "utf-8")
    stepless = observed_extra_steps(empty / "manifest.json", perfect_play_steps=7)
    assert (stepless["read"], stepless["unread_reason"]) == (False, UNREAD_NO_STEPS)


def test_the_observed_steps_are_read_against_the_budget_the_probe_priced(tmp_path):
    inside = cell_observed_extra_steps(
        {
            "compact": str(_bundle(tmp_path / "a-compact", [7, 9])),
            "observation_cap": str(_bundle(tmp_path / "a-cap", [7, 7])),
        },
        perfect_play_steps=7,
    )
    assert margin_is_covered(inside, budgeted_extra_steps=3) is True
    assert margin_is_covered(inside, budgeted_extra_steps=1) is False

    # No arm readable is not the same answer as "the observed steps fit".
    unread = cell_observed_extra_steps({"compact": None}, perfect_play_steps=7)
    assert margin_is_covered(unread, budgeted_extra_steps=3) is None
