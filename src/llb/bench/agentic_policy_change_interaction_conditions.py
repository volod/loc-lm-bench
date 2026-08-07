"""What each pair of moved policy fields demands of a prompt guard, as intervals the solver reads.

Every builder here answers the same three questions for one fold step -- is the first field silent
audited alone, is the second, and do the two together move a prompt -- in the arithmetic that pair
actually runs on (`agentic_policy_change_interaction_terms` holds the shape of an answer). For
`compact_share` x `summary_input_cap` all three are intervals the fold-step ladder and the boundary
probe compute between them, which is why that pair has a solvable band. For every other pair two of the three contradict each other by
construction, and the condition says so in its own words rather than by returning an empty answer
nobody can read: a keep that folds the whole transcript cannot also fold a different span, and a
head share that moves no prompt LENGTH gives no partner field anything to read.

`observation_cap_chars` is answered in `agentic_policy_change_interaction_cap` instead of here: its
conditions have to separate the prompts the episode SENDS from the ones the loop merely builds, and
that arithmetic is the substance of the answer rather than a detail of it.
"""

from typing import cast

from llb.bench.agentic.context import SUMMARY_INPUT_CAP_WINDOW
from llb.bench.agentic_memory_boundary_probe import compact_fold_input_probe
from llb.bench.agentic_memory_fold_step_ladder import (
    fold_step_guard_interval,
    fold_step_trigger_interval,
    live_entries_at_fold_step,
    smallest_guard_reaching,
)
from llb.bench.agentic_policy_change_audit import PolicyChange
from llb.bench.agentic_policy_change_interaction_terms import (
    AUDITED_POLICY_FIELDS,
    FIELD_BOUND,
    FIELD_HEAD,
    FIELD_KEEP_RECENT,
    FIELD_SHARE,
    SEPARATING_BOUNDS,
    BandCondition,
    StepConditions,
    StepGeometry,
)


def folds_at_this_step(step: StepGeometry) -> BandCondition:
    """The guards whose trigger selects THIS fold step -- every coupling starts from it."""
    low, high = fold_step_guard_interval(step.sequence, step.step, step.share)
    return BandCondition(
        name="the_guard_folds_at_this_step",
        why="else this step is not where the change would bite",
        low=low,
        high=high,
    )


def share_bound_conditions(step: StepGeometry) -> StepConditions:
    """`compact_share` x `summary_input_cap`: three intervals the ladder and the probe already give."""
    baseline_share, candidate_share = _separating_shares(step.change)
    trigger = fold_step_trigger_interval(step.sequence, step.step)
    folds = [
        fold_step_guard_interval(step.sequence, step.step, share)
        for share in (baseline_share, candidate_share)
    ]
    both = BandCondition(
        name="both_shares_fold_at_this_step",
        why="else the share alone already changes the prompts",
        low=max(edge[0] for edge in folds),
        high=min(edge[1] for edge in folds),
    )
    triggers = f"folds at triggers [{trigger[0]}, {trigger[1]})"
    if both.is_empty:
        return StepConditions(detail=triggers, conditions=(both,))
    offered = _offered_at_one_fold(cast(int, both.low), step)
    if offered is None:
        return StepConditions(
            detail=triggers,
            conditions=(
                both,
                BandCondition.impossible(
                    "the_episode_folds_exactly_once_here",
                    "`summary_input_chars` is a sum over folds, and the elision inequality is a "
                    "statement about ONE offered transcript",
                ),
            ),
        )
    return StepConditions(
        detail=f"offered {offered}, {triggers}",
        conditions=(
            both,
            BandCondition(
                name="the_baseline_share_elides_nothing",
                why=f"else the bound audited alone already reports the change (offered {offered})",
                low=smallest_guard_reaching(offered, baseline_share),
            ),
            BandCondition(
                name="the_candidate_share_elides",
                why=f"else the compound reading has nothing to report either (offered {offered})",
                high=smallest_guard_reaching(offered, candidate_share),
            ),
        ),
    )


def keep_recent_conditions(step: StepGeometry) -> StepConditions:
    """`compact_keep_recent` x anything: the keep's own two conditions contradict each other.

    The keep reaches a prompt through ONE decision -- which entries the fold hands the summarizer --
    and `compact_state` falls back to folding the whole transcript when the keep would leave nothing
    to fold. So two keeps fold the same span exactly when the fold has at most `min(keeps)` live
    entries, which is also exactly when the moved keep contributes nothing to the compound.
    """
    keeps = step.moved(FIELD_KEEP_RECENT)
    live, kept = live_entries_at_fold_step(step.step), min(int(keeps[0]), int(keeps[1]))
    return StepConditions(
        detail=f"{live} live entries at the fold, keeps {keeps[0]} -> {keeps[1]}",
        conditions=(
            folds_at_this_step(step),
            BandCondition.satisfied_when(
                live <= kept,
                "the_keep_audited_alone_is_silent",
                f"the fold must hand the summarizer the same span under both keeps, which needs at "
                f"most {kept} live entries; step {step.step} has {live}",
            ),
            BandCondition.satisfied_when(
                live > kept,
                "the_moved_keep_folds_a_different_span",
                f"the compound must fold something the partner field alone does not, which needs "
                f"more than {kept} live entries; step {step.step} has {live}",
            ),
        ),
    )


def head_share_conditions(step: StepGeometry) -> StepConditions:
    """`observation_head_share` x anything: it moves bytes inside a trim, never a LENGTH.

    `trim_observation` keeps `head + tail = cap_chars` around a marker whose own length depends on
    how many chars went missing, not on where the split fell. So the head share cannot move a prompt
    size, and therefore cannot move a fold step, a trigger crossing, or an overflow -- the only
    things another policy field reads. Whatever bytes it does move, it moves at the same steps under
    either value of any partner field, which is a change the union reports on its own.
    """
    head = step.moved(FIELD_HEAD)
    return StepConditions(
        detail=f"head share {head[0]} -> {head[1]} keeps every trimmed length exactly",
        conditions=(
            folds_at_this_step(step),
            BandCondition.always(
                "the_head_share_moves_no_prompt_length",
                "a head/tail split of the same cap keeps the trimmed length exactly, so no partner "
                "field can read the head share through a prompt size",
            ),
            BandCondition.impossible(
                "the_partner_field_hides_the_moved_bytes",
                "the bytes the head share moves are shown at the same steps under either partner "
                "value, so what it changes it already changes audited alone",
            ),
        ),
    )


def inert_field_conditions(step: StepGeometry) -> StepConditions:
    """`keep_last_n` x anything: the field parameterizes neither audited policy, so it moves nothing.

    A published cell is replayed under `observation_cap` and `compact`; `keep_last_n` is read by the
    `keep_last_n` policy alone. Nothing it can be set to reaches either arm, so it is invariant
    audited alone AND contributes nothing to a compound -- the two readings agree by construction.
    """
    inert = sorted(set(step.change.fields) - AUDITED_POLICY_FIELDS)
    return StepConditions(
        detail=f"{', '.join(inert)} parameterizes neither audited arm",
        conditions=(
            folds_at_this_step(step),
            BandCondition.always(
                "the_inert_field_is_silent_audited_alone",
                f"{', '.join(inert)} is read by neither `observation_cap` nor `compact`",
            ),
            BandCondition.impossible(
                "the_inert_field_moves_the_compound",
                f"a field neither audited arm reads cannot move a prompt in one, so the compound is "
                f"the change without {', '.join(inert)}",
            ),
        ),
    )


def _offered_at_one_fold(guard_chars: int, step: StepGeometry) -> int | None:
    """The transcript ONE fold hands the summarizer, or None when the episode folds more than once.

    Probed under the window bound so nothing is elided, and at a guard that folds at the wanted step
    by construction. A second compaction would make `summary_input_chars` a sum over folds, and the
    elision inequality is a statement about ONE offered transcript -- so that step is reported as no
    band rather than as a wrong one.
    """
    probe = compact_fold_input_probe(
        max_prompt_chars=guard_chars,
        compact_share=step.share,
        summary_input_cap=SUMMARY_INPUT_CAP_WINDOW,
        **step.geometry,
    )
    if probe["n_compactions"] != 1 or probe["summary_input_elided_chars"] != 0:
        return None
    return probe["summary_input_chars"]


def _separating_shares(change: PolicyChange) -> tuple[float, float]:
    """The two shares, once the change is confirmed to be one the band arithmetic can answer."""
    bounds = (change.baseline[FIELD_BOUND], change.candidate[FIELD_BOUND])
    if bounds != SEPARATING_BOUNDS:
        raise ValueError(
            f"only a {SEPARATING_BOUNDS[0]} -> {SEPARATING_BOUNDS[1]} bound move can separate the "
            f"two readings, got {bounds[0]} -> {bounds[1]}"
        )
    return (
        float(cast(float, change.baseline[FIELD_SHARE])),
        float(cast(float, change.candidate[FIELD_SHARE])),
    )
