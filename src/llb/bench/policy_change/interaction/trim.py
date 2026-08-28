"""What `summary_trim_strategy` demands of a guard -- and why no partner field can supply it.

The trim strategy is the one auditable field that reaches NOTHING unless another decision has
already been made. `head_tail` and `per_entry_head` render the identical transcript whenever the
folded transcript fits the summarize-input cap (`_bounded_summary_transcript` returns the offered
bytes untouched), so the field is readable only inside an episode that ELIDES. That single property
decides every pair it is in, and it decides them all the same way:

  - if the change's BASELINE policy already elides at this fold, the trim audited alone renders a
    different summarize prompt, so the per-field union reports it and there is nothing to separate;
  - if it does not, the compound can only read the trim when the partner field OPENS the elision --
    and a partner that turns an un-elided fold into an elided one has changed the summarize prompt
    by itself, under the held `head_tail` trim, so the union reports the partner instead.

Both branches are measured rather than asserted: the probe walks the fold at the guard the step
selects, once under the whole baseline policy and once under the whole candidate policy, and the
condition carries the two elision counts it read. The answer is `no_geometry` for every pair whose
partner can move elision at all, and the pairs whose partner cannot (`observation_head_share`,
`keep_last_n`) keep their own `independent` conditions.
"""

from collections.abc import Mapping
from typing import Any, cast

from llb.bench.memory.boundary.probe import compact_fold_input_probe
from llb.bench.policy_change.interaction.conditions import folds_at_this_step
from llb.bench.policy_change.interaction.terms import (
    FIELD_BOUND,
    FIELD_CAP,
    FIELD_HEAD,
    FIELD_SHARE,
    BandCondition,
    StepConditions,
    StepGeometry,
    shipped_policy_value,
)

CONDITION_SILENT = "the_trim_audited_alone_is_silent"
CONDITION_PARTNER_OPENS = "the_partner_opens_the_elision_silently"
# Task-world keywords `compact_fold_input_probe` takes straight from the geometry. A moved field
# outside this pair either has its own probe keyword (`compact_share`, `summary_input_cap`) or
# cannot reach the probe seam at all, in which case the reading is stated at the geometry's value.
_GEOMETRY_FIELDS = (FIELD_CAP, FIELD_HEAD)


def trim_strategy_conditions(step: StepGeometry) -> StepConditions:
    """`summary_trim_strategy` x a field that can move elision: the two branches above."""
    folds = folds_at_this_step(step)
    if folds.is_empty:
        return StepConditions(
            detail=f"no guard folds at step {step.step}",
            conditions=(folds,),
        )
    guard = cast(int, folds.low)
    baseline = _elision_at(step, guard, step.change.baseline)
    candidate = _elision_at(step, guard, step.change.candidate)
    return StepConditions(
        detail=(
            f"at guard {guard} the fold offers {baseline['offered']} chars and elides "
            f"{baseline['elided']} under the baseline policy, {candidate['elided']} under the "
            "candidate"
        ),
        conditions=(
            folds,
            _silent_audited_alone(baseline, guard),
            _partner_opens_the_elision(baseline, candidate, guard),
        ),
    )


def _silent_audited_alone(baseline: dict[str, int], guard: int) -> BandCondition:
    """The union misses the trim only while the baseline fold hands the summarizer everything."""
    return BandCondition.satisfied_when(
        baseline["elided"] == 0,
        CONDITION_SILENT,
        (
            "the two trims render the identical summarize prompt while nothing is elided, so the "
            f"trim is silent audited alone only there; at guard {guard} the baseline policy elides "
            f"{baseline['elided']} of {baseline['offered']} offered chars"
        ),
    )


def _partner_opens_the_elision(
    baseline: dict[str, int], candidate: dict[str, int], guard: int
) -> BandCondition:
    """The compound needs an elision the partner opens without being reported for opening it."""
    if candidate["elided"] == 0 and baseline["elided"] == 0:
        return BandCondition.impossible(
            CONDITION_PARTNER_OPENS,
            (
                f"neither policy elides at guard {guard} ({candidate['offered']} offered chars "
                "fit the summarize-input cap under both), so both trims render the same summarize "
                "prompt and the compound has nothing to report either"
            ),
        )
    return BandCondition.impossible(
        CONDITION_PARTNER_OPENS,
        (
            f"the elision the trim needs is opened by the partner field itself ({baseline['elided']}"
            f" -> {candidate['elided']} elided chars at guard {guard}), and cutting a folded "
            "transcript that previously fit already moves the summarize prompt under the held "
            "`head_tail` trim -- so the per-field union reports the partner"
        ),
    )


def _elision_at(step: StepGeometry, guard: int, settings: Mapping[str, Any]) -> dict[str, int]:
    """Walk the fold this step selects under ONE whole policy, and read what it elides.

    Probed under the trim the fold is measured with rather than the one being audited: the offered
    transcript and the cap it is cut to are the same under both trims by construction, which is
    exactly why the field cannot be read where nothing is cut.
    """
    overrides = {field: settings[field] for field in _GEOMETRY_FIELDS if field in settings}
    probe = compact_fold_input_probe(
        **{**step.geometry, **overrides},
        max_prompt_chars=guard,
        compact_share=float(settings.get(FIELD_SHARE, step.share)),
        summary_input_cap=str(settings.get(FIELD_BOUND, shipped_policy_value(FIELD_BOUND))),
    )
    fold_inputs = cast(list[int], probe["summary_fold_input_chars"])
    return {
        "offered": fold_inputs[0] if fold_inputs else 0,
        "elided": int(cast(int, probe["summary_input_elided_chars"])),
    }
