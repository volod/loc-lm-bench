"""Which prompt guards a workload can be run at and still measure what it was built to measure.

A per-family guard fit is only allowed to choose from guards that keep the workload's declared
REGIME, and for the middle-critical set that is four properties at once, none of which the guard
leaves alone. Lower the guard and the fold lands earlier -- but the summarize-input bound is the
same window, so the fold offers a shorter transcript against a smaller cap, and the span the cap
elides moves with both. Far enough down, the fold either elides nothing at all, or elides a span
that no longer contains the fact its task planted in that stratum, or lands before that fact has
entered the transcript at all -- and the workload has quietly stopped being the experiment it is
named after.

So every candidate is walked with an oracle first, and one refused with a NAMED reason rather than
silently skipped: a fit that runs out of usable guards has to be able to say which property ran out
and where. Nothing here is family-specific -- these are facts about the geometry, so the same scan
answers for every family and the fit only supplies the score.
"""

from typing import cast

from llb.bench.agentic.context_policy import SUMMARY_TRIM_HEAD_TAIL, SUMMARY_TRIM_PER_ENTRY_HEAD
from llb.bench.agentic.context_summary import summary_prompt_overhead_chars
from llb.bench.agentic.design_fields import as_int, as_mapping
from llb.bench.context_policy.guard_band import guard_grid
from llb.bench.memory.window_elision.tasks import FactNotOffered, answer_fact_placement
from llb.bench.summary_trim.design import probe_workload_task
from llb.bench.summary_trim.workloads import build_workload_tasks

# Why a candidate guard is not a usable member of the band. Each one is a way a guard would still
# fold, and stop measuring what the workload was built to measure.
REFUSED_FOLD_COUNT = "fold_count_leaves_the_declared_regime"
REFUSED_NO_ELISION = "the_fold_fits_the_summarize_input_bound_and_elides_nothing"
REFUSED_UNFOLDED_FACT = "an_answer_fact_is_not_inside_the_folded_transcript_yet"
REFUSED_PLACEMENT = "an_answer_fact_leaves_its_declared_elision_stratum"
REFUSED_UNPAIRED = "the_two_arms_do_not_offer_the_summarizer_the_same_transcript"


def at_guard(workload: dict[str, object], guard: int) -> dict[str, object]:
    """The workload as one candidate guard would run it."""
    return {**workload, "max_prompt_chars": guard}


def scan_guard_band(
    workload: dict[str, object], held: dict[str, object], spec: dict[str, object]
) -> list[dict[str, object]]:
    """Every candidate guard in the declared band, with its fold step or its refusal."""
    return [probe_guard(workload, held, guard) for guard in guard_grid(spec)]


def usable_guards(scan: list[dict[str, object]]) -> dict[int, int]:
    """The fold step of every scanned guard that still produces the declared regime."""
    return {
        int(cast(int, row["max_prompt_chars"])): int(cast(int, row["fold_step"]))
        for row in scan
        if row["refusal"] is None
    }


def probe_guard(
    workload: dict[str, object], held: dict[str, object], guard: int
) -> dict[str, object]:
    """One candidate guard's model-free geometry, and whether it still holds the regime."""
    candidate = at_guard(workload, guard)
    baseline = probe_arm(candidate, held, SUMMARY_TRIM_HEAD_TAIL)
    folded = [int(cast(int, row["n_compactions"])) for row in baseline]
    return {
        "max_prompt_chars": guard,
        "fold_step": max(_first_fold_step(row) for row in baseline),
        "n_compactions": max(folded),
        "summary_input_chars": max(int(cast(int, row["summary_input_chars"])) for row in baseline),
        "summary_input_elided_chars": max(
            int(cast(int, row["summary_input_elided_chars"])) for row in baseline
        ),
        "refusal": _refusal(candidate, held, baseline, folded),
    }


def probe_arm(
    workload: dict[str, object], held: dict[str, object], arm: str
) -> list[dict[str, object]]:
    """One probe per task, so a per-task placement and a per-task elision stay addressable."""
    return [
        probe_workload_task(workload, held, arm, index)
        for index in range(len(build_workload_tasks(workload)))
    ]


def _refusal(
    candidate: dict[str, object],
    held: dict[str, object],
    baseline: list[dict[str, object]],
    folded: list[int],
) -> str | None:
    """Why this guard cannot carry the workload's declared regime, or `None` when it can.

    Every task is required to hold the property, not the worst or the average one: the stratum is
    read per case, so one case whose fold elides nothing contributes an outcome the comparison
    cannot attribute to the trim.
    """
    expected = as_mapping(candidate, "expected")
    if set(folded) != {as_int(expected, "n_compactions")}:
        return REFUSED_FOLD_COUNT
    if min(int(cast(int, row["summary_input_elided_chars"])) for row in baseline) <= 0:
        return REFUSED_NO_ELISION
    misplaced = _misplaced_fact(candidate, baseline)
    if misplaced is not None:
        return misplaced
    # The pairing property itself. It holds by construction on today's geometry -- both trims read
    # the same offered transcript and differ only in what the cap lets through -- and it is checked
    # anyway, because it is the assumption every paired case in this study rests on.
    entry_aware = probe_arm(candidate, held, SUMMARY_TRIM_PER_ENTRY_HEAD)
    if [row["summary_fold_input_chars"] for row in entry_aware] != [
        row["summary_fold_input_chars"] for row in baseline
    ]:
        return REFUSED_UNPAIRED
    return None


def _misplaced_fact(candidate: dict[str, object], baseline: list[dict[str, object]]) -> str | None:
    """Why an answer fact no longer occupies the stratum its task planted it in, or `None`.

    The two ways it can fail are different facts about the guard and the band reports them apart,
    because at a fast-growing transcript it is the FIRST one that bounds the band. A guard low
    enough folds a transcript the fact has not entered yet -- its stage is simply past the last
    folded entry -- which says the fold is too early for the workload's stages, not that the trim
    boundaries moved. A guard high enough folds a transcript that does contain the fact and puts it
    in the wrong stratum, which says the boundaries moved under it.
    """
    cap = as_int(candidate, "max_prompt_chars") - summary_prompt_overhead_chars()
    for record, probe in zip(build_workload_tasks(candidate), baseline, strict=True):
        try:
            placement = answer_fact_placement(
                record,
                offered_chars=int(cast(int, probe["summary_input_chars"])),
                transcript_cap_chars=cap,
            )
        except FactNotOffered:
            return REFUSED_UNFOLDED_FACT
        except ValueError:
            # The offered span matches no prefix of this task's transcript, so the fact cannot be
            # located against the trim boundaries at all -- a refusal, not a placement to argue
            # about.
            return REFUSED_PLACEMENT
        if placement["declared_stratum"] != placement["measured_stratum"]:
            return REFUSED_PLACEMENT
    return None


def _first_fold_step(probe: dict[str, object]) -> int:
    steps = cast(list[int], probe["summary_fold_steps"])
    return steps[0] if steps else 0
