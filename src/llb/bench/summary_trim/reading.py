"""What a measured adoption run says about one workload and one family.

Four quantities are compared per workload, and they answer different halves of the question.
Completion says whether the entry-aware trim is SAFE; total model-input chars, summary prompt
chars, and fold count say what it COSTS. A strategy that recovers a lost case by spending more of
the window is a different recommendation from one that recovers it at the same prompt size, so the
cost columns travel with the outcome; `agentic_summary_trim_adoption` turns them into a verdict.

Pairing is exact and per case, never rate-against-rate. The two arms run the identical task set and
are byte-identical up to and including the transcript the first fold OFFERS the summarizer, so a
case pairs when both arms fold the same offered bytes; anything else is an unpaired case and is
named as one instead of being averaged into a delta.

An unpaired case has two kinds, and only one of them can be about the treatment. When an arm never
FOLDS, no trim ran in it at all -- the two arms executed the identical program over prompts a replay
shows to be byte-identical up to the fold -- so that case diverged for a reason upstream of the
strategy. (The observed cause is the SERVING stack: an episode can end the walk early on a
byte-identical prompt because the completion an Ollama endpoint returns depends on the request
history before it, which is why such a case reproduces exactly across runs instead of behaving like
sampling noise.) It is excluded from the delta, counted, and named, and the exclusion cannot be
correlated with the arm. When BOTH arms fold and still offer different bytes, the divergence is
downstream of a trim that did run, so it is not separable from the treatment and the whole workload
reading is refused. Excluded cases stay in the completion RATE either way, so an arm cannot buy a
better rate by ending episodes early.
"""

from typing import cast

from llb.bench.agentic.model import STATUS_CONTEXT_OVERFLOW
from llb.bench.agentic.context_policy import SUMMARY_TRIM_HEAD_TAIL, SUMMARY_TRIM_PER_ENTRY_HEAD
from llb.bench.memory.window_elision.tasks import STRATA

WORKLOAD_UNCHANGED = "entry_aware_changes_no_paired_outcome"
WORKLOAD_RECOVERS = "entry_aware_recovers_paired_completion"
WORKLOAD_REGRESSES = "entry_aware_loses_paired_completion"
WORKLOAD_MIXED = "entry_aware_paired_outcomes_conflict"
WORKLOAD_UNPAIRED = "workload_arms_are_not_exactly_paired"

# What the balanced schedule buys: with each task's two arms adjacent and the first position
# alternating, "ran second" is its own column, so a run can say whether the DROPOUT tracks position
# or the treatment instead of having to assume it does not.
ORDER_NO_POSITION_EFFECT = "reaching_the_fold_does_not_track_arm_order"
ORDER_POSITION_EFFECT = "reaching_the_fold_tracks_arm_order"
ORDER_UNBALANCED = "arm_order_is_not_balanced_across_the_task_set"

# A workload whose fold elides nothing cannot say anything about the strategies, so it is a
# CONTROL: it must stay byte-identical, and it never contributes evidence for adoption.
CONTROL_REASON = "the fold fits the summarize-input bound, so both arms render the same prompt"


def workload_reading(
    baseline_row: dict[str, object], candidate_row: dict[str, object]
) -> dict[str, object]:
    """One workload's paired comparison between the `head_tail` reference and the entry-aware trim."""
    baseline = _by_id(baseline_row)
    candidate = _by_id(candidate_row)
    shared = sorted(baseline.keys() & candidate.keys())
    paired = [item for item in shared if _pairs(baseline[item], candidate[item])]
    unfolded = [
        item
        for item in shared
        if item not in set(paired) and _no_trim_ran(baseline[item], candidate[item])
    ]
    divergent = len(baseline) - len(paired) - len(unfolded)
    readable = (
        bool(paired)
        and not divergent
        and baseline_row["task_set_digest"] == candidate_row["task_set_digest"]
    )
    wins = sum(_won(candidate[item], baseline[item]) for item in paired)
    losses = sum(_won(baseline[item], candidate[item]) for item in paired)
    reading = _reading(readable, wins, losses, len(paired))
    return {
        "workload": baseline_row["workload"],
        "n_pairs": len(paired),
        "n_unpaired": len(baseline) - len(paired),
        # Split so a report can say which kind of unpaired case it is looking at.
        "n_unpaired_no_fold": len(unfolded),
        "n_unpaired_divergent_fold": divergent,
        "entry_aware_wins": wins,
        "head_tail_wins": losses,
        "unchanged": len(paired) - wins - losses,
        "completion_delta": _rate(candidate_row) - _rate(baseline_row),
        **{
            f"d_{field}": _total(candidate, paired, field) - _total(baseline, paired, field)
            for field in ("model_input_prompt_chars", "summary_prompt_chars", "measured_folds")
        },
        "elides": any(
            int(cast(int, row["summary_input_elided_chars"])) > 0 for row in baseline.values()
        ),
        "reading": reading,
    }


def family_reading(run_rows: list[dict[str, object]]) -> dict[str, object]:
    """Every workload of one family, plus the per-stratum recovery the middle-critical set gives."""
    by_key = {(cast(str, row["workload"]), cast(str, row["arm"])): row for row in run_rows}
    names = list(dict.fromkeys(cast(str, row["workload"]) for row in run_rows))
    readings = [
        workload_reading(
            by_key[(name, SUMMARY_TRIM_HEAD_TAIL)], by_key[(name, SUMMARY_TRIM_PER_ENTRY_HEAD)]
        )
        for name in names
        if (name, SUMMARY_TRIM_HEAD_TAIL) in by_key
        and (name, SUMMARY_TRIM_PER_ENTRY_HEAD) in by_key
    ]
    return {
        "workloads": readings,
        "strata": _stratum_outcomes(by_key, names),
        "arm_order": position_reading(run_rows),
    }


def _stratum_outcomes(
    by_key: dict[tuple[str, str], dict[str, object]], names: list[str]
) -> dict[str, dict[str, int]]:
    """Per-stratum paired wins on whichever workload carries evidence strata."""
    stratum_workload = _stratum_workload(by_key, names)
    if stratum_workload is None:
        return {}
    baseline = _by_id(by_key[(stratum_workload, SUMMARY_TRIM_HEAD_TAIL)])
    candidate = _by_id(by_key[(stratum_workload, SUMMARY_TRIM_PER_ENTRY_HEAD)])
    return {stratum: _stratum_counts(baseline, candidate, stratum) for stratum in STRATA}


def _stratum_workload(
    by_key: dict[tuple[str, str], dict[str, object]], names: list[str]
) -> str | None:
    """The one workload whose cases carry an evidence stratum, if the run holds one."""
    for name in names:
        cases = cast(list[dict[str, object]], by_key[(name, SUMMARY_TRIM_HEAD_TAIL)]["cases"])
        if any("evidence_stratum" in row for row in cases):
            return name
    return None


def _stratum_counts(
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
    stratum: str,
) -> dict[str, int]:
    """One stratum's outcomes, over the pairs where a trim actually ran in BOTH arms.

    Counted on the same pairs the workload delta uses, for the same reason: a case where one arm
    never folded ran no trim at all, so scoring it against the entry-aware arm would charge a walk
    that ended early to a strategy that never executed. `n_declared` keeps the stratum's full size
    beside `n_pairs` so an under-powered stratum is visible rather than silently smaller.
    """
    declared = [
        item
        for item, row in baseline.items()
        if row.get("evidence_stratum") == stratum and item in candidate
    ]
    items = [
        item
        for item in declared
        if _pairs(baseline[item], candidate[item])
        and not _no_trim_ran(baseline[item], candidate[item])
    ]
    return {
        "n_declared": len(declared),
        "n_pairs": len(items),
        "entry_aware_wins": sum(_won(candidate[item], baseline[item]) for item in items),
        "head_tail_wins": sum(_won(baseline[item], candidate[item]) for item in items),
        "head_tail_completed": sum(bool(baseline[item]["success"]) for item in items),
        "entry_aware_completed": sum(bool(candidate[item]["success"]) for item in items),
    }


def _pairs(baseline: dict[str, object], candidate: dict[str, object]) -> bool:
    """Both arms folded the same offered transcript and neither ended on a refused prompt."""
    return (
        baseline["first_fold_input_chars"] == candidate["first_fold_input_chars"]
        and baseline["status"] != STATUS_CONTEXT_OVERFLOW
        and candidate["status"] != STATUS_CONTEXT_OVERFLOW
    )


def _no_trim_ran(baseline: dict[str, object], candidate: dict[str, object]) -> bool:
    """An arm that never folded ran no trim, so this case cannot be about the strategy."""
    return not int(cast(int, baseline["measured_folds"])) or not int(
        cast(int, candidate["measured_folds"])
    )


def _reading(readable: bool, wins: int, losses: int, n_pairs: int) -> str:
    if not readable or not n_pairs:
        return WORKLOAD_UNPAIRED
    if wins and losses:
        return WORKLOAD_MIXED
    if wins:
        return WORKLOAD_RECOVERS
    if losses:
        return WORKLOAD_REGRESSES
    return WORKLOAD_UNCHANGED


def _by_id(row: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        cast(str, case["item_id"]): case for case in cast(list[dict[str, object]], row["cases"])
    }


def _won(one: dict[str, object], other: dict[str, object]) -> bool:
    return bool(one["success"]) and not bool(other["success"])


def _rate(row: dict[str, object]) -> float:
    return float(cast(float, row["completion"]))


def _total(cases: dict[str, dict[str, object]], items: list[str], field: str) -> int:
    return sum(int(cast(int, cases[item][field])) for item in items)


def position_reading(run_rows: list[dict[str, object]]) -> dict[str, object]:
    """Whether the endpoint's request history, not the trim, moved this family's outcomes.

    Every task ran both arms back to back, so each task contributes exactly one first-position and
    one second-position episode. Reading the SAME episodes by position rather than by arm is
    therefore a clean nuisance-factor check on the identical data: if the second position loses
    folds or completions, the serving stack is doing it, because position and treatment are now
    orthogonal by construction.

    The verdict reads the FOLDING channel, not completion, and the asymmetry is deliberate.
    Whether an episode reaches its first fold at all is decided before the arms can diverge -- they
    build byte-identical prompts up to and including the transcript that fold offers -- so a
    position gap there cannot be the treatment and is the serving stack by elimination. It is also
    the dropout that costs this study its power, because a case that never folds leaves the paired
    comparison entirely. Completion is the opposite: it is the treatment's OWN outcome, so it moves
    with position whenever an arm's wins happen to fall unevenly across the two slots, and reading
    a position effect off it would report the recovery itself as a scheduling artifact. Both counts
    are reported; only the one that can be attributed decides the reading.
    """
    cases = _ordered_cases(run_rows)
    if not cases:
        return {}
    first = [case for case in cases if int(cast(int, case["order_position"])) == 1]
    second = [case for case in cases if int(cast(int, case["order_position"])) != 1]
    counts = {
        "n_episodes": len(cases),
        **_first_position_by_arm(first),
        "n_first": len(first),
        "n_second": len(second),
        "first_folded": sum(_folded(case) for case in first),
        "second_folded": sum(_folded(case) for case in second),
        "first_completed": sum(bool(case["success"]) for case in first),
        "second_completed": sum(bool(case["success"]) for case in second),
    }
    return {**counts, "reading": _order_reading(counts)}


def _ordered_cases(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Every case row that carries its execution position; empty under a fixed-order run."""
    return [
        case
        for row in run_rows
        for case in cast(list[dict[str, object]], row["cases"])
        if "order_position" in case
    ]


def _first_position_by_arm(first: list[dict[str, object]]) -> dict[str, int]:
    """How many tasks each arm opened -- the schedule's balance, counted from the episodes."""
    return {
        f"n_first_{arm}": sum(1 for case in first if case["first_arm"] == arm)
        for arm in (SUMMARY_TRIM_HEAD_TAIL, SUMMARY_TRIM_PER_ENTRY_HEAD)
    }


def _order_reading(counts: dict[str, int]) -> str:
    """Balanced within one task (the remainder of an odd set), and flat on the pre-fold channel."""
    if abs(counts["n_first_head_tail"] - counts["n_first_per_entry_head"]) > 1:
        return ORDER_UNBALANCED
    if counts["first_folded"] != counts["second_folded"]:
        return ORDER_POSITION_EFFECT
    return ORDER_NO_POSITION_EFFECT


def _folded(case: dict[str, object]) -> bool:
    """The episode reached the regime under test at all -- the dropout this study keeps meeting."""
    return bool(int(cast(int, case["measured_folds"])))
