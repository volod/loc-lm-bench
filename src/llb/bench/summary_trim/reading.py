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

# A workload whose fold elides nothing cannot say anything about the strategies, so it is a
# CONTROL: it must stay byte-identical, and it never contributes evidence for adoption.
CONTROL_REASON = "the fold fits the summarize-input bound, so both arms render the same prompt"


def workload_reading(
    baseline_row: dict[str, object], candidate_row: dict[str, object]
) -> dict[str, object]:
    """One workload's paired comparison between the shipped trim and the entry-aware one."""
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
    return {"workloads": readings, "strata": _stratum_outcomes(by_key, names)}


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
