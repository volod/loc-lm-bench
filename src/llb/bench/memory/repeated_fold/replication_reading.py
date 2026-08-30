"""Per-fold paired uncertainty and the cross-family reading for the replication.

The single-family completion reading grouped cases by measured fold count and compared marginal
rates. That is enough to see a ceiling and not enough to see a BOUND: a rate of 3/3 says nothing
about the interval around it, and two rates measured on the same tasks are paired evidence being
read as if it were independent. Every higher-fold case here is therefore paired against the SAME
task's one-fold control outcome, and each measured fold group carries its own interval plus the
paired ledger that produced it. A group below the predeclared evidence floor is reported and is
not allowed to cut the fold-count verdict -- an underpowered group can neither extend a rule nor
break one -- but a loss inside one is still NAMED, because "too few cases to read" is a reason to
report a signal with its width, never a reason to drop it.
"""

from typing import cast

from llb.conflicts.interval_stats import wilson_interval
from llb.rag.fusion_evidence.paired import PairedComparison, paired_comparison
from llb.rag.fusion_evidence.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_index_sets

ARM_TYPED_MARKER = "typed_marker"
ARM_SUMMARY_ONLY = "model_summary_only"

REPLICATION_EXTENDS = "fold_count_rule_extends_across_families"
REPLICATION_FAILS = "fold_count_rule_fails_on_a_qualified_family"
REPLICATION_INELIGIBLE = "fewer_than_required_qualified_families"
REPLICATION_UNDERPOWERED = "no_shared_powered_repeated_fold_count"


def fold_group_rows(
    cells: list[dict[str, object]], *, evidence_floor: int
) -> list[dict[str, object]]:
    """One row per measured fold count: its interval, its pairing, and whether it is powered."""
    control = _control_outcomes(cells)
    ablation = _ablation_outcomes(cells)
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in cells:
        if row["arm"] != ARM_TYPED_MARKER:
            continue
        for case in cast(list[dict[str, object]], row["cases"]):
            grouped.setdefault(int(cast(int, case["measured_folds"])), []).append(
                {
                    **case,
                    "cell_id": row["cell_id"],
                    "cap_fitting_control": bool(row["cap_fitting_control"]),
                }
            )
    return [
        _fold_row(folds, cases, control, ablation, evidence_floor)
        for folds, cases in sorted(grouped.items())
    ]


def powered_fold_limit(rows: list[dict[str, object]]) -> tuple[int | None, str]:
    """The highest fold count reached with floor-clearing evidence and no paired completion loss."""
    powered = [row for row in rows if row["meets_evidence_floor"]]
    if not powered or int(cast(int, powered[0]["measured_folds"])) != 1:
        return None, "no floor-clearing one-fold group anchors the family reading"
    limit = 1
    for row in powered[1:]:
        folds = int(cast(int, row["measured_folds"]))
        if _control_wins(row) > 0:
            return limit, (
                f"a paired case completes at one fold and fails at {folds} measured folds"
            )
        limit = folds
    return (
        limit,
        f"no paired case is lost through {limit} measured folds{_ladder_note(rows, limit)}",
    )


def _ladder_note(rows: list[dict[str, object]], limit: int) -> str:
    """Name what the ladder does not cover, so a limit is never read as continuous coverage.

    A fold count NOBODY measured is not evidence of a loss, and neither is a group too small to
    read -- but both are holes in what the limit above actually rests on, and a reader who cannot
    see them would take the number for more than it is.
    """
    measured = {
        int(cast(int, row["measured_folds"])) for row in rows if row["meets_evidence_floor"]
    }
    gaps = [folds for folds in range(1, limit) if folds not in measured]
    unpowered = [
        int(cast(int, row["measured_folds"])) for row in rows if not row["meets_evidence_floor"]
    ]
    losses = sorted(
        int(cast(int, row["measured_folds"]))
        for row in rows
        if not row["meets_evidence_floor"] and _control_wins(row) > 0
    )
    notes = []
    if gaps:
        notes.append(f"no case measured {gaps} folds")
    if unpowered:
        notes.append(f"fold groups below the evidence floor: {unpowered}")
    if losses:
        notes.append(f"under-floor groups that DID lose a paired case: {losses}")
    return f"; {'; '.join(notes)}" if notes else ""


def _control_wins(row: dict[str, object]) -> int:
    return int(cast(int, cast(dict[str, object], row["paired"])["control_wins"]))


def replication_reading(
    families: list[dict[str, object]], *, required_families: int
) -> tuple[str, str, list[dict[str, object]]]:
    """Extend the fold-count rule only as far as EVERY qualified family carries it."""
    qualified = [row for row in families if row["control_eligible"]][:required_families]
    if len(qualified) < required_families:
        return (
            REPLICATION_INELIGIBLE,
            f"only {len(qualified)} of {required_families} required families passed the one-fold "
            "control gate, so no cross-family fold-count rule is stated",
            qualified,
        )
    failing = [row for row in qualified if row["fold_count_lost_a_paired_case"]]
    if failing:
        first = failing[0]
        return (
            REPLICATION_FAILS,
            f"family {first['model_family']!r} ({first['model']}) loses a paired case above "
            f"{first['powered_fold_limit']} measured folds: {first['powered_fold_reason']}",
            qualified,
        )
    limits = [int(cast(int, row["powered_fold_limit"])) for row in qualified]
    shared = min(limits)
    if shared < 2:
        return (
            REPLICATION_UNDERPOWERED,
            f"the shared powered fold count is {shared}, so no family pair carries the rule past "
            "a single fold",
            qualified,
        )
    return (
        REPLICATION_EXTENDS,
        f"completion is stable through {shared} measured folds on all {len(qualified)} qualified "
        f"families (per-family powered limits {limits})",
        qualified,
    )


def _fold_row(
    folds: int,
    cases: list[dict[str, object]],
    control: dict[str, bool],
    ablation: dict[tuple[str, str], bool],
    evidence_floor: int,
) -> dict[str, object]:
    successes = [bool(case["success"]) for case in cases]
    lo, hi = wilson_interval(sum(successes), len(successes))
    paired = _paired_against_control(folds, cases, control)
    reference = all(case["cap_fitting_control"] for case in cases)
    n_evidence = len(cases) if reference else int(cast(int, paired["n_pairs"]))
    return {
        "measured_folds": folds,
        "n_cases": len(cases),
        "n_completed": sum(successes),
        "completion": sum(successes) / len(successes),
        "completion_lo": lo,
        "completion_hi": hi,
        "is_reference_group": reference,
        "evidence_kind": "control_cases" if reference else "paired_cases",
        "n_evidence": n_evidence,
        "meets_evidence_floor": n_evidence >= evidence_floor,
        "paired": paired,
        "marker_ablation": _ablation_ledger(cases, ablation),
    }


def _paired_against_control(
    folds: int, cases: list[dict[str, object]], control: dict[str, bool]
) -> dict[str, object]:
    """Pair every non-control case in this group against the same task's one-fold outcome."""
    pairs = [
        (control[cast(str, case["item_id"])], bool(case["success"]))
        for case in cases
        if not case["cap_fitting_control"] and cast(str, case["item_id"]) in control
    ]
    control_wins = sum(baseline and not candidate for baseline, candidate in pairs)
    group_wins = sum(candidate and not baseline for baseline, candidate in pairs)
    ledger: dict[str, object] = {
        "measured_folds": folds,
        "n_pairs": len(pairs),
        "control_wins": control_wins,
        "group_wins": group_wins,
        "unchanged": len(pairs) - control_wins - group_wins,
    }
    if pairs:
        ledger["delta"] = _delta(pairs)
    return ledger


def _delta(pairs: list[tuple[bool, bool]]) -> PairedComparison:
    candidate = [float(candidate) for _baseline, candidate in pairs]
    baseline = [float(baseline) for baseline, _candidate in pairs]
    return paired_comparison(
        candidate,
        baseline,
        bootstrap_index_sets(len(pairs), DEFAULT_RESAMPLES, DEFAULT_SEED),
    )


def _ablation_ledger(
    cases: list[dict[str, object]], ablation: dict[tuple[str, str], bool]
) -> dict[str, int]:
    """The marker ablation, restated inside one measured fold group."""
    pairs = [
        (bool(case["success"]), ablation[(cast(str, case["cell_id"]), cast(str, case["item_id"]))])
        for case in cases
        if (cast(str, case["cell_id"]), cast(str, case["item_id"])) in ablation
    ]
    marker_wins = sum(marker and not summary for marker, summary in pairs)
    summary_wins = sum(summary and not marker for marker, summary in pairs)
    return {
        "n_pairs": len(pairs),
        "marker_wins": marker_wins,
        "summary_only_wins": summary_wins,
        "unchanged": len(pairs) - marker_wins - summary_wins,
    }


def _control_outcomes(cells: list[dict[str, object]]) -> dict[str, bool]:
    return {
        cast(str, case["item_id"]): bool(case["success"])
        for row in cells
        if row["arm"] == ARM_TYPED_MARKER and row["cap_fitting_control"]
        for case in cast(list[dict[str, object]], row["cases"])
    }


def _ablation_outcomes(cells: list[dict[str, object]]) -> dict[tuple[str, str], bool]:
    return {
        (cast(str, row["cell_id"]), cast(str, case["item_id"])): bool(case["success"])
        for row in cells
        if row["arm"] == ARM_SUMMARY_ONLY
        for case in cast(list[dict[str, object]], row["cases"])
    }
