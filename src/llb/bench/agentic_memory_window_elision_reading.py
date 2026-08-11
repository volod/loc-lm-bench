"""Exact paired completion reading for the unavoidable window-elision study."""

from typing import cast

READING_FREE = "window_elision_costs_no_completion"
READING_COSTS = "window_elision_costs_completion"
READING_IMPROVES = "window_elision_improves_completion"
READING_MIXED = "mixed_or_inconclusive_window_elision_effect"
READING_INELIGIBLE = "window_elision_comparison_ineligible"


def completion_reading(
    fit_cases: list[dict[str, object]],
    elided_cases: list[dict[str, object]],
    *,
    eligible: bool,
    eligibility_reason: str,
) -> tuple[str, str, dict[str, object]]:
    """Compare exact per-task outcomes; make no population-level statistical claim."""
    fit, elided, shared = _paired_outcomes(fit_cases, elided_cases)
    paired = _paired_counts(fit, elided, shared)
    complete_pairing = len(shared) == len(fit) and len(shared) == len(elided)
    if not eligible or not complete_pairing:
        return READING_INELIGIBLE, eligibility_reason, paired
    return _outcome_reading(paired)


def _paired_outcomes(
    fit_cases: list[dict[str, object]], elided_cases: list[dict[str, object]]
) -> tuple[dict[str, bool], dict[str, bool], list[str]]:
    """Index both arms by task id and name their exact shared pairs."""
    fit = {cast(str, row["item_id"]): bool(row["success"]) for row in fit_cases}
    elided = {cast(str, row["item_id"]): bool(row["success"]) for row in elided_cases}
    shared = sorted(fit.keys() & elided.keys())
    return fit, elided, shared


def _paired_counts(
    fit: dict[str, bool], elided: dict[str, bool], shared: list[str]
) -> dict[str, object]:
    """Count directional wins and the direct completion-rate difference."""
    fit_wins = sum(fit[item] and not elided[item] for item in shared)
    elided_wins = sum(elided[item] and not fit[item] for item in shared)
    unchanged = len(shared) - fit_wins - elided_wins
    return {
        "n_pairs": len(shared),
        "fit_wins": fit_wins,
        "elided_wins": elided_wins,
        "unchanged": unchanged,
        "completion_delta": (
            sum(fit.values()) / len(fit) - sum(elided.values()) / len(elided)
            if fit and elided
            else 0.0
        ),
    }


def _outcome_reading(paired: dict[str, object]) -> tuple[str, str, dict[str, object]]:
    """Select the bounded task-set verdict from directional paired counts."""
    fit_wins = int(cast(int, paired["fit_wins"]))
    elided_wins = int(cast(int, paired["elided_wins"]))
    if fit_wins and not elided_wins:
        return (
            READING_COSTS,
            f"the fitting control wins {fit_wins} exact paired tasks and loses none",
            paired,
        )
    if elided_wins and not fit_wins:
        return (
            READING_IMPROVES,
            f"the elided arm wins {elided_wins} exact paired tasks and loses none",
            paired,
        )
    if not fit_wins and not elided_wins:
        return (
            READING_FREE,
            f"eliding the declared middle span changes none of {paired['n_pairs']} exact paired tasks",
            paired,
        )
    return (
        READING_MIXED,
        f"paired outcomes conflict: fit wins={fit_wins}, elided wins={elided_wins}",
        paired,
    )


def operator_recommendation(reading: str) -> str:
    """Turn the measured task-set reading into a bounded operator action."""
    if reading == READING_COSTS:
        return (
            "evaluate an entry-aware fold before using this over-window geometry; the current "
            "head-and-tail trim lost completion on measured tasks"
        )
    if reading == READING_FREE:
        return (
            "keep the shipped fold on this measured geometry; its unavoidable middle elision "
            "did not change completion"
        )
    return (
        "do not change the shipped fold from this run; the paired completion reading is not clean"
    )
