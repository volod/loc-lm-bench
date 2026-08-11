"""Per-stratum, cross-family, and conditional prototype readings for window elision."""

from typing import cast

from llb.bench.agentic_memory_window_elision_reading import (
    READING_COSTS,
    READING_FREE,
    completion_reading,
)
from llb.bench.agentic_memory_window_elision_tasks import STRATA, STRATUM_MIDDLE
from llb.bench.agentic_memory_window_elision_design import ROLE_ELIDED, ROLE_FIT

TRANSFER_MIDDLE_COSTS = "middle_critical_elision_costs_completion_across_families"
TRANSFER_FREE = "window_elision_costs_no_completion_across_strata_and_families"
TRANSFER_MIXED = "window_elision_transfer_is_mixed"
TRANSFER_INELIGIBLE = "fewer_than_two_qualified_model_families"
PROTOTYPE_RECOVERS = "entry_aware_fold_recovers_middle_completion"
PROTOTYPE_NO_RECOVERY = "entry_aware_fold_does_not_recover_middle_completion"
PROTOTYPE_NOT_RUN = "entry_aware_prototype_not_gated"


def family_stratum_reading(analysis: dict[str, object]) -> dict[str, object]:
    """Read fitting minus elided completion independently inside every evidence stratum."""
    cells = cast(list[dict[str, object]], analysis["cells"])
    by_role = {cast(str, row["role"]): row for row in cells}
    fit = cast(list[dict[str, object]], by_role.get(ROLE_FIT, {}).get("cases", []))
    elided = cast(list[dict[str, object]], by_role.get(ROLE_ELIDED, {}).get("cases", []))
    eligible = bool(analysis["comparison_eligible"])
    strata: dict[str, dict[str, object]] = {}
    for stratum in STRATA:
        fit_cases = [row for row in fit if row.get("evidence_stratum") == stratum]
        elided_cases = [row for row in elided if row.get("evidence_stratum") == stratum]
        reading, reason, paired = completion_reading(
            fit_cases,
            elided_cases,
            eligible=eligible,
            eligibility_reason=cast(str, analysis["eligibility_reason"]),
        )
        strata[stratum] = {"reading": reading, "reason": reason, "paired": paired}
    return {
        "model": analysis["model"],
        "backend": analysis["backend"],
        "eligible": eligible,
        "eligibility_reason": analysis["eligibility_reason"],
        "strata": strata,
    }


def transfer_reading(
    families: list[dict[str, object]], *, required_families: int
) -> tuple[str, str, bool, list[dict[str, object]]]:
    """Require the same stratum result on both qualified model families."""
    qualified = [row for row in families if row["eligible"]][:required_families]
    if len(qualified) < required_families:
        return (
            TRANSFER_INELIGIBLE,
            f"only {len(qualified)} of {required_families} required families qualified",
            False,
            qualified,
        )
    controls_free = all(
        _stratum_reading(row, stratum) == READING_FREE
        for row in qualified
        for stratum in ("head", "tail")
    )
    middle_costs = all(_stratum_reading(row, STRATUM_MIDDLE) == READING_COSTS for row in qualified)
    all_free = all(
        _stratum_reading(row, stratum) == READING_FREE for row in qualified for stratum in STRATA
    )
    if controls_free and middle_costs:
        return (
            TRANSFER_MIDDLE_COSTS,
            "both qualified families lose middle-critical completion while head and tail controls stay unchanged",
            True,
            qualified,
        )
    if all_free:
        return (
            TRANSFER_FREE,
            "no exact paired task changes in any stratum on either qualified family",
            False,
            qualified,
        )
    return (
        TRANSFER_MIXED,
        "the stratum-specific exact outcomes do not agree across both qualified families",
        False,
        qualified,
    )


def prototype_reading(
    qualified: list[dict[str, object]], prototype_rows: list[dict[str, object]]
) -> tuple[str, str, dict[str, object]]:
    """Whether entry-aware elision recovers middle cases without moving prompt bytes."""
    prototypes = {cast(str, row["model"]): row for row in prototype_rows}
    family_rows: list[dict[str, object]] = []
    for family in qualified:
        model = cast(str, family["model"])
        prototype = prototypes.get(model)
        if prototype is None:
            continue
        family_rows.append(_prototype_family(family, prototype))
    recovered = len(family_rows) == len(qualified) and all(
        row["middle_recovers"] and row["controls_unchanged"] and row["same_prompt_chars"]
        for row in family_rows
    )
    reading = PROTOTYPE_RECOVERS if recovered else PROTOTYPE_NO_RECOVERY
    reason = (
        "entry-aware trimming recovers every lost middle case on both families at identical summary prompt chars"
        if recovered
        else "entry-aware trimming does not cleanly recover middle completion on both families at fixed prompt chars"
    )
    return reading, reason, {"families": family_rows}


def _prototype_family(family: dict[str, object], prototype: dict[str, object]) -> dict[str, object]:
    strata = cast(dict[str, dict[str, object]], family["strata"])
    head_tail_cases = cast(list[dict[str, object]], prototype["head_tail_cases"])
    entry_cases = cast(list[dict[str, object]], prototype["entry_aware_cases"])
    by_id = {cast(str, row["item_id"]): row for row in head_tail_cases}
    candidate = {cast(str, row["item_id"]): row for row in entry_cases}
    outcomes, same_chars = _prototype_outcomes(by_id, candidate)
    middle_losses = int(
        cast(int, cast(dict[str, object], strata[STRATUM_MIDDLE]["paired"])["fit_wins"])
    )
    return {
        "model": family["model"],
        "middle_recovers": outcomes[STRATUM_MIDDLE]["entry_aware_wins"] == middle_losses
        and outcomes[STRATUM_MIDDLE]["head_tail_wins"] == 0,
        "controls_unchanged": _controls_unchanged(outcomes),
        "same_prompt_chars": same_chars,
        "strata": outcomes,
    }


def _prototype_outcomes(
    by_id: dict[str, dict[str, object]], candidate: dict[str, dict[str, object]]
) -> tuple[dict[str, dict[str, int]], bool]:
    outcomes: dict[str, dict[str, int]] = {}
    same_chars = True
    for stratum in STRATA:
        ids = [
            item_id
            for item_id, row in by_id.items()
            if row.get("evidence_stratum") == stratum and item_id in candidate
        ]
        entry_wins = sum(
            bool(candidate[item]["success"]) and not bool(by_id[item]["success"]) for item in ids
        )
        head_tail_wins = sum(
            bool(by_id[item]["success"]) and not bool(candidate[item]["success"]) for item in ids
        )
        outcomes[stratum] = {
            "entry_aware_wins": entry_wins,
            "head_tail_wins": head_tail_wins,
            "unchanged": len(ids) - entry_wins - head_tail_wins,
        }
        same_chars = same_chars and all(
            candidate[item]["compaction_prompt_chars"] == by_id[item]["compaction_prompt_chars"]
            for item in ids
        )
    return outcomes, same_chars


def _controls_unchanged(outcomes: dict[str, dict[str, int]]) -> bool:
    return all(
        outcomes[stratum]["entry_aware_wins"] == 0 and outcomes[stratum]["head_tail_wins"] == 0
        for stratum in ("head", "tail")
    )


def _stratum_reading(family: dict[str, object], stratum: str) -> str:
    strata = cast(dict[str, dict[str, object]], family["strata"])
    return cast(str, strata[stratum]["reading"])
