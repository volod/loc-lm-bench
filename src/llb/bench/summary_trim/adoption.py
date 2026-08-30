"""The cross-family adoption verdict: adopt as default, ship as an option, or keep head_tail.

The study answers TWO questions with different evidence bars, and collapsing them would misreport
both. "Is the entry-aware trim safe to offer an operator?" needs no workload to regress and the
recovery to hold on every middle pair the run could read. "Should it replace the shipped default?"
needs all of that PLUS an execution order that cannot stand in for the treatment, a fully powered
middle stratum, no extra window bytes on any workload, and a policy-change audit that retires no
published cell -- because a default change is a product decision
every future run inherits, while an option is one an operator opts into for a session they can see.

So the gates form a ladder rather than a single pass/fail. A regression or an unreadable comparison
refuses the strategy outright; extra bytes, an unclean audit, or a middle stratum that did not reach
its declared size in usable pairs each downgrade a default change to an option, naming which one
did it; only a run that clears everything recommends moving the default. The audit is the half of
the question that lives outside this study, so it enters as a gate rather than as a footnote.
"""

from typing import cast

from llb.bench.memory.window_elision.tasks import STRATUM_MIDDLE
from llb.bench.summary_trim.reading import (
    ORDER_NO_POSITION_EFFECT,
    ORDER_POSITION_EFFECT,
    WORKLOAD_MIXED,
    WORKLOAD_REGRESSES,
    WORKLOAD_UNPAIRED,
)

ADOPT_AS_DEFAULT = "adopt_entry_aware_as_the_shipped_default"
ADOPT_AS_OPTION = "ship_entry_aware_as_a_supported_option"
ADOPT_REFUSE = "keep_head_tail_and_do_not_ship_entry_aware"
ADOPT_INELIGIBLE = "fewer_than_two_qualified_model_families"
# The middle stratum yielded NO usable pair in some family, so the run cannot read the recovery the
# strategy exists for -- in either direction. A stratum that yields SOME but not all of its declared
# pairs is not this: it is a downgrade to `ADOPT_AS_OPTION`, because what it read still held.
ADOPT_INCONCLUSIVE = "middle_stratum_unreadable_in_this_run"


def adoption_reading(
    families: list[dict[str, object]],
    *,
    required_families: int,
    audit_invariant: bool | None,
    required_middle_pairs: int = 0,
) -> tuple[str, str]:
    """The cross-family verdict, and whether the shipped default may move with it.

    `audit_invariant` is the policy-change audit's answer for this field under the pinned policy:
    a default change is only recommendable when moving it retires no published cell. `None` means
    the audit was not supplied, which is reported rather than assumed either way.
    """
    if len(families) < required_families:
        return (
            ADOPT_INELIGIBLE,
            f"only {len(families)} of {required_families} required families qualified",
        )
    readings = [
        row for family in families for row in cast(list[dict[str, object]], family["workloads"])
    ]
    recovery_verdict, recovery_note = _middle_recovery(families)
    refusal = _refusal(readings, recovery=(recovery_verdict, recovery_note))
    if refusal is not None:
        return refusal
    downgrade = _downgrade(
        readings,
        families,
        audit_invariant=audit_invariant,
        required_middle_pairs=required_middle_pairs,
    )
    if downgrade is not None:
        return downgrade
    excluded = sum(int(cast(int, row["n_unpaired_no_fold"])) for row in readings)
    note = f" ({excluded} case(s) excluded for never folding in one arm)" if excluded else ""
    return (
        ADOPT_AS_DEFAULT,
        f"no workload regresses{note}, {recovery_note}, arm order is balanced across every task "
        "set, no workload spends more summary prompt bytes, and the audit retires no published "
        "cell",
    )


def _refusal(
    readings: list[dict[str, object]], *, recovery: tuple[str | None, str]
) -> tuple[str, str] | None:
    """What stops the run short of a recommendation: an unreadable, losing, or thin comparison."""
    unpaired = [row for row in readings if row["reading"] == WORKLOAD_UNPAIRED]
    if unpaired:
        names = sorted({cast(str, row["workload"]) for row in unpaired})
        divergent = sum(int(cast(int, row["n_unpaired_divergent_fold"])) for row in unpaired)
        return (
            ADOPT_REFUSE,
            f"{names} is unreadable: {divergent} case(s) folded and still offered the summarizer "
            "different bytes in the two arms, which is not separable from the trim",
        )
    regressed = [row for row in readings if row["reading"] in (WORKLOAD_REGRESSES, WORKLOAD_MIXED)]
    if regressed:
        names = sorted({cast(str, row["workload"]) for row in regressed})
        return ADOPT_REFUSE, f"entry-aware trimming loses paired completion on {names}"
    verdict, note = recovery
    return (verdict, note) if verdict is not None else None


def _downgrade(
    readings: list[dict[str, object]],
    families: list[dict[str, object]],
    *,
    audit_invariant: bool | None,
    required_middle_pairs: int,
) -> tuple[str, str] | None:
    """What makes it shippable as an OPTION but not as the default.

    Four things do, and each is reported by name: an execution order that leaves "ran second"
    aligned with an arm, a workload that spends more window bytes, an audit that retires a
    published cell, and a middle stratum that read cleanly but on fewer cases than the design
    declared. The last one is a statement about POWER, not about direction -- every pair it did
    read recovered -- so it withholds the default without withholding the option.
    """
    confounded = _confounded_order(families)
    if confounded is not None:
        return ADOPT_AS_OPTION, confounded
    thin = _under_powered(families, required_middle_pairs)
    if thin is not None:
        return ADOPT_AS_OPTION, thin
    costlier = [row for row in readings if int(cast(int, row["d_summary_prompt_chars"])) > 0]
    if costlier:
        names = sorted({cast(str, row["workload"]) for row in costlier})
        return ADOPT_AS_OPTION, f"entry-aware trimming spends more summary prompt bytes on {names}"
    if audit_invariant is not True:
        return (
            ADOPT_AS_OPTION,
            "moving the shipped default is not cleared: the policy-change audit does not report "
            "the published cells invariant under this field",
        )
    return None


def _confounded_order(families: list[dict[str, object]]) -> str | None:
    """A family whose arms did not run in a balanced order, so position could carry the result.

    A run with no order reading at all is the FIXED-order shape this study started in: its arms ran
    as whole blocks, so "the second arm" and "the arm under test" are one column and no amount of
    clean completion evidence separates them. It is offered as an option on what it read and stops
    there, which is the same conclusion the first reading of this study reached by hand.
    """
    for family in families:
        order = cast(dict[str, object], family.get("arm_order") or {})
        reading = order.get("reading")
        if reading is None:
            return (
                f"{family['model_family']} ran its arms in fixed blocks, so 'ran second' and 'ran "
                "under the entry-aware trim' are the same column and no dropout in it is "
                "attributable"
            )
        if reading not in (ORDER_NO_POSITION_EFFECT, ORDER_POSITION_EFFECT):
            return (
                f"{family['model_family']} did not execute a balanced arm order "
                f"({int(cast(int, order['n_first_head_tail']))} vs "
                f"{int(cast(int, order['n_first_per_entry_head']))} first positions), so order is "
                "not separable from the trim"
            )
    return None


def _under_powered(families: list[dict[str, object]], required_pairs: int) -> str | None:
    """A family whose middle stratum read fewer usable pairs than the design declared.

    The guard fit travels with the shortfall, because the two readings a thin stratum can have are
    not the same finding: a guard that was never fitted to this family leaves the shortfall open,
    while a fit that exhausted its declared band says the walk itself is what the stratum is short
    of -- and names the guard that came closest.
    """
    for family in families:
        strata = cast(dict[str, dict[str, int]], family["strata"])
        middle = strata.get(STRATUM_MIDDLE, {})
        usable = int(middle.get("n_pairs", 0))
        if usable < required_pairs:
            fit = cast(dict[str, object], family.get("guard_fit") or {})
            note = f"; the guard fit reports: {fit['fit_reason']}" if fit else ""
            return (
                f"the middle stratum is under-powered: {family['model_family']} put only {usable} "
                f"of {required_pairs} declared middle cases into the folding regime, so the "
                f"recovery is established on what ran rather than on the whole declared stratum"
                f"{note}"
            )
    return None


def _middle_recovery(families: list[dict[str, object]]) -> tuple[str | None, str]:
    """Whether every qualified family ends with the middle stratum whole under the entry-aware trim.

    Returns the verdict that BLOCKS a recommendation, or `None` plus the note to quote when nothing
    does. Three outcomes, and they are deliberately not one: a stratum that did not reach its
    declared size in usable pairs is INCONCLUSIVE rather than refused -- the run simply did not put
    those episodes into the regime under test -- while a middle case the entry-aware trim leaves
    unfinished, or one it loses that the `head_tail` reference completed, is a refusal.

    The whole-stratum condition is stated as an OUTCOME rather than as a win count, because the two
    shapes an accepting run can take are both fine: `head_tail` loses middle cases and the
    entry-aware trim recovers all of them, or `head_tail` loses none and there is nothing to
    recover.
    """
    recovered = 0
    for family in families:
        strata = cast(dict[str, dict[str, int]], family["strata"])
        middle = strata.get(STRATUM_MIDDLE)
        if middle is None or not middle["n_pairs"]:
            return ADOPT_INCONCLUSIVE, "no qualified family measured the middle evidence stratum"
        if middle["head_tail_wins"]:
            return ADOPT_REFUSE, (
                f"{family['model_family']} loses {middle['head_tail_wins']} middle case(s) under "
                "the entry-aware trim that the `head_tail` reference completed"
            )
        if middle["entry_aware_completed"] != middle["n_pairs"]:
            return ADOPT_REFUSE, (
                f"{family['model_family']} leaves "
                f"{middle['n_pairs'] - middle['entry_aware_completed']} middle case(s) "
                "unfinished under the entry-aware trim"
            )
        recovered += middle["entry_aware_wins"]
    if recovered:
        return None, (
            f"the entry-aware trim recovers {recovered} middle case(s) `head_tail` lost and "
            "leaves the middle stratum whole in every qualified family"
        )
    return None, (
        "the `head_tail` reference lost no middle case on this task set, so there was nothing "
        "to recover "
        "and the entry-aware trim leaves the middle stratum whole"
    )
