"""What a paired reading is allowed to SAY, and the item count that entitles it to say it.

Every adopt-or-retain call in this repo is a binary cut of a paired interval: does the delta's
lower bound clear zero. That test is silent about how much evidence produced it. A paired block's
evidence is its DISCORDANT items -- the pairs where the two lanes actually differ, since the ties
carry no information about the direction -- and a percentile bootstrap of a handful of them prints
exactly like a decisive result: a 2-item slice with 2 wins publishes `+1.000 [+1.000, +1.000]`,
because resampling two differing items can only ever draw those two.

The exact two-sided sign test the same block already reports bounds what `d` discordant items can
ever show. Its smallest attainable p is `2 * 0.5^d` -- every pair falling the same way, the most
extreme arrangement there is -- so below `minimum_discordant_pairs(confidence)` items the reporting
level is UNREACHABLE whatever the data. `apply_evidence_gate` relabels a claim resting on fewer
than that `insufficient_evidence`, and no verdict in the repo may be decided on a row carrying that
state.

The bound is DERIVED from the reporting confidence rather than chosen, so this introduces no
constant to tune: 5 items at 90%, 6 at 95%, 7 at 97.5%. That the bound moves with the level is also
why a row can read `separated` at the reporting level and `insufficient_evidence` one convention
tighter -- exactly the flip the borderline flag in `llb.rag.fusion_evidence.stability` exists to
mark, on the same scale.

A relabeled reading leaves an OPEN QUESTION, not a finding of no difference, so this module also
says what would close it. `resolving_item_count` prices the item set: at the block's own observed
discordance rate, how many items it takes before the reporting level is reachable at all. That is a
FLOOR on the item count derived from the same arithmetic as the bound -- necessary, not sufficient,
and not a power target: it says nothing about the effect size the larger set could resolve, which
is what an a priori MDE contract prices.

This module owns the reading VOCABULARY and the gate. `stability.py` (how settled a reading is) and
`stats.py` (the percentile arithmetic) both import from here, never the other way round.

Pure Python and dependency-free, so it imports and is unit-tested in the lightweight CI install.
"""

import math
from collections.abc import Sequence

DEFAULT_CONFIDENCE = 0.95

# The two-state reading of ONE paired delta -- what most lanes cut from their interval. A lane with
# a richer reading supplies its own state names (the adoption bar reads three).
READING_SEPARATED = "separated"
READING_FLAT = "flat"

# The gate's own state: the delta's interval clears zero, but too few items differ for the
# reporting level to be reachable, so the row states nothing. It is NOT a third outcome any lane
# may adopt on -- it is the absence of a readable one, and it is an OPEN question rather than a
# measured absence of a difference (`resolving_item_count` prices what would close it).
READING_INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def reading_label(reading: str) -> str:
    """`rank_only` -> `rank only` -- the one rendering of a reading every table shares."""
    return reading.replace("_", " ")


def level_label(confidence: float) -> str:
    """`0.975` -> `97.5%` -- a percentage that keeps the half point the 97.5% level needs."""
    return f"{confidence * 100:g}%"


def smallest_reachable_sign_test_p(discordant: int) -> float:
    """The smallest two-sided sign-test p that `discordant` non-tied pairs can produce.

    `2 * 0.5^d`: every pair falling the same way is the most extreme arrangement there is, so this
    is a property of the item count alone -- no data needed, and nothing to fit.
    """
    return min(1.0, 2.0 * 0.5**discordant)


def minimum_discordant_pairs(confidence: float = DEFAULT_CONFIDENCE) -> int:
    """Fewest discordant items whose exact sign test can reach `1 - confidence` (6 at 95%).

    Derived from the reporting confidence rather than chosen: it is the point below which the test
    the block already prints beside its interval cannot produce a significant p under ANY
    arrangement of the data.
    """
    alpha = 1.0 - confidence
    if alpha <= 0.0:
        raise ValueError(f"no discordant count can reach a confidence of {confidence}")
    discordant = 0
    while smallest_reachable_sign_test_p(discordant) > alpha:
        discordant += 1
    return discordant


def reaches_reporting_level(discordant: int, confidence: float = DEFAULT_CONFIDENCE) -> bool:
    """Whether `discordant` differing items could support a reading at `confidence` at all."""
    return discordant >= minimum_discordant_pairs(confidence)


def apply_evidence_gate(
    reading: str,
    *,
    discordant: int,
    confidence: float = DEFAULT_CONFIDENCE,
    null_reading: str = READING_FLAT,
) -> str:
    """`reading`, or `insufficient_evidence` when the item count cannot support the claim.

    Only a reading that CLAIMS a difference is gated. The null reading (`flat`, or whatever a
    richer lane calls its no-separation state) is a statement that nothing was found, which a thin
    item set is entitled to make -- relabelling it would turn "no evidence of a difference" into a
    second, redundant way of saying the same thing.
    """
    if reading == null_reading or reaches_reporting_level(discordant, confidence):
        return reading
    return READING_INSUFFICIENT_EVIDENCE


def resolving_item_count(
    discordant: int, pairs: int, confidence: float = DEFAULT_CONFIDENCE
) -> int | None:
    """Items this block would need, at its OWN discordance rate, for `confidence` to be reachable.

    The gate is a statement about DISCORDANT items, so the item count that answers it follows from
    the rate at which this comparison produces them: `d` differing items out of `n` extrapolates to
    `minimum_discordant_pairs(confidence) * n / d`, rounded up.

    A floor, and only that. It is the point below which no arrangement of the data could reach the
    level, so a set smaller than it cannot decide the question whatever the effect is; a set larger
    than it still resolves only effects big enough to survive the interval. Reading it as "run this
    many items and the question is answered" is the one misreading to avoid -- what that would need
    is an a priori minimum detectable effect, which this repo prices separately.

    None when the level is already reachable (nothing to resolve) or when nothing differed, since a
    rate of zero extrapolates to no item count at all.
    """
    if pairs <= 0 or discordant <= 0 or reaches_reporting_level(discordant, confidence):
        return None
    return math.ceil(minimum_discordant_pairs(confidence) * pairs / discordant)


def _gated(marked: Sequence[tuple[str, int, int]], confidence: float) -> list[tuple[str, int, int]]:
    """The named rows whose discordant count cannot reach `confidence`, in the order given."""
    return [row for row in marked if not reaches_reporting_level(row[1], confidence)]


def evidence_gate_note(
    marked: Sequence[tuple[str, int, int]], confidence: float = DEFAULT_CONFIDENCE
) -> str:
    """The clause a verdict appends when a row it named cannot support a separation, else "".

    ONE phrasing, shared by every lane, exactly as `borderline_note` is. Pass every row the verdict
    would otherwise have been DECIDED ON as `(label, discordant, pairs)`; rows that clear the
    reachable minimum drop out, and the clause is empty when none is left. The item count that
    would resolve each surviving row is appended by `open_question_note`, so a verdict never states
    that a reading was withdrawn without stating what would restore it.
    """
    gated = _gated(marked, confidence)
    if not gated:
        return ""
    needed = minimum_discordant_pairs(confidence)
    detail = "; ".join(
        f"`{label}` differs on {discordant} of {pairs}" for label, discordant, pairs in gated
    )
    subject = "the row" if len(gated) == 1 else f"{len(gated)} of the rows"
    verb = "carries" if len(gated) == 1 else "carry"
    return (
        f". INSUFFICIENT EVIDENCE: {subject} that would otherwise decide this verdict {verb} fewer "
        f"than the {needed} differing items an exact sign test needs to reach "
        f"{level_label(confidence)} ({detail}), so the reading is not a separation and the decision "
        "does not rest on it"
    ) + open_question_note(gated, confidence)


def _resolving_detail(
    label: str, discordant: int, pairs: int, confidence: float = DEFAULT_CONFIDENCE
) -> str:
    """``recall_at_k` 4 of 35 -> 53 items` -- one row's price, or why it has none."""
    required = resolving_item_count(discordant, pairs, confidence)
    if required is None:
        return f"`{label}` has no differing item, so its rate prices no item count"
    return f"`{label}` {discordant} of {pairs} -> {required} items"


def open_question_note(
    marked: Sequence[tuple[str, int, int]], confidence: float = DEFAULT_CONFIDENCE
) -> str:
    """The clause that says what would settle a reading the gate withdrew, else "".

    A relabeled row is an unanswered question, not an answer of no difference, and the two must not
    print alike -- an operator acting on a recorded recommendation needs to know which one they are
    holding and what it would take to close it.
    """
    gated = _gated(marked, confidence)
    if not gated:
        return ""
    detail = "; ".join(
        _resolving_detail(label, discordant, pairs, confidence)
        for label, discordant, pairs in gated
    )
    return (
        ". OPEN QUESTION: this is an unanswered comparison rather than a measured absence of a "
        f"difference; at each row's own discordance rate {level_label(confidence)} stays "
        f"unreachable below the item count named here ({detail}), which is a floor on the item set "
        "and not a minimum-detectable-effect target"
    )


def evidence_gate_summary(
    n_gated: int, n_rows: int, confidence: float = DEFAULT_CONFIDENCE, *, subject: str = "reading"
) -> list[str]:
    """The line a report prints about how many of its rows the gate relabeled, as Markdown lines.

    Always rendered, including when nothing was relabeled: "0 of 48" is the statement an operator
    needs in order to read the rest of the table at face value.
    """
    if n_rows <= 0:
        return []
    needed = minimum_discordant_pairs(confidence)
    return [
        f"Minimum evidence: **{n_gated} of {n_rows}** paired {subject}s are "
        f"`{reading_label(READING_INSUFFICIENT_EVIDENCE)}` -- the delta's interval clears zero but "
        f"fewer than {needed} items differ, which is the fewest an exact two-sided sign test needs "
        f"to reach {level_label(confidence)} under any arrangement of the data. No verdict is "
        "decided on such a row, and it is an OPEN question rather than a measured absence of a "
        "difference: the `n to reach` column prices the item set that would close it.",
        "",
    ]
