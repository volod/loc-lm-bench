"""Is a paired row's reading settled evidence, or is it the threshold talking?

Every adopt-or-retain call in this repo is a BINARY cut of a continuous paired interval by one
test: does the delta's lower bound clear zero. A row whose bound sits ON zero therefore prints
exactly like a row that missed by a mile -- the bake-off's `retain`, the fusion sweep's `reject`,
the ablation's `no_retrieval_gain` and the answer lane's `no_gain` are all stated in the same words
whether the evidence was decisive or the convention was.

Two additive signals fix that, and neither changes any adoption rule:

- `p_positive`, the share of paired resamples in which the candidate is ahead. It is the CONTINUOUS
  quantity the reading thresholds: a 95% percentile interval clears zero exactly when
  `p_positive > 0.975`, so one number says both what the reading is and how far from the cut it sits.
- `borderline`, a flag raised when the reading would CHANGE under a looser or a tighter but equally
  conventional confidence level. That is a statement about robustness to an arbitrary convention,
  not a fitted threshold: no constant is tuned to the data, and all three levels compared are ones
  the repo already reports at.

A reading also has to be ENTITLED to what it says, which is a different question from how settled
it is: `llb.rag.fusion_evidence.evidence_gate` owns the reading vocabulary and relabels a claim
that rests on too few differing items `insufficient_evidence`. This module imports that vocabulary
and records the discordant count beside every annotation, because the gate's own bound moves with
the confidence level -- so a row can read `separated` here and `insufficient_evidence` one
convention tighter, which is precisely a `borderline` row.

This module is the ASSEMBLY of a row's readings; the percentile arithmetic stays in `stats.py`,
which imports from here (never the other way round). A lane whose reading is richer than
separated/flat -- the embedder adoption bar reads three states -- computes its own reading at each
of the three levels and hands them to `stability_from_readings`, so every lane persists and renders
the identical shape.

Pure Python and dependency-free, so it imports and is unit-tested in the lightweight CI install.
"""

from collections.abc import Sequence

from typing_extensions import NotRequired, TypedDict

from llb.rag.fusion_evidence.evidence_gate import (
    DEFAULT_CONFIDENCE,
    READING_INSUFFICIENT_EVIDENCE,
    level_label,
    minimum_discordant_pairs,
    reading_label,
    resolving_item_count,
)

# The two neighbouring conventional confidences a reading is re-checked at. All three levels are
# standard reporting conventions, so a reading that differs across them was decided by the
# convention rather than by the evidence. Neither is fitted to a recorded row.
LOOSER_CONFIDENCE = 0.90
TIGHTER_CONFIDENCE = 0.975

# Marks a reading that flips at either neighbouring level. It is a QUALIFIER on a lane's readings,
# never an extra outcome any verdict can adopt on.
BORDERLINE_MARK = "borderline"

# Which side of the cut a borderline row sits on. `below` fails the bar but would clear a looser
# one; `above` clears it but a tighter one would drop it. The distinction matters to an operator:
# `below` is an undecided negative, `above` is a positive resting on the convention.
SIDE_BELOW = "below"
SIDE_ABOVE = "above"


class ReadingStability(TypedDict):
    """One row's reading plus how close it sits to the cut that produced it.

    The same shape for every lane, so a boundary table, a persisted artifact field, and a qualified
    verdict sentence all read identically whichever comparison produced them.
    """

    reading: str
    # Share of paired resamples in which the delta is above zero. The reading's own threshold in
    # this scale is `1 - (1 - confidence) / 2` (0.975 at the default 95%).
    p_positive: float
    # What a looser / tighter interval would read; both equal `reading` when the row is settled.
    looser_reading: str
    tighter_reading: str
    borderline: bool
    # SIDE_BELOW / SIDE_ABOVE, or None when settled.
    side: str | None
    # Items whose two lanes differ -- the evidence the sign test and the gate are read on, measured
    # on the metric `p_positive` comes from. Absent on an artifact recorded before the gate existed.
    discordant: NotRequired[int]
    # Items compared, ties included. Carried beside `discordant` so a relabeled row can price the
    # item set that would resolve it from its own discordance rate; absent on the same artifacts.
    pairs: NotRequired[int]


def decision_probability(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """The `p_positive` a reading must exceed to clear zero at `confidence` (0.975 at 95%)."""
    return 1.0 - (1.0 - confidence) / 2.0


def assert_neighbouring_levels(
    confidence: float, looser_confidence: float, tighter_confidence: float
) -> None:
    """The flag is only meaningful when the two neighbours actually bracket the reporting level."""
    if not looser_confidence < confidence < tighter_confidence:
        raise ValueError(
            f"the neighbouring levels must straddle the reporting confidence: "
            f"{looser_confidence} < {confidence} < {tighter_confidence}"
        )


def brackets(
    confidence: float,
    looser_confidence: float = LOOSER_CONFIDENCE,
    tighter_confidence: float = TIGHTER_CONFIDENCE,
) -> bool:
    """Whether the annotation is defined at this reporting level, without raising.

    A caller that annotates as a side effect of an ordinary comparison (`paired_comparison`) must
    not fail a whole run because the operator asked for a 99% interval; it simply reports no
    annotation there.
    """
    return looser_confidence < confidence < tighter_confidence


def exceedance(samples: Sequence[float]) -> float:
    """Share of bootstrap resample MEANS above zero -- a one-sided bootstrap p-value.

    Takes the already-drawn resample means rather than re-resampling, because every caller has
    just computed them to take its percentile bounds. That is what makes the annotation free: the
    continuous quantity and the binary cut come out of one pass over the same draw.
    """
    if not samples:
        return 0.0
    return sum(1 for value in samples if value > 0.0) / len(samples)


def stability_from_readings(
    *,
    reading: str,
    looser_reading: str,
    tighter_reading: str,
    p_positive: float,
    discordant: int,
    pairs: int,
) -> ReadingStability:
    """Assemble the flag and the side from three already-computed readings of the same draw.

    The check is TWO-SIDED: a row that fails the bar but would clear a looser one is an undecided
    negative, and a row that clears the bar but a tighter one would drop is a positive resting on
    the convention. Reporting only the first would leave every near-miss positive looking settled.

    All three readings must come from the SAME resample draw, so the only thing that differs
    between them is the percentile cut -- the comparison isolates the confidence convention rather
    than mixing in a second draw's noise. Each is expected to have passed `apply_evidence_gate` at
    ITS OWN level already, because the reachable item count is a function of the level: a caller
    gates where it reads, and this function only records what came out.

    `pairs` is recorded beside `discordant` for the same reason the gate needs both: the two
    together are the discordance RATE, which is what prices the item set a withdrawn reading would
    need before it could be read at all.
    """
    return {
        "reading": reading,
        "p_positive": p_positive,
        "looser_reading": looser_reading,
        "tighter_reading": tighter_reading,
        "borderline": looser_reading != reading or tighter_reading != reading,
        "side": SIDE_BELOW
        if looser_reading != reading
        else (SIDE_ABOVE if tighter_reading != reading else None),
        "discordant": discordant,
        "pairs": pairs,
    }


def format_reading(stability: ReadingStability | None, reading: str) -> str:
    """`flat` / `flat (borderline)` -- the reading as a table cell.

    Falls back to the bare reading when no stability was measured, which is what happens on an
    artifact recorded before the annotation existed or a run that drew no resamples.
    """
    label = reading_label(reading)
    if stability is None or not stability["borderline"]:
        return label
    return f"{label} ({BORDERLINE_MARK})"


def borderline_detail(stability: ReadingStability) -> str:
    """`p_positive 0.981, a 0.975 interval would read it `flat`` -- one row's distance from the cut."""
    flip = (
        f"a {TIGHTER_CONFIDENCE:.3f} interval would read it `{stability['tighter_reading']}`"
        if stability["side"] == SIDE_ABOVE
        else f"a {LOOSER_CONFIDENCE:.2f} interval would read it `{stability['looser_reading']}`"
    )
    return f"p_positive {stability['p_positive']:.3f}, {flip}"


def borderline_note(rows: Sequence[tuple[str, ReadingStability | None]]) -> str:
    """The clause a verdict appends when a row it just named sits ON the cut, else "".

    ONE phrasing, shared by every lane, so an operator who has read it once in the bake-off knows
    exactly what it means in the fusion sweep. Pass every row the verdict was DECIDED ON, not only
    the one its sentence quotes: the ablation's `rag_pays_off` is reached by the long-context check
    failing first, so a knife-edge long-context delta is exactly what put that verdict where it is.
    Settled rows and unannotated ones drop out, and the clause is empty when none survives.
    """
    marked = [(label, entry) for label, entry in rows if entry is not None and entry["borderline"]]
    if not marked:
        return ""
    detail = "; ".join(f"`{label}` {borderline_detail(entry)}" for label, entry in marked)
    subject = "the row" if len(marked) == 1 else f"{len(marked)} of the rows"
    verb = "rests" if len(marked) == 1 else "rest"
    return (
        f". BORDERLINE: {subject} this verdict was decided on {verb} on a reading a neighbouring "
        f"conventional confidence level would change ({detail}), so read it as too close to call "
        "rather than as settled evidence"
    )


def unsettled(stability: ReadingStability | None) -> ReadingStability | None:
    """`stability` when a neighbouring convention would read it differently, else None."""
    return stability if stability is not None and stability["borderline"] else None


def _resolving_cell(stability: ReadingStability, confidence: float) -> str:
    """The item count that would make a reading of this row reachable, or `-` when there is none.

    Priced for every row short of the bound, not only the ones whose reading was withdrawn: a
    `flat` row on three differing items could not have shown a difference at this level whichever
    way its interval fell, and that is worth telling apart from a tie measured on wide evidence.

    `-` covers both rows that need nothing (the level is already reachable) and rows whose rate
    prices nothing (no item differed, or the artifact predates the recorded pair count).
    """
    discordant = stability.get("discordant")
    pairs = stability.get("pairs")
    if discordant is None or pairs is None:
        return "-"
    required = resolving_item_count(discordant, pairs, confidence)
    return "-" if required is None else str(required)


def boundary_table(
    rows: Sequence[tuple[str, ReadingStability]],
    *,
    title: str,
    key_header: str,
    subject: str,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[str]:
    """Where each row sits on the continuous scale its binary reading cut, as Markdown lines.

    Shared by every lane's report, so a knife-edge row is placed on one scale rather than on as
    many differently-worded tables as the repo has comparisons.
    """
    if not rows:
        return []
    lines = [
        f"### {title}",
        "",
        f"`p_positive` is the share of paired resamples in which {subject} is ahead; the reading "
        f"clears zero exactly when it exceeds {decision_probability(confidence):.3f}. A row is "
        "unsettled when either neighbouring convention would read it differently. `d` is the "
        "number of items the two lanes differ on: below "
        f"{minimum_discordant_pairs(confidence)} of them the level is unreachable whatever the "
        f"interval says, and the reading is `{reading_label(READING_INSUFFICIENT_EVIDENCE)}` "
        f"(the neighbouring levels need {minimum_discordant_pairs(LOOSER_CONFIDENCE)} and "
        f"{minimum_discordant_pairs(TIGHTER_CONFIDENCE)}). `n to reach` prices ANY row short of "
        "that bound, whatever its interval says: the items its own discordance rate would need "
        "before the level becomes reachable at all -- a floor on the item set, not an effect size "
        "it could then resolve. A `flat` row carrying one is a reading that could not have shown a "
        "difference, which is not the same as one that looked and found none.",
        "",
        f"| {key_header} | at {level_label(LOOSER_CONFIDENCE)} | reading ({level_label(confidence)}) "
        f"| at {level_label(TIGHTER_CONFIDENCE)} | p_positive | d | n to reach | settled? |",
        "| --- | :-: | :-: | :-: | ---: | ---: | ---: | :-: |",
    ]
    for key, entry in rows:
        settled = f"NO ({entry['side']})" if entry["borderline"] else "yes"
        discordant = entry.get("discordant")
        lines.append(
            f"| {key} | {reading_label(entry['looser_reading'])} "
            f"| {reading_label(entry['reading'])} | {reading_label(entry['tighter_reading'])} "
            f"| {entry['p_positive']:.3f} | {'-' if discordant is None else discordant} "
            f"| {_resolving_cell(entry, confidence)} | {settled} |"
        )
    lines.append("")
    return lines
