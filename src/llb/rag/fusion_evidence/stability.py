"""Is a paired row's calibrated reading settled evidence, or is the threshold talking?

Every adopt-or-retain call cuts the one-sided paired sign-flip p-value at the alpha corresponding
to its two-sided confidence convention.  A row close to that cut otherwise prints exactly like one
that missed by a mile.

Two additive signals fix that, and neither changes any adoption rule:

- `randomization_p`, the calibrated quantity the reading thresholds, persisted beside the
  bootstrap diagnostic `p_positive`.
- `borderline`, a flag raised when the reading would CHANGE under a looser or a tighter but equally
  conventional confidence level. That is a statement about robustness to an arbitrary convention,
  not a fitted threshold: no constant is tuned to the data, and all three levels compared are ones
  the repo already reports at.

A reading also has to be entitled to what it says. The evidence gate relabels a claim resting on
too few differing items `insufficient_evidence`; its bound moves with the confidence level, so that
change itself can make a row borderline.
A lane whose reading is richer than
separated/flat -- the embedder adoption bar reads three states -- computes its own reading at each
of the three levels and hands them to `stability_from_readings`, so every lane persists and renders
the identical shape.

"""

from collections.abc import Sequence

from typing_extensions import NotRequired, TypedDict

from llb.rag.fusion_evidence.evidence_gate import (
    DEFAULT_CONFIDENCE,
    reading_label,
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
    # One-sided paired sign-flip p-value.  This is the calibrated quantity the reading is cut on;
    # p_positive remains beside it as the bootstrap diagnostic and for archived-artifact context.
    randomization_p: NotRequired[float]
    randomization_method: NotRequired[str]
    randomization_samples: NotRequired[int]
    # What a looser / tighter confidence convention reads; both equal `reading` when settled.
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
    randomization_p: float | None = None,
    randomization_method: str | None = None,
    randomization_samples: int | None = None,
    discordant: int | None = None,
    pairs: int | None = None,
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

    When supplied, `pairs` is recorded beside `discordant` for the same reason the paired gate
    needs both: together they are the discordance RATE that prices a withdrawn reading's item set.
    Non-paired bootstrap ratios omit both fields and therefore never imply an exact-sign gate.
    """
    stability: ReadingStability = {
        "reading": reading,
        "p_positive": p_positive,
        "looser_reading": looser_reading,
        "tighter_reading": tighter_reading,
        "borderline": looser_reading != reading or tighter_reading != reading,
        "side": SIDE_BELOW
        if looser_reading != reading
        else (SIDE_ABOVE if tighter_reading != reading else None),
    }
    if discordant is not None and pairs is not None:
        stability["discordant"] = discordant
        stability["pairs"] = pairs
    if randomization_p is not None:
        stability["randomization_p"] = randomization_p
    if randomization_method is not None:
        stability["randomization_method"] = randomization_method
    if randomization_samples is not None:
        stability["randomization_samples"] = randomization_samples
    return stability


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
    """One row's distance from the decision cut and the neighbouring reading that changes it."""
    flip = (
        f"a {TIGHTER_CONFIDENCE:.3f} convention would read it `{stability['tighter_reading']}`"
        if stability["side"] == SIDE_ABOVE
        else f"a {LOOSER_CONFIDENCE:.2f} convention would read it `{stability['looser_reading']}`"
    )
    calibrated = stability.get("randomization_p")
    prefix = (
        f"randomization p {calibrated:.4f}, p_positive {stability['p_positive']:.3f}"
        if calibrated is not None
        else f"p_positive {stability['p_positive']:.3f}"
    )
    return f"{prefix}, {flip}"


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


def boundary_table(
    rows: Sequence[tuple[str, ReadingStability]],
    *,
    title: str,
    key_header: str,
    subject: str,
    confidence: float = DEFAULT_CONFIDENCE,
    evidence_counts: bool = True,
    positive_event: str | None = None,
) -> list[str]:
    """Render through the presentation module while preserving the established import seam."""
    from llb.rag.fusion_evidence.stability_report import boundary_table as render

    return render(
        rows,
        title=title,
        key_header=key_header,
        subject=subject,
        confidence=confidence,
        evidence_counts=evidence_counts,
        positive_event=positive_event,
    )
