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

This module is the VOCABULARY and the assembly; the percentile arithmetic stays in `stats.py`,
which imports from here (never the other way round). A lane whose reading is richer than
separated/flat -- the embedder adoption bar reads three states -- computes its own reading at each
of the three levels and hands them to `stability_from_readings`, so every lane persists and renders
the identical shape.

Pure Python and dependency-free, so it imports and is unit-tested in the lightweight CI install.
"""

from collections.abc import Sequence

from typing_extensions import TypedDict

DEFAULT_CONFIDENCE = 0.95

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

# The two-state reading of ONE paired delta -- what most lanes cut from their interval. A lane with
# a richer reading supplies its own state names through `stability_from_readings`.
READING_SEPARATED = "separated"
READING_FLAT = "flat"


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
    *, reading: str, looser_reading: str, tighter_reading: str, p_positive: float
) -> ReadingStability:
    """Assemble the flag and the side from three already-computed readings of the same draw.

    The check is TWO-SIDED: a row that fails the bar but would clear a looser one is an undecided
    negative, and a row that clears the bar but a tighter one would drop is a positive resting on
    the convention. Reporting only the first would leave every near-miss positive looking settled.

    All three readings must come from the SAME resample draw, so the only thing that differs
    between them is the percentile cut -- the comparison isolates the confidence convention rather
    than mixing in a second draw's noise.
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
    }


def reading_label(reading: str) -> str:
    """`rank_only` -> `rank only` -- the one rendering of a reading every table shares."""
    return reading.replace("_", " ")


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


def _level(confidence: float) -> str:
    """`0.975` -> `97.5%` -- a percentage that keeps the half point the 97.5% level needs."""
    return f"{confidence * 100:g}%"


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
        "unsettled when either neighbouring convention would read it differently.",
        "",
        f"| {key_header} | at {_level(LOOSER_CONFIDENCE)} | reading ({_level(confidence)}) "
        f"| at {_level(TIGHTER_CONFIDENCE)} | p_positive | settled? |",
        "| --- | :-: | :-: | :-: | ---: | :-: |",
    ]
    for key, entry in rows:
        settled = f"NO ({entry['side']})" if entry["borderline"] else "yes"
        lines.append(
            f"| {key} | {reading_label(entry['looser_reading'])} "
            f"| {reading_label(entry['reading'])} | {reading_label(entry['tighter_reading'])} "
            f"| {entry['p_positive']:.3f} | {settled} |"
        )
    lines.append("")
    return lines
