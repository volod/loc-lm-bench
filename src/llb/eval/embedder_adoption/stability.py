"""Is a cell's reading settled evidence, or is it the threshold talking?

Every adoption-bar reading is a BINARY cut (`answer` / `rank only` / `neither`) taken from a
continuous interval by one test: does the delta's lower bound clear zero. A row whose bound sits ON
zero therefore prints exactly like a row that missed by a mile, and the measured case is
`lapa-v0.1.2`, whose `k10+rerank` objective delta is +0.024 `[-0.000, +0.059]` -- printed `neither`
beside `mistral`, whose `neither` is nowhere near the line.

Two additive signals fix that, and neither changes the bar:

- `p_positive`, the share of paired resamples in which the candidate is ahead. It is the CONTINUOUS
  quantity the reading thresholds: a 95% percentile interval clears zero exactly when
  `p_positive > 0.975`, so one number says both what the reading is and how far from the cut it sits.
- `borderline`, a flag raised when the reading would CHANGE under a looser but equally conventional
  confidence level (90% by default). That is a statement about robustness to an arbitrary
  convention, not a fitted threshold: no constant is tuned to the data, and the two levels compared
  are both ones the repo already reports at.

Deliberately additive. `cell_reading` still returns the same three states and the adopt bar still
reads the same interval, so no recorded verdict moves; what changes is that a knife-edge row is
now labelled as one instead of passing for settled evidence.

Both readers of the annotation come through here: the SWEEP measures it inline from the per-item
deltas and the resample draw it already holds (so a one-model run is as honest as a roster table),
and the roster re-derives the same quantity from the run bundles a finished sweep names. The two
agree exactly, because the deltas are in the same sorted item order and the draw is a pure function
of `(n, resamples, seed)`.

Pure Python and dependency-free, so it imports and is unit-tested in the lightweight CI install.
"""

from collections.abc import Sequence

from typing_extensions import TypedDict

from llb.eval.embedder_adoption.cross_model import (
    READING_ANSWER,
    READING_NEITHER,
    READING_RANK_ONLY,
)
from llb.eval.embedder_adoption.models import ItemDeltas
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_SEED,
    bootstrap_index_sets,
    bootstrap_interval,
)

# The two neighbouring conventional confidences the reading is re-checked at. All three levels are
# standard reporting conventions, so a reading that differs across them was decided by the
# convention rather than by the evidence. Neither is fitted to a recorded row.
LOOSER_CONFIDENCE = 0.90
TIGHTER_CONFIDENCE = 0.975

# Marks a reading that flips at either neighbouring level. It is a QUALIFIER on the three readings,
# never a fourth outcome the bar can adopt on.
BORDERLINE_MARK = "borderline"

# Which side of the cut a borderline row sits on. `below` fails the bar but would clear a looser
# one; `above` clears it but a tighter one would drop it. The distinction matters to an operator:
# `below` is an undecided negative, `above` is a positive resting on the convention.
SIDE_BELOW = "below"
SIDE_ABOVE = "above"


class RowStability(TypedDict):
    """One cell's reading plus how close it sits to the cut that produced it."""

    reading: str
    # Share of paired resamples in which the objective delta is above zero. The reading's own
    # threshold in this scale is `1 - (1 - confidence) / 2` (0.975 at the default 95%).
    p_positive: float
    # What a looser / tighter interval would read; both equal `reading` when the row is settled.
    looser_reading: str
    tighter_reading: str
    borderline: bool
    # SIDE_BELOW / SIDE_ABOVE, or None when settled.
    side: str | None


def reading_from_deltas(
    deltas: ItemDeltas,
    index_sets: list[list[int]],
    confidence: float = DEFAULT_CONFIDENCE,
) -> str:
    """The `answer` / `rank only` / `neither` reading of one delta vector pair.

    Identical rule to `cross_model.cell_reading` -- the objective first, then first-hit rank --
    but read straight off per-item deltas, so a SUBSET of the items or a different percentile cut
    can be judged exactly the way the whole set at the reporting level was.
    """
    if bootstrap_interval(deltas.objective, index_sets, confidence)["lo"] > 0.0:
        return READING_ANSWER
    if bootstrap_interval(deltas.reciprocal_rank, index_sets, confidence)["lo"] > 0.0:
        return READING_RANK_ONLY
    return READING_NEITHER


def exceedance(values: Sequence[float], index_sets: Sequence[Sequence[int]]) -> float:
    """Share of bootstrap resamples whose mean is above zero (a one-sided bootstrap p-value).

    This is the continuous quantity the `lo > 0` reading thresholds, so it places a row on the
    decision scale instead of only reporting which side of it the row landed.
    """
    if not values or not index_sets:
        return 0.0
    above = 0
    for indexes in index_sets:
        if sum(values[i] for i in indexes) > 0.0:  # mean > 0 iff sum > 0 for a fixed n
            above += 1
    return above / len(index_sets)


def decision_probability(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """The `p_positive` a reading must exceed to clear zero at `confidence` (0.975 at 95%)."""
    return 1.0 - (1.0 - confidence) / 2.0


def stability_from_index_sets(
    deltas: ItemDeltas,
    index_sets: list[list[int]],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    looser_confidence: float = LOOSER_CONFIDENCE,
    tighter_confidence: float = TIGHTER_CONFIDENCE,
) -> RowStability:
    """The reading, its exceedance probability, and whether a neighbouring level would change it.

    Takes an ALREADY DRAWN set of resample indexes so a caller that has one -- the sweep, which
    shares a single draw across every cell and metric -- annotates the very intervals it published
    rather than re-drawing beside them.

    The check is TWO-SIDED: a row that fails the bar but would clear a looser one is an undecided
    negative, and a row that clears the bar but a tighter one would drop is a positive resting on
    the convention. Reporting only the first would leave every near-miss `answer` looking settled.

    All three readings are drawn from the SAME resample index sets, so the only thing that differs
    between them is the percentile cut -- the comparison isolates the confidence convention rather
    than mixing in a second draw's noise.
    """
    if not looser_confidence < confidence < tighter_confidence:
        raise ValueError(
            f"the neighbouring levels must straddle the reporting confidence: "
            f"{looser_confidence} < {confidence} < {tighter_confidence}"
        )
    reading = reading_from_deltas(deltas, index_sets, confidence)
    looser = reading_from_deltas(deltas, index_sets, looser_confidence)
    tighter = reading_from_deltas(deltas, index_sets, tighter_confidence)
    return {
        "reading": reading,
        "p_positive": exceedance(deltas.objective, index_sets),
        "looser_reading": looser,
        "tighter_reading": tighter,
        "borderline": looser != reading or tighter != reading,
        "side": SIDE_BELOW if looser != reading else (SIDE_ABOVE if tighter != reading else None),
    }


def row_stability(
    deltas: ItemDeltas,
    *,
    resamples: int,
    confidence: float = DEFAULT_CONFIDENCE,
    looser_confidence: float = LOOSER_CONFIDENCE,
    tighter_confidence: float = TIGHTER_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> RowStability:
    """`stability_from_index_sets` for a caller holding no draw of its own -- it draws one.

    The draw is a pure function of `(n, resamples, seed)`, so a roster re-measuring a finished
    sweep from its bundles reproduces the stability that sweep persisted, bit for bit.
    """
    return stability_from_index_sets(
        deltas,
        bootstrap_index_sets(len(deltas), resamples, seed),
        confidence=confidence,
        looser_confidence=looser_confidence,
        tighter_confidence=tighter_confidence,
    )


def reading_label(reading: str) -> str:
    """`rank_only` -> `rank only` -- the one rendering of a reading every table shares."""
    return reading.replace("_", " ")


def format_reading(stability: RowStability | None, reading: str) -> str:
    """`neither` / `neither (borderline)` -- the reading as a table cell.

    Falls back to the bare reading when no stability was measured, which is what happens when the
    run bundles a sweep names are no longer on disk.
    """
    label = reading_label(reading)
    if stability is None or not stability["borderline"]:
        return label
    return f"{label} ({BORDERLINE_MARK})"


def _level(confidence: float) -> str:
    """`0.975` -> `97.5%` -- a percentage that keeps the half point the 97.5% level needs."""
    return f"{confidence * 100:g}%"


def boundary_table(
    rows: Sequence[tuple[str, RowStability]],
    *,
    title: str,
    key_header: str,
    subject: str,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[str]:
    """Where each row sits on the continuous scale its binary reading cut, as Markdown lines.

    Shared by the SWEEP report (one row per cell) and the ROSTER report (one row per model in the
    focus cell), so a single-model run and a five-model table place a knife-edge row on exactly the
    same scale rather than on two tables an operator has to reconcile.
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
