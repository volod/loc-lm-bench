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

Pure Python and dependency-free, so it imports and is unit-tested in the lightweight CI install.
"""

from collections.abc import Sequence

from typing_extensions import TypedDict

from llb.eval.embedder_adoption.screen import ItemDeltas, reading_from_deltas
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_SEED, bootstrap_index_sets

# The looser conventional confidence the reading is re-checked at. 90% and 95% are both standard
# reporting levels, so a reading that differs between them is decided by the convention rather than
# by the evidence. Not fitted to any recorded row.
BORDERLINE_CONFIDENCE = 0.90

# Marks a reading that flips between the reporting and the looser confidence level. It is a
# QUALIFIER on the three readings, never a fourth outcome the bar can adopt on.
BORDERLINE_MARK = "borderline"


class RowStability(TypedDict):
    """One cell's reading plus how close it sits to the cut that produced it."""

    reading: str
    # Share of paired resamples in which the objective delta is above zero. The reading's own
    # threshold in this scale is `1 - (1 - confidence) / 2` (0.975 at the default 95%).
    p_positive: float
    # The reading a `borderline_confidence` interval would give; equal to `reading` when settled.
    looser_reading: str
    borderline: bool


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


def row_stability(
    deltas: ItemDeltas,
    *,
    resamples: int,
    confidence: float = DEFAULT_CONFIDENCE,
    borderline_confidence: float = BORDERLINE_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> RowStability:
    """The reading, its exceedance probability, and whether a looser interval would change it.

    Both readings are drawn from the SAME resample index sets, so the only thing that differs
    between them is the percentile cut -- the comparison isolates the confidence convention rather
    than mixing in a second draw's noise.
    """
    if borderline_confidence >= confidence:
        raise ValueError(
            f"the borderline confidence ({borderline_confidence}) must be LOOSER than the "
            f"reporting confidence ({confidence}); a tighter level would flag every row"
        )
    index_sets = bootstrap_index_sets(len(deltas), resamples, seed)
    reading = reading_from_deltas(deltas, index_sets, confidence)
    looser = reading_from_deltas(deltas, index_sets, borderline_confidence)
    return {
        "reading": reading,
        "p_positive": exceedance(deltas.objective, index_sets),
        "looser_reading": looser,
        "borderline": looser != reading,
    }


def format_reading(stability: RowStability | None, reading: str) -> str:
    """`neither` / `neither (borderline)` -- the reading as a table cell.

    Falls back to the bare reading when no stability was measured, which is what happens when the
    run bundles a sweep names are no longer on disk.
    """
    label = reading.replace("_", " ")
    if stability is None or not stability["borderline"]:
        return label
    return f"{label} ({BORDERLINE_MARK})"
