"""The adoption bar's THREE-state reading, placed on the same cut scale every paired lane uses.

The shared annotation (`llb.rag.fusion_evidence.stability`) qualifies a two-state
`separated` / `flat` reading, which is what a single paired delta cuts. This lane's reading is
richer: `answer` / `rank only` / `neither`, decided by reading the objective delta first and the
first-hit-rank delta second. So it computes its own reading at each of the three conventional
levels and hands them to `stability_from_readings`, producing the identical `ReadingStability`
shape -- the persisted field, the boundary table, and the qualified verdict sentence are then the
same ones the bake-off, the fusion sweep, the ablation, and the answer lane report.

Both readers of the annotation come through here: the SWEEP measures it inline from the per-item
deltas and the resample draw it already holds (so a one-model run is as honest as a roster table),
and the roster re-derives the same quantity from the run bundles a finished sweep names. The two
agree exactly, because the deltas are in the same sorted item order and the draw is a pure function
of `(n, resamples, seed)`.

Deliberately additive. `cell_reading` still returns the same three states and the adopt bar still
reads the same interval, so no recorded verdict moves; what changes is that a knife-edge row is
now labelled as one instead of passing for settled evidence.

Pure Python and dependency-free, so it imports and is unit-tested in the lightweight CI install.
"""

from llb.eval.embedder_adoption.cross_model import (
    READING_ANSWER,
    READING_NEITHER,
    READING_RANK_ONLY,
)
from llb.eval.embedder_adoption.models import ItemDeltas
from llb.rag.fusion_evidence.evidence_gate import (
    READING_INSUFFICIENT_EVIDENCE,
    reaches_reporting_level,
)
from llb.rag.fusion_evidence.stability import (
    LOOSER_CONFIDENCE,
    ReadingStability,
    TIGHTER_CONFIDENCE,
    assert_neighbouring_levels,
    exceedance,
    stability_from_readings,
)
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_SEED,
    bootstrap_index_sets,
    bootstrap_interval,
    bootstrap_samples,
)
from llb.rag.fusion_evidence.paired import discordant_deltas

# This lane's rows carry the shared shape; the alias keeps the local name its call sites read by.
RowStability = ReadingStability


def reading_from_deltas(
    deltas: ItemDeltas,
    index_sets: list[list[int]],
    confidence: float = DEFAULT_CONFIDENCE,
) -> str:
    """The `answer` / `rank only` / `neither` reading of one delta vector pair.

    Identical rule to `cross_model.cell_reading` -- the objective first, then first-hit rank --
    but read straight off per-item deltas, so a SUBSET of the items or a different percentile cut
    can be judged exactly the way the whole set at the reporting level was.

    Each state is gated on ITS OWN metric's discordant count, since that is the evidence the claim
    would rest on. The order is preserved through the gate: a metric that clears zero on too few
    differing items ends the reading at `insufficient_evidence` rather than falling through to the
    next state, because "the answer did not move" is exactly the thing this row cannot establish.
    """
    if bootstrap_interval(deltas.objective, index_sets, confidence)["lo"] > 0.0:
        return _gate(deltas.objective, READING_ANSWER, confidence)
    if bootstrap_interval(deltas.reciprocal_rank, index_sets, confidence)["lo"] > 0.0:
        return _gate(deltas.reciprocal_rank, READING_RANK_ONLY, confidence)
    return READING_NEITHER


def _gate(values: list[float], reading: str, confidence: float) -> str:
    """`reading`, or `insufficient_evidence` when too few items differ to support the claim."""
    if reaches_reporting_level(discordant_deltas(values), confidence):
        return reading
    return READING_INSUFFICIENT_EVIDENCE


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
    rather than re-drawing beside them. All three readings come from that one draw, so the only
    thing differing between them is the percentile cut.
    """
    assert_neighbouring_levels(confidence, looser_confidence, tighter_confidence)
    return stability_from_readings(
        reading=reading_from_deltas(deltas, index_sets, confidence),
        looser_reading=reading_from_deltas(deltas, index_sets, looser_confidence),
        tighter_reading=reading_from_deltas(deltas, index_sets, tighter_confidence),
        p_positive=exceedance(bootstrap_samples(deltas.objective, index_sets)),
        # The objective's count, matching `p_positive`: it is the metric the leading state of this
        # lane's reading is decided on.
        discordant=discordant_deltas(deltas.objective),
        pairs=len(deltas),
    )


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
