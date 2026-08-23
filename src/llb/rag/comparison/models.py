"""Shared row labels and typed contracts for retrieval comparisons."""

from typing import TYPE_CHECKING, Protocol

from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.rag import ChunkRecord, RetrievalMetrics, SourceSpanRecord

if TYPE_CHECKING:
    from llb.rag.duplicates.models import DuplicateStats
    from llb.rag.stitching import StitchCensus
    from llb.rag.embedding_bakeoff.uncertainty import PairedRow
    from llb.rag.noise_floor.models import NoiseFloorReport
    from llb.rag.comparison.uncertainty import RetrievalComparisonVerdict

CompareItem = tuple[str, list[SourceSpanRecord]]

ROW_DENSE = "dense"
ROW_HYBRID = "hybrid"
ROW_HYBRID_LEMMAS = "hybrid+lemmas"
ROW_ORACLE_DOC = "dense+oracle-doc"
ROW_LEXICAL = "lexical"
RERANK_ROW_SUFFIX = "+rerank"
# The assembly-time twin (`llb.rag.stitching`): the SAME lane, its top-k reflowed into contiguous
# blocks. It is a REPORTED lever, never an adoption candidate -- it cannot move the finding metrics
# a verdict is decided on -- so `compare-retrieval` keeps it out of the eligible lanes.
STITCH_ROW_SUFFIX = "+stitch"
# A `size` lane label. `size` is a build parameter, not a strategy, so the lane names the strategy
# it varies and carries the size as a `#` qualifier, the way an answer-quality lane carries its
# retrieval budget in `vector#k50`.
SIZE_ROW_QUALIFIER = "#size"


def size_row_label(strategy: str, size: int) -> str:
    """The lane label of one chunk `size` under `strategy` (e.g. `recursive#size400`)."""
    return f"{strategy}{SIZE_ROW_QUALIFIER}{size}"


# Question-type slices always present in the report, even at n=0, so a reader can tell "this
# corpus labels no numeric question" from "nobody looked". These are the slices a CHUNKING change
# is read on: in converted Ukrainian PDFs the numeric and comparative answers live in tables,
# multi-hop answers need every span carried at once, and a procedural answer is a multi-line step
# sequence -- the shape a `size` cap cuts, so it is where evidence INTACTNESS is read.
FOCUS_SLICES = ("numeric", "comparative", "multi-hop", "procedural")


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> list[ChunkRecord]: ...


class ComparisonSlice(TypedDict):
    """One question-type slice: its item count and every lane scored on those items alone.

    A lane row is a `ComparisonLane`, so it carries the slice's OWN paired reading against the
    baseline -- a slice this small is decided by one or two questions, and a point delta cannot
    say which.
    """

    n: int
    backends: dict[str, "ComparisonLane"]


class ComparisonLane(RetrievalMetrics):
    paired_vs_baseline: NotRequired["PairedRow"]


class StitchLaneReport(TypedDict):
    """One stitched twin's accounting, plus the invariance its reading depends on.

    Stitching reflows retrieved chunks and retrieves nothing new, so `recall@k` and
    `span_char_coverage_at_k` MUST reproduce the base lane exactly. The two flags record that
    check on the run's own numbers rather than leaving it to a reader.
    """

    base: str
    census: "StitchCensus"
    recall_invariant: bool
    coverage_invariant: bool


class ComparisonItemOutcome(TypedDict):
    item_id: str
    lanes: dict[str, dict[str, float]]


class ComparisonUncertainty(TypedDict):
    baseline: str | None
    eligible_lanes: list[str]
    resamples: int
    confidence: float
    seed: int


class ComparisonReport(TypedDict):
    k: int
    n: int
    backends: dict[str, ComparisonLane]
    best_recall: str | None
    paired_items: list[ComparisonItemOutcome]
    uncertainty: ComparisonUncertainty
    verdict: "RetrievalComparisonVerdict"
    slices: NotRequired[dict[str, ComparisonSlice]]
    duplicates: NotRequired[dict[str, "DuplicateStats"]]
    # Why a censused store indexed every copy, for the lanes that did not collapse. A store built
    # with `--keep-duplicate-chunks`, or under a strategy whose vector is not a pure function of
    # its text (`late`), indexes every copy, and the census line must say which of the two it is.
    # A lane absent from this map collapsed -- as does every lane of an artifact recorded before
    # the key existed.
    duplicates_kept: NotRequired[dict[str, str]]
    stitching: NotRequired[dict[str, StitchLaneReport]]
    noise_floor: NotRequired["NoiseFloorReport"]
