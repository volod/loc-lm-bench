"""Shared row labels and typed contracts for retrieval comparisons."""

from typing import TYPE_CHECKING, Protocol

from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.rag import ChunkRecord, RetrievalMetrics, SourceSpanRecord

if TYPE_CHECKING:
    from llb.rag.duplicates.models import DuplicateStats
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

# Question-type slices always present in the report, even at n=0, so a reader can tell "this
# corpus labels no numeric question" from "nobody looked". These are the slices a CHUNKING change
# is read on: in converted Ukrainian PDFs the numeric and comparative answers live in tables, and
# multi-hop answers need every span carried at once.
FOCUS_SLICES = ("numeric", "comparative", "multi-hop")


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> list[ChunkRecord]: ...


class ComparisonSlice(TypedDict):
    n: int
    backends: dict[str, RetrievalMetrics]


class ComparisonLane(RetrievalMetrics):
    paired_vs_baseline: NotRequired["PairedRow"]


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
    noise_floor: NotRequired["NoiseFloorReport"]
