"""Shared row labels and typed contracts for retrieval comparisons."""

from typing import TYPE_CHECKING, Protocol

from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.rag import ChunkRecord, RetrievalMetrics, SourceSpanRecord

if TYPE_CHECKING:
    from llb.rag.duplicate_models import DuplicateStats
    from llb.rag.embedding_bakeoff_uncertainty import PairedRow
    from llb.rag.noise_floor_models import NoiseFloorReport
    from llb.rag.retrieval_comparison_uncertainty import RetrievalComparisonVerdict

CompareItem = tuple[str, list[SourceSpanRecord]]

ROW_DENSE = "dense"
ROW_HYBRID = "hybrid"
ROW_HYBRID_LEMMAS = "hybrid+lemmas"
ROW_ORACLE_DOC = "dense+oracle-doc"
ROW_LEXICAL = "lexical"
RERANK_ROW_SUFFIX = "+rerank"


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
    noise_floor: NotRequired["NoiseFloorReport"]
