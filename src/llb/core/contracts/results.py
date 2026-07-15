"""Evaluation score rows and leaderboard result contracts."""

from typing_extensions import NotRequired, TypedDict


class CaseScoreRow(TypedDict):
    item_id: str
    split: str
    status: str
    objective_score: float
    token_f1: float
    exact: float
    contains: float
    retrieval_hit: float
    first_hit_rank: int | None
    tokens_per_s: float
    latency_s: float
    completion_tokens: int
    answer_preview: str
    semantic: NotRequired[float]
    judge_score: NotRequired[float]
    retrieve_latency_s: NotRequired[float]
    rerank_latency_s: NotRequired[float]
    query_processed: NotRequired[str]
    query_corrections: NotRequired[int]
    groundedness: NotRequired[float]
    citation_validity: NotRequired[float]
    citation_coverage: NotRequired[float]
    hallucinated_citation_rate: NotRequired[float]
    n_citations: NotRequired[int]


class LeaderboardRow(TypedDict):
    rank: int | None
    model: str
    backend: str
    quality: float
    objective: float
    judge: float | None
    reliability: float
    tokens_per_s: float
    peak_vram_mb: float | None
    feasible: bool
    n_cases: int


class BoardRow(TypedDict):
    rank: int | None
    model: str
    backend: str
    tier: str
    quality: float
    quality_ci: NotRequired[tuple[float, float]]
    objective_ci: NotRequired[tuple[float, float]]
    semantic_ci: NotRequired[tuple[float, float]]
    judge_ci: NotRequired[tuple[float, float]]
    avg_rank: float
    objective: float
    judge: float | None
    semantic: float | None
    reliability: float
    tokens_per_s: float
    peak_vram_mb: float | None
    pareto: bool
    unresolved: bool
    feasible: bool
    n_cases: int
