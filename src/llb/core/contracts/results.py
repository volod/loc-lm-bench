"""Evaluation score rows and leaderboard result contracts."""

from typing_extensions import NotRequired, TypedDict


class CaseScoreRow(TypedDict):
    item_id: str
    split: str
    status: str
    objective_score: float
    token_f1: float
    token_precision: float
    token_recall: float
    ranking_score: float
    exact: float
    contains: float
    retrieval_hit: float
    first_hit_rank: int | None
    tokens_per_s: float
    latency_s: float
    completion_tokens: int
    answer_preview: str
    # The prompt the backend actually consumed, in the model's own units. Present only when the
    # backend reports it, so a lane that compares context sizes can tell a measured prompt from a
    # missing measurement -- and a run whose context was silently truncated to the served window
    # from one that fit.
    prompt_tokens: NotRequired[int]
    semantic: NotRequired[float]
    judge_score: NotRequired[float]
    retrieve_latency_s: NotRequired[float]
    rerank_latency_s: NotRequired[float]
    query_processed: NotRequired[str]
    query_corrections: NotRequired[int]
    query_dense: NotRequired[str]
    query_hypothetical_answer: NotRequired[str]
    query_decomposition: NotRequired[str]
    query_subqueries: NotRequired[list[str]]
    # Prompt-side table-header restoration (table-header-context-restoration): how many retrieved
    # chunks were given back their column names in the prompt and what that added in characters.
    # Present on every case that RETRIEVED (0 / 0 when the step is off), absent on a lane that
    # supplied its own context, so an off lane and an on lane compare the same measured column.
    table_headers_restored: NotRequired[int]
    table_header_chars: NotRequired[float]
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
    token_precision: NotRequired[float]
    token_recall: NotRequired[float]
    found_rate: NotRequired[float]
    mean_completion_tokens: NotRequired[float]
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
    token_precision: NotRequired[float]
    token_recall: NotRequired[float]
    found_rate: NotRequired[float]
    mean_completion_tokens: NotRequired[float]
    judge: float | None
    semantic: float | None
    reliability: float
    tokens_per_s: float
    peak_vram_mb: float | None
    pareto: bool
    unresolved: bool
    feasible: bool
    n_cases: int
