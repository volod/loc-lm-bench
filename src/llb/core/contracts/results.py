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
    # Declared answer contract (typed-rag-answer-envelope). Present only on an envelope-format
    # run, so a free-text bundle keeps exactly the shape it had. `envelope_status` is the parse
    # verdict (`ok` / `malformed` / `schema_invalid`), `repaired` says the one bounded repair
    # reprompt was spent on this case (first-attempt conformance is `1 - repair_rate`), and
    # `n_claims` / `envelope_abstained` are read off the declaration itself.
    envelope_status: NotRequired[str]
    repaired: NotRequired[bool]
    n_claims: NotRequired[int]
    envelope_abstained: NotRequired[bool]
    # Step two of the answer gate (ontology-validated-answer-gate). Present only when the ontology
    # gate ran, so an ungated envelope bundle keeps the shape it had. `validation_checked_triples`
    # is the population the verdict rests on (an envelope declaring no triple was UNCHECKED, not
    # cleared), `validation_classes` / `validation_axioms` name what broke -- which is what the
    # per-axiom-class adopt-or-reject verdict keys on -- and `validation_repaired` says the one
    # bounded semantic reprompt was spent on this case.
    validation_checked_triples: NotRequired[int]
    validation_violations: NotRequired[int]
    validation_classes: NotRequired[list[str]]
    validation_axioms: NotRequired[list[str]]
    validation_repaired: NotRequired[bool]
    # Answer-side gold-span coverage (`llb.scoring.answer_spans`): whether the ANSWER states each
    # labeled span's fact, and the all-spans gate over them. Every current run writes all three;
    # they are optional only so a bundle recorded before the metric existed still re-reads.
    answer_span_coverage: NotRequired[float]
    answer_all_spans: NotRequired[float]
    answer_spans_measured: NotRequired[int]
    # Response-integrity guard (`llb.scoring.answer_guard`): did the completion leak deliberation
    # into the answer body despite the backend's native thinking-suppression flag, and did the
    # model answer in the question's language? Both are ADDITIVE -- they never change `status` or
    # the objective, they name a delivery failure the correctness columns cannot express. Every
    # current run writes all five; they are optional only so a bundle recorded before the guard
    # existed still re-reads. `reasoning_leak_marker` names the signal that fired (a reasoning
    # delimiter, or the deliberation opener of a leak whose terminator the token budget cut off),
    # and `reasoning_leak_chars` is how much of the generation the leak accounts for -- the term
    # that inflates `completion_tokens`, and with it throughput and cost.
    reasoning_leak: NotRequired[bool]
    reasoning_leak_marker: NotRequired[str]
    reasoning_leak_chars: NotRequired[int]
    answer_language: NotRequired[str]
    language_mismatch: NotRequired[bool]
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
