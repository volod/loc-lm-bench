"""The record contracts a published run bundle is made of, beside its manifest.

`manifest.json` says what a run WAS (`llb.core.contracts.runs`); these are the rows and sidecars
that say what it measured. Until now every one of them was a `dict[str, object]` handed to
`json.dumps`, so a board could not tell a bundle this build wrote from one a newer build wrote,
and an outside consumer had to guess a column roster from a filename.

Two shapes appear here, and the difference is deliberate:

* A row whose columns are the SAME for every producer is modelled column by column -- the
  evaluation case score, the retrieved-span record, the resume journal, the abort record.
* A row or document whose body is the LANE'S OWN is an envelope: identity, what kind of reading it
  is, and the body under one named field. A benchmark cell's columns are named by whatever that
  lane swept, and a study's design and analysis grow a section per added measurement, so freezing
  either body would make every new measurement a contract change. This is the same split the
  retrieval comparison sidecar already draws between its envelope and its report.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject

CASE_SCORE_SCHEMA_ID = "llb.case-score"
BENCHMARK_CELL_SCHEMA_ID = "llb.benchmark-cell"
CASE_RETRIEVAL_SCHEMA_ID = "llb.case-retrieval"
CASE_PROGRESS_SCHEMA_ID = "llb.case-progress"
RUN_PROGRESS_META_SCHEMA_ID = "llb.run-progress-meta"
RUN_ABORT_SCHEMA_ID = "llb.run-abort"
STUDY_DESIGN_SCHEMA_ID = "llb.study-design"
STUDY_ANALYSIS_SCHEMA_ID = "llb.study-analysis"
CONTEXT_PROBE_SCHEMA_ID = "llb.context-probe"

# The body field each envelope family carries, which is also what a pre-contract file of that
# family became: an old benchmark row was the bare cell, an old design sidecar the bare design.
CELL_FIELD = "cell"
DESIGN_FIELD = "design"
ANALYSIS_FIELD = "analysis"


class CaseScoreRecord(ArtifactContract):
    """One row of `scores.jsonl` from an evaluation run: how one case scored.

    The required columns are the ones `score_case` writes on every case, whatever lane ran. Every
    optional column belongs to a lane that may not have run -- query preparation, the declared
    answer envelope, the ontology gate, the answer-side citation metrics -- so an absent column
    means the lane did not run, never a defaulted value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.case-score"]
    schema_version: Literal["1.0.0"]
    item_id: str = Field(min_length=1)
    split: str
    status: str
    objective_score: float
    token_f1: float
    exact: float
    contains: float
    retrieval_hit: float
    first_hit_rank: int | None = None
    tokens_per_s: float
    latency_s: float
    completion_tokens: int
    answer_preview: str
    # Recorded by every current run, and absent from the bundles written before the pair-based
    # ranking score existed -- which are still on disk, so their absence is optional rather than a
    # second version. A reader that needs one reads the absence, never a default.
    token_precision: float | None = None
    token_recall: float | None = None
    ranking_score: float | None = None
    prompt_tokens: int | None = None
    semantic: float | None = None
    judge_score: float | None = None
    retrieve_latency_s: float | None = None
    rerank_latency_s: float | None = None
    query_processed: str | None = None
    query_corrections: int | None = None
    query_dense: str | None = None
    query_hypothetical_answer: str | None = None
    query_decomposition: str | None = None
    query_subqueries: list[str] | None = None
    table_headers_restored: int | None = None
    table_header_chars: float | None = None
    envelope_status: str | None = None
    repaired: bool | None = None
    n_claims: int | None = None
    envelope_abstained: bool | None = None
    validation_checked_triples: int | None = None
    validation_violations: int | None = None
    validation_classes: list[str] | None = None
    validation_axioms: list[str] | None = None
    validation_repaired: bool | None = None
    answer_span_coverage: float | None = None
    answer_all_spans: float | None = None
    answer_spans_measured: int | None = None
    reasoning_leak: bool | None = None
    reasoning_leak_marker: str | None = None
    reasoning_leak_chars: int | None = None
    answer_language: str | None = None
    language_mismatch: bool | None = None
    groundedness: float | None = None
    citation_validity: float | None = None
    citation_coverage: float | None = None
    hallucinated_citation_rate: float | None = None
    n_citations: int | None = None


class BenchmarkCellRecord(ArtifactContract):
    """One row of `scores.jsonl` from a benchmark category run: one measured cell.

    The envelope owns the identity; the cell is the lane's own. A memory-fold lane's cell is a
    ladder level and its band, a tool lane's is an episode and its calls, a seed row is a seed and
    its verdict -- naming those columns here would make the contract a union of every benchmark
    this project will ever add, and every added column a version.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.benchmark-cell"]
    schema_version: Literal["1.0.0"]
    cell: JsonObject = Field(default_factory=dict)


class SourceSpanRecord(BaseModel):
    """One gold span of the item a case was scored against."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    doc_id: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str


class RetrievedOccurrenceRecord(BaseModel):
    """One other place a collapsed chunk's text appears (`llb.rag.duplicates.collapse`)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    doc_id: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    chunk_id: str | None = None


class RetrievedSpanRecord(BaseModel):
    """One retrieved span as miss analysis reads it back: where it came from and at what rank.

    `duplicate_count` and `duplicate_occurrences` are present only for a chunk that collapsed
    byte-identical copies, so their absence says the chunk collapsed nothing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    doc_id: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    rank: int = Field(ge=0)
    retrieval_score: float | None = None
    text_preview: str | None = None
    duplicate_count: int | None = None
    duplicate_occurrences: list[RetrievedOccurrenceRecord] | None = None


class CaseRetrievalRecord(ArtifactContract):
    """One row of `retrieval.jsonl`: what a case's context held against what it needed."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.case-retrieval"]
    schema_version: Literal["1.0.0"]
    item_id: str = Field(min_length=1)
    retrieved: list[RetrievedSpanRecord] = Field(default_factory=list)
    gold_spans: list[SourceSpanRecord] = Field(default_factory=list)


class CaseProgressRecord(ArtifactContract):
    """One row of the resume journal: a completed case and the state it was scored from.

    The journal lives in the staging directory and is dropped before publication, so it is never a
    member of a bundle -- but it is durable across a crash, which is the whole point of it, and a
    resume that read a journal a newer build wrote would re-score cases against state it cannot
    see. The state map stays open: its keys are the journalled graph-state keys, owned by the lanes
    that produce them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.case-progress"]
    schema_version: Literal["1.0.0"]
    item_id: str = Field(min_length=1)
    state: JsonObject = Field(default_factory=dict)


class RunProgressMeta(ArtifactContract):
    """The determinism-critical identity a `--resume` is checked against."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.run-progress-meta"]
    schema_version: Literal["1.0.0"]
    run_id: str = Field(min_length=1)
    split: str
    config_digest: str = Field(min_length=1)
    goldset_digest: str = Field(min_length=1)
    n_items: int = Field(ge=0)


class RunAbortRecord(ArtifactContract):
    """`scorer/abort.json`: the run stopped on a declared budget, and can be resumed."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.run-abort"]
    schema_version: Literal["1.0.0"]
    status: str = Field(min_length=1)
    resumable: bool
    reason: str
    calls: int = Field(ge=0)
    cost_usd: float


class StudyDesignRecord(ArtifactContract):
    """A study's `<stem>-design.json` sidecar: what the lane declared before it measured."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.study-design"]
    schema_version: Literal["1.0.0"]
    design: JsonObject = Field(default_factory=dict)


class StudyAnalysisRecord(ArtifactContract):
    """A study's `<stem>-analysis.json` sidecar: the reading taken against that design."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.study-analysis"]
    schema_version: Literal["1.0.0"]
    analysis: JsonObject = Field(default_factory=dict)


class ContextProbeRecord(ArtifactContract):
    """One row of `probes.jsonl`: how one case answered with its gold context withheld."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.context-probe"]
    schema_version: Literal["1.0.0"]
    item_id: str = Field(min_length=1)
    probe: bool = True
    n_context: int = Field(ge=0)
    status: str
    abstained: bool
    answer_preview: str = ""
