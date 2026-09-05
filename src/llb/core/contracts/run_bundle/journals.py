"""Resume and abort records: what an interrupted run leaves behind so it can be finished.

None of these is published inside a run bundle -- the journal and its meta are dropped from the
staging directory before the atomic rename, and the budget abort is written beside a scorer's
resumable state. They are durable all the same: a resume READS them, and a resumed bundle must
score to the same rows an uninterrupted one would, so a record this project cannot validate is a
run that silently finishes differently.
"""

from typing import Final, Literal

from pydantic import Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject, UsageRecord
from llb.core.contracts.rag import ChunkRecord
from llb.core.contracts.run_bundle.common import RunBundleRow

CASE_PROGRESS_SCHEMA_ID: Final = "llb.run-progress"
CASE_PROGRESS_META_SCHEMA_ID: Final = "llb.run-progress-meta"
JUDGE_BUDGET_ABORT_SCHEMA_ID: Final = "llb.judge-budget-abort"


class JournaledCaseState(RunBundleRow):
    """The durable projection of one completed case's graph state.

    This is deliberately NARROWER than the in-memory `RagState`: the question, the gold spans, and
    the assembled context are inputs a resume rebuilds, while everything here is an OUTPUT that
    cannot be recomputed without calling the model again. Every field is optional because a lane
    that never ran a step never journals its columns -- but the SET is exhaustive, and
    `llb.executor.durability_journal` trims a state to exactly these keys, so a column added to a
    score row without being added here is a column a resumed case would silently lose.
    """

    retrieved: list[ChunkRecord] | None = None
    prompt_chunks: list[ChunkRecord] | None = None
    answer: str | None = None
    status: str | None = None
    error: str | None = None
    usage: UsageRecord | None = None
    retrieve_latency_s: float | None = None
    rerank_latency_s: float | None = None
    query_processed: str | None = None
    query_corrections: int | None = None
    query_dense: str | None = None
    query_hypothetical_answer: str | None = None
    query_decomposition: str | None = None
    query_subqueries: list[str] | None = None
    table_headers_restored: int | None = None
    table_header_chars: int | None = None
    envelope: JsonObject | None = None
    envelope_status: str | None = None
    envelope_error: str | None = None
    envelope_repaired: bool | None = None
    validation_checked_triples: int | None = None
    validation_violations: int | None = None
    validation_classes: list[str] | None = None
    validation_axioms: list[str] | None = None
    validation_repaired: bool | None = None
    validation_error: str | None = None


class CaseProgressRow(ArtifactContract):
    """One completed case in the append-only resume journal (`cases.progress.jsonl`)."""

    schema_id: Literal["llb.run-progress"] = CASE_PROGRESS_SCHEMA_ID
    schema_version: Literal["1.0.0"] = "1.0.0"
    item_id: str = Field(min_length=1)
    state: JournaledCaseState


class CaseProgressMeta(ArtifactContract):
    """`cases.progress.meta.json`: the identity a resume must still match to be the same run.

    The retry budget is deliberately not part of it -- a run may be resumed with a different one
    -- so a mismatch here is always a difference that would change the scored result.
    """

    schema_id: Literal["llb.run-progress-meta"] = CASE_PROGRESS_META_SCHEMA_ID
    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: str = Field(min_length=1)
    split: str
    config_digest: str = Field(min_length=1)
    goldset_digest: str = Field(min_length=1)
    n_items: int = Field(ge=0)


class JudgeBudgetAbort(ArtifactContract):
    """`scorer/abort.json`: the judge stopped on its declared spend or call budget.

    `resumable` is the load-bearing field: this is a bounded stop with the scorer's state intact,
    not a failed run, and an operator who raises the budget continues from here.
    """

    schema_id: Literal["llb.judge-budget-abort"] = JUDGE_BUDGET_ABORT_SCHEMA_ID
    schema_version: Literal["1.0.0"] = "1.0.0"
    status: Literal["aborted"] = "aborted"
    resumable: bool
    reason: str = Field(min_length=1)
    calls: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
