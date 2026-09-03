"""Contracts for the records an orchestrated run and a board analysis leave behind.

An auto-RAG run is resumable, so its manifest, its journal, and every stage result are read back
by a LATER process -- which is exactly the case a version identity exists for: a resume that read
a journal a newer build wrote would replay stages against results it cannot fully see. The board
side is the other direction: the miss analysis, the operator recommendation, and the composed
agent profile are what leaves this project for a person or another runtime to act on.

Where a body is the producer's own it stays open under one named field -- a stage result is
whatever that stage measured, and a profile field's value is whichever lane produced it. What the
contract fixes is the frame a reader needs before it opens anything.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject

AUTO_RAG_MANIFEST_SCHEMA_ID = "llb.auto-rag-manifest"
AUTO_RAG_JOURNAL_EVENT_SCHEMA_ID = "llb.auto-rag-journal-event"
AUTO_RAG_STAGE_RESULT_SCHEMA_ID = "llb.auto-rag-stage-result"
AUTO_RAG_STAGE_LINKS_SCHEMA_ID = "llb.auto-rag-stage-links"
AUTO_RAG_RECOMMENDATION_SCHEMA_ID = "llb.auto-rag-recommendation"
MISS_ANALYSIS_SCHEMA_ID = "llb.miss-analysis"
MISS_RECORD_SCHEMA_ID = "llb.miss-record"
AGENT_PROFILE_SCHEMA_ID = "llb.agent-profile"

AUTO_RAG_KIND = "auto-rag"
# What a pre-contract `artifacts.json` was: the bare stage-to-result map, with nowhere to put an
# identity of its own.
STAGES_FIELD = "stages"


class AutoRagManifest(ArtifactContract):
    """`manifest.json` of an auto-RAG run: the settings a resume is refused against.

    `fingerprint` is the digest of `settings`; a resume whose settings hash differently is a
    different run and is refused rather than continued.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.auto-rag-manifest"]
    schema_version: Literal["1.0.0"]
    kind: str = AUTO_RAG_KIND
    fingerprint: str = Field(min_length=1)
    settings: JsonObject = Field(default_factory=dict)


class AutoRagJournalEvent(ArtifactContract):
    """One appended line of `journal.jsonl`: a stage changed state at a time.

    `fields` carries whatever the event reported -- a pause reason, an error type, a result digest
    -- which is the event's own, not a fixed column roster.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.auto-rag-journal-event"]
    schema_version: Literal["1.0.0"]
    time: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    status: str = Field(min_length=1)
    fields: JsonObject = Field(default_factory=dict)


class AutoRagStageResult(ArtifactContract):
    """`stages/<stage>/result.json`: the durable marker a resume skips a completed stage on."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.auto-rag-stage-result"]
    schema_version: Literal["1.0.0"]
    status: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    result: JsonObject = Field(default_factory=dict)


class AutoRagStageLinks(ArtifactContract):
    """`artifacts.json`: every stage's result, collected beside the recommendation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.auto-rag-stage-links"]
    schema_version: Literal["1.0.0"]
    stages: JsonObject = Field(default_factory=dict)


class AutoRagRecommendation(ArtifactContract):
    """`rag_recommendation.yaml`: the configuration an auto-RAG run recommends, and its evidence.

    This is the one artifact of the pipeline an operator copies into a serving config by hand, so
    its top level is named field by field. The five blocks under it stay open: each is the knob set
    of a component that evolves on its own schedule, and `evidence` is a map of stage results whose
    shape each stage owns.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.auto-rag-recommendation"]
    schema_version: Literal["1.0.0"]
    model: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    serving: JsonObject = Field(default_factory=dict)
    chunking: JsonObject = Field(default_factory=dict)
    retrieval: JsonObject = Field(default_factory=dict)
    prompt_system: JsonObject = Field(default_factory=dict)
    evidence: JsonObject = Field(default_factory=dict)


class MissRecord(ArtifactContract):
    """One row of `misses.jsonl`: a case that missed, and what the analysis classified it as."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.miss-record"]
    schema_version: Literal["1.0.0"]
    item_id: str = Field(min_length=1)
    miss_class: str
    status: str
    objective_score: float
    judge_score: float | None = None
    retrieval_hit: bool
    first_hit_rank: int | None = None
    question: str
    source_doc_id: str
    topic: str
    question_type: str
    answer_preview: str
    retrieved_docs: list[str] = Field(default_factory=list)


class MissAnalysisReport(ArtifactContract):
    """`analysis.json`: the machine-readable miss analysis `llb recommend` reads.

    The clusters, probes, and recommendations are lists of the analysis's own row shapes, which is
    where new dimensions and new recommendation kinds land; the frame -- which run was analysed,
    at what threshold, with how many misses -- is what a reader needs first.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.miss-analysis"]
    schema_version: Literal["1.0.0"]
    created_at: str = Field(min_length=1)
    run_dir: str
    model: str
    backend: str
    split: str
    n_cases: int = Field(ge=0)
    n_misses: int = Field(ge=0)
    threshold: float
    rag_config: JsonObject = Field(default_factory=dict)
    class_counts: dict[str, int] = Field(default_factory=dict)
    clusters: dict[str, list[JsonObject]] = Field(default_factory=dict)
    probes: list[JsonObject] = Field(default_factory=list)
    recommendations: list[JsonObject] = Field(default_factory=list)


class ProfileAnchor(BaseModel):
    """What every field of a composed profile was required to agree with.

    `retrieval_fingerprint` is the KNOB SET a lane's reading was taken under -- the encoder, the
    chunker, and the retrieval mode -- not a digest of them, because the drift report names which
    knob moved. It stays an open map for the same reason the chunk metadata does: the knobs are
    the retrieval surface's to add to.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    resolved: bool
    model: str | None = None
    corpus_root: str | None = None
    retrieval_fingerprint: JsonObject | None = None


class AgentProfileRecord(ArtifactContract):
    """`agent_profile.json`: one composed operating profile and where each value came from.

    A field's `value` is whichever lane measured it -- a top-k integer, a policy name, a context
    budget -- so the per-field entry stays an open map; what the contract fixes is that every field
    HAS one, beside the anchor it was checked against, the drift findings, and the replay commands.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.agent-profile"]
    schema_version: Literal["1.0.0"]
    generated_at: str = Field(min_length=1)
    anchor: ProfileAnchor
    drift: JsonObject = Field(default_factory=dict)
    states: dict[str, int] = Field(default_factory=dict)
    fields: dict[str, JsonObject] = Field(default_factory=dict)
    replay: JsonObject = Field(default_factory=dict)
