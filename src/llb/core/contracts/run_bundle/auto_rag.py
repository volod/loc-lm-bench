"""Auto-RAG orchestration records: the pinned run, its stage results, and the recommendation.

The auto-RAG pipeline is a resumable sequence of stages, so its records answer two questions no
single run bundle does: whether resuming this directory is the SAME run (the manifest fingerprint),
and which stages already completed durably (the stage results and the journal). The recommendation
at the end is the artifact an operator copies into a standalone configuration, which is why it is
the one auto-RAG record with a fully modelled body.
"""

from typing import Final, Literal

from pydantic import Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject
from llb.core.contracts.run_bundle.common import RunBundleRow

AUTO_RAG_MANIFEST_SCHEMA_ID: Final = "llb.auto-rag-manifest"
AUTO_RAG_STAGE_RESULT_SCHEMA_ID: Final = "llb.auto-rag-stage-result"
AUTO_RAG_JOURNAL_EVENT_SCHEMA_ID: Final = "llb.auto-rag-journal-event"
RAG_RECOMMENDATION_SCHEMA_ID: Final = "llb.rag-recommendation"


class AutoRagRunSettings(RunBundleRow):
    """Every score- or artifact-affecting input of one auto-RAG run.

    This is the record the run fingerprint is taken over, which is why it is modelled field by
    field rather than left an open mapping: a resume that compared a digest of "whatever keys the
    file held" would accept a manifest whose settings this build can no longer read.
    """

    corpus: str
    data_dir: str
    run_id: str = Field(min_length=1)
    draft_model: str
    candidates: str
    candidate_models: list[str] = Field(default_factory=list)
    gate_policy: str
    judge_model: str | None = None
    judge_base_url: str | None = None
    egress_consent: bool
    max_usd: float | None = None
    max_calls: int | None = None
    max_items: int
    doc_limit: int | None = None
    seed: int
    draft_max_tokens: int
    draft_num_ctx: int | None = None
    draft_concurrency: int
    verify_threshold: float
    min_accept_rate: float
    retrieval_k: int
    retrieval_recall_gate: float
    trials: int
    screen_limit: int
    min_finalists: int
    objectives: str
    eval_limit: int | None = None
    max_model_len: int
    parity_check: bool


class AutoRagManifestDocument(ArtifactContract):
    """`manifest.json`: the settings pinned at the first stage, and their fingerprint.

    `fingerprint` is a digest of the settings below it, and it is what a resume compares against:
    a directory reopened with different settings is a different run, and continuing it would mix
    two configurations into one recommendation.
    """

    schema_id: Literal["llb.auto-rag-manifest"] = AUTO_RAG_MANIFEST_SCHEMA_ID
    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: Literal["auto-rag"] = "auto-rag"
    fingerprint: str = Field(min_length=1)
    settings: AutoRagRunSettings


class AutoRagStageResult(ArtifactContract):
    """`stages/<stage>/result.json`: one stage published only after its artifacts are durable.

    `result` is the stage's own payload -- an ingest states a corpus, a joint search states a
    scoreboard -- so what the contract binds is the completion protocol the resume depends on:
    which stage this is, and that it finished.
    """

    schema_id: Literal["llb.auto-rag-stage-result"] = AUTO_RAG_STAGE_RESULT_SCHEMA_ID
    schema_version: Literal["1.0.0"] = "1.0.0"
    status: Literal["completed"] = "completed"
    stage: str = Field(min_length=1)
    result: JsonObject


class AutoRagJournalEvent(ArtifactContract):
    """One append-only line of `journal.jsonl`: what happened to a stage and when."""

    schema_id: Literal["llb.auto-rag-journal-event"] = AUTO_RAG_JOURNAL_EVENT_SCHEMA_ID
    schema_version: Literal["1.0.0"] = "1.0.0"
    time: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    status: str = Field(min_length=1)
    result_digest: str | None = None
    error_type: str | None = None
    reason: str | None = None


class ServingRecommendation(RunBundleRow):
    """How the recommended model is to be served."""

    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    cpu_offload_gb: float | None = None
    kv_offloading_size_gb: float | None = None
    dtype: str | None = None
    quantization: str | None = None
    n_gpu_layers: int | None = None


class ChunkingRecommendation(RunBundleRow):
    """The chunking the recommended store was built with."""

    strategy: str
    size: int
    overlap: int


class RetrievalRecommendation(RunBundleRow):
    """The retrieval lane the recommendation was measured on."""

    backend: str
    mode: str
    top_k: int
    fusion_weight: float | None = None
    fusion_candidates: int | None = None
    reranker: str | None = None
    rerank_candidates: int | None = None
    query_prep: list[str] = Field(default_factory=list)
    context_budget: int | None = None


class PromptSystemRecommendation(RunBundleRow):
    """The prompt-system package the recommendation was measured with.

    `knowledge_tree` is the selected candidate's own tree provenance -- its depth, budget, and
    digest -- so its keys belong to the prompt-system packager that produced them rather than to
    this record, which only says which package the measurement used.
    """

    id: str
    package: str
    knowledge_tree: JsonObject | None = None


class RagRecommendationDocument(ArtifactContract):
    """`rag_recommendation.yaml`: the configuration auto-RAG selected, and what backs it.

    `evidence` holds each stage's own reading verbatim -- the verification accept rate, the
    retrieval validation, the joint search scoreboard, and the final split -- so the operator
    copying the configuration above it can see, in the same file, what the numbers were.
    """

    schema_id: Literal["llb.rag-recommendation"] = RAG_RECOMMENDATION_SCHEMA_ID
    schema_version: Literal["1.0.0"] = "1.0.0"
    model: str
    model_name: str
    backend: str
    serving: ServingRecommendation
    chunking: ChunkingRecommendation
    retrieval: RetrievalRecommendation
    prompt_system: PromptSystemRecommendation
    evidence: JsonObject
