"""The registered run, study, board, and orchestration contract families.

A run bundle is the primary downstream API of this project: everything a board ranks, an analysis
re-reads, or an external consumer validates comes out of one. The families below name every
project-owned record in that surface -- the bundle head, the per-case rows of each measured
method, the resume and abort records an interrupted run leaves, the study records a benchmark
predeclares and reads out, the board's miss analysis, and the auto-RAG orchestration trail.

Every family declares `legacy_version`, because every one of these records already exists on disk
without an identity: a reader that knows WHICH file it opened supplies the version, exactly as the
data-prep and retrieval families do.
"""

from llb.artifacts.definitions import ContractDefinition, MigrationStep
from llb.artifacts.run_bundle.migrations import run_manifest_v1_to_v2
from llb.core.contracts.artifact_catalog import FormatBinding
from llb.core.contracts.run_bundle.auto_rag import (
    AutoRagJournalEvent,
    AutoRagManifestDocument,
    AutoRagStageResult,
    RagRecommendationDocument,
)
from llb.core.contracts.run_bundle.board import MissAnalysisDocument, MissCaseRow
from llb.core.contracts.run_bundle.journals import (
    CaseProgressMeta,
    CaseProgressRow,
    JudgeBudgetAbort,
)
from llb.core.contracts.run_bundle.manifest import RunManifestDocument, RunManifestV1
from llb.core.contracts.run_bundle.rows import (
    AgenticCaseRecord,
    CaseScoreRecord,
    RetrievalCaseRecord,
    SecurityCaseRecord,
    StructuredCaseRecord,
    SummarizationCaseRecord,
    TextAnalysisCaseRecord,
    ToolingCaseRecord,
)
from llb.core.contracts.run_bundle.studies import StudyAnalysisDocument, StudyDesignDocument

JSON_DOCUMENT = (
    FormatBinding(format="json", media_type="application/json", granularity="document"),
)
YAML_DOCUMENT = (
    FormatBinding(format="yaml", media_type="application/yaml", granularity="document"),
)
JSONL_ROWS = (FormatBinding(format="jsonl", media_type="application/x-ndjson", granularity="row"),)

_RELEASE_LINE = "Version 1 remains readable for this release line."

# Every per-case row family shares the same shape of declaration: one version, JSONL rows, and a
# legacy read version, because each one was written for releases before the registry existed.
_CASE_ROW_FAMILIES = (
    (
        "llb.agentic-case",
        AgenticCaseRecord,
        "One agentic episode: its outcome, loop counters, and context accounting.",
    ),
    (
        "llb.case-score",
        CaseScoreRecord,
        "One scored RAG evaluation case: correctness, retrieval, and delivery columns.",
    ),
    (
        "llb.retrieval-case",
        RetrievalCaseRecord,
        "What one case's scored context contained, against the gold spans it was scored on.",
    ),
    (
        "llb.security-case",
        SecurityCaseRecord,
        "One security attack or benign-control case and the refusal it earned.",
    ),
    (
        "llb.structured-case",
        StructuredCaseRecord,
        "One structured-output case: schema conformance and per-field accuracy.",
    ),
    (
        "llb.summarization-case",
        SummarizationCaseRecord,
        "One summarization case: the coverage it reached and its faithfulness.",
    ),
    (
        "llb.text-analysis-case",
        TextAnalysisCaseRecord,
        "One text-analysis document: objective label recovery over its subtasks.",
    ),
    (
        "llb.tooling-case",
        ToolingCaseRecord,
        "One tooling case: which tool was selected and whether its arguments held.",
    ),
)


def _case_row_definitions() -> tuple[ContractDefinition, ...]:
    return tuple(
        ContractDefinition(
            schema_id=schema_id,
            description=description,
            current_version="1.0.0",
            models={"1.0.0": model},
            bindings=JSONL_ROWS,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        )
        for schema_id, model, description in _CASE_ROW_FAMILIES
    )


def run_bundle_definitions() -> tuple[ContractDefinition, ...]:
    """Every run, study, board, and orchestration family, in schema-id order."""
    return (
        ContractDefinition(
            schema_id="llb.auto-rag-journal-event",
            description="One append-only auto-RAG journal line: what happened to a stage, and when.",
            current_version="1.0.0",
            models={"1.0.0": AutoRagJournalEvent},
            bindings=JSONL_ROWS,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.auto-rag-manifest",
            description="Every score- or artifact-affecting auto-RAG input, pinned at the first stage.",
            current_version="1.0.0",
            models={"1.0.0": AutoRagManifestDocument},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.auto-rag-stage-result",
            description="One auto-RAG stage, published only after all of its artifacts are durable.",
            current_version="1.0.0",
            models={"1.0.0": AutoRagStageResult},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        *_case_row_definitions(),
        ContractDefinition(
            schema_id="llb.judge-budget-abort",
            description="A judge stopped on its declared spend or call budget, state intact.",
            current_version="1.0.0",
            models={"1.0.0": JudgeBudgetAbort},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.miss-analysis",
            description="The machine-readable miss summary a tuning recommendation is read from.",
            current_version="1.0.0",
            models={"1.0.0": MissAnalysisDocument},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.miss-case",
            description="One classified miss: a pointer a human follows back into the run.",
            current_version="1.0.0",
            models={"1.0.0": MissCaseRow},
            bindings=JSONL_ROWS,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.rag-recommendation",
            description="The configuration auto-RAG selected, and the stage readings behind it.",
            current_version="1.0.0",
            models={"1.0.0": RagRecommendationDocument},
            bindings=YAML_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.run-manifest",
            description="The head of one run bundle: what ran, on what, and what it measured.",
            current_version="2.0.0",
            models={"1.0.0": RunManifestV1, "2.0.0": RunManifestDocument},
            bindings=JSON_DOCUMENT,
            deprecation_policy=(
                "Manifest 1.0 is read-and-migrate: it predates the declared bundle members."
            ),
            migrations=(
                MigrationStep(
                    from_version="1.0.0",
                    to_version="2.0.0",
                    description=(
                        "Carry the run head forward and declare that it states no score-row "
                        "contract and no additional artifacts."
                    ),
                    transform=run_manifest_v1_to_v2,
                ),
            ),
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.run-progress",
            description="One completed case in the append-only resume journal of a durable run.",
            current_version="1.0.0",
            models={"1.0.0": CaseProgressRow},
            bindings=JSONL_ROWS,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.run-progress-meta",
            description="The identity a resume must still match to be a continuation of the run.",
            current_version="1.0.0",
            models={"1.0.0": CaseProgressMeta},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.study-analysis",
            description="What a benchmark study read out of the run it had committed to.",
            current_version="1.0.0",
            models={"1.0.0": StudyAnalysisDocument},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.study-design",
            description="What a benchmark study fixed before it ran: sample, roster, and gates.",
            current_version="1.0.0",
            models={"1.0.0": StudyDesignDocument},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
    )
