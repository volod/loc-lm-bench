"""The registered run-bundle, board, and orchestration contract families.

Every durable file a published run leaves behind now names a family, a current version, and the
older form this build still reads. As in data prep and retrieval, `legacy_version` is what lets a
reader open the bundles this project has already written: they carry no identity at all, so the
caller that knows WHICH family it opened supplies the version their form corresponds to.

Four families here also declare `legacy_document_field`, because their pre-contract file was not a
record with room for an identity: a benchmark row was the bare cell, a design and an analysis
sidecar were the bare bodies, and `artifacts.json` was the bare stage map.
"""

from llb.artifacts.definitions import ContractDefinition
from llb.core.contracts.artifact_catalog import FormatBinding
from llb.core.contracts.orchestration import (
    STAGES_FIELD,
    AgentProfileRecord,
    AutoRagJournalEvent,
    AutoRagManifest,
    AutoRagRecommendation,
    AutoRagStageLinks,
    AutoRagStageResult,
    MissAnalysisReport,
    MissRecord,
)
from llb.core.contracts.run_bundle import (
    ANALYSIS_FIELD,
    CELL_FIELD,
    DESIGN_FIELD,
    BenchmarkCellRecord,
    CaseProgressRecord,
    CaseRetrievalRecord,
    CaseScoreRecord,
    ContextProbeRecord,
    RunAbortRecord,
    RunProgressMeta,
    StudyAnalysisRecord,
    StudyDesignRecord,
)
from llb.core.contracts.runs import RunManifest

JSON_DOCUMENT = (
    FormatBinding(format="json", media_type="application/json", granularity="document"),
)
YAML_DOCUMENT = (
    FormatBinding(format="yaml", media_type="application/yaml", granularity="document"),
)
JSONL_ROWS = (FormatBinding(format="jsonl", media_type="application/x-ndjson", granularity="row"),)

READABLE_FOR_RELEASE = "Version 1 remains readable for this release line."


def run_definitions() -> tuple[ContractDefinition, ...]:
    """Every run-bundle, board, and orchestration family, in schema-id order."""
    return (
        ContractDefinition(
            schema_id="llb.agent-profile",
            description="One composed agent operating profile and the evidence behind each field.",
            current_version="1.0.0",
            models={"1.0.0": AgentProfileRecord},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.auto-rag-journal-event",
            description="One appended auto-RAG journal line: a stage changed state at a time.",
            current_version="1.0.0",
            models={"1.0.0": AutoRagJournalEvent},
            bindings=JSONL_ROWS,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.auto-rag-manifest",
            description="The pinned settings an auto-RAG resume is refused against.",
            current_version="1.0.0",
            models={"1.0.0": AutoRagManifest},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.auto-rag-recommendation",
            description="The serving, chunking, retrieval, and prompt configuration recommended.",
            current_version="1.0.0",
            models={"1.0.0": AutoRagRecommendation},
            bindings=(*JSON_DOCUMENT, *YAML_DOCUMENT),
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.auto-rag-stage-links",
            description="Every auto-RAG stage result, collected beside the recommendation.",
            current_version="1.0.0",
            models={"1.0.0": AutoRagStageLinks},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
            legacy_document_field=STAGES_FIELD,
        ),
        ContractDefinition(
            schema_id="llb.auto-rag-stage-result",
            description="The durable marker a resume skips one completed auto-RAG stage on.",
            current_version="1.0.0",
            models={"1.0.0": AutoRagStageResult},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.benchmark-cell",
            description="One measured cell of a benchmark category run, in that lane's columns.",
            current_version="1.0.0",
            models={"1.0.0": BenchmarkCellRecord},
            bindings=JSONL_ROWS,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
            legacy_document_field=CELL_FIELD,
        ),
        ContractDefinition(
            schema_id="llb.case-progress",
            description="One completed case of an interrupted run, as a resume replays it.",
            current_version="1.0.0",
            models={"1.0.0": CaseProgressRecord},
            bindings=JSONL_ROWS,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.case-retrieval",
            description="What one scored case's context held, against the spans it needed.",
            current_version="1.0.0",
            models={"1.0.0": CaseRetrievalRecord},
            bindings=JSONL_ROWS,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.case-score",
            description="How one evaluation case scored, column by column.",
            current_version="1.0.0",
            models={"1.0.0": CaseScoreRecord},
            bindings=JSONL_ROWS,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.context-probe",
            description="How one case answered with its gold context deliberately withheld.",
            current_version="1.0.0",
            models={"1.0.0": ContextProbeRecord},
            bindings=JSONL_ROWS,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.miss-analysis",
            description="The machine-readable miss analysis an operator recommendation reads.",
            current_version="1.0.0",
            models={"1.0.0": MissAnalysisReport},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.miss-record",
            description="One case that missed, and the class the analysis put it in.",
            current_version="1.0.0",
            models={"1.0.0": MissRecord},
            bindings=JSONL_ROWS,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.run-abort",
            description="A run stopped on a declared budget, and whether it can be resumed.",
            current_version="1.0.0",
            models={"1.0.0": RunAbortRecord},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.run-manifest",
            description="One run's immutable record: config, environment, and headline metrics.",
            current_version="1.0.0",
            models={"1.0.0": RunManifest},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.run-progress-meta",
            description="The run identity a resume onto an interrupted staging directory needs.",
            current_version="1.0.0",
            models={"1.0.0": RunProgressMeta},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.study-analysis",
            description="The reading one study took against its declared design.",
            current_version="1.0.0",
            models={"1.0.0": StudyAnalysisRecord},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
            legacy_document_field=ANALYSIS_FIELD,
        ),
        ContractDefinition(
            schema_id="llb.study-design",
            description="What one study declared about its cells before it measured any of them.",
            current_version="1.0.0",
            models={"1.0.0": StudyDesignRecord},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
            legacy_document_field=DESIGN_FIELD,
        ),
    )
