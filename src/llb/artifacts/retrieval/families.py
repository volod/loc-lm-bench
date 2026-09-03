"""The registered retrieval, graph, and prompt-system contract families.

Every file a store, a graph, or a prompt-system package is made of now names a family, a current
version, and the older versions this build still reads. As in data prep, `legacy_version` is what
lets a reader open the artifacts this project already wrote: they carry no identity at all, so the
caller that knows WHICH family it opened supplies the version their form corresponds to.
"""

from llb.artifacts.definitions import ContractDefinition
from llb.core.contracts.artifact_catalog import FormatBinding
from llb.core.contracts.retrieval.comparison import RetrievalComparisonSidecar
from llb.core.contracts.retrieval.graph import (
    GraphCommunitySummaries,
    GraphEdgeRecord,
    GraphNodeRecord,
    GraphStoreMetaRecord,
)
from llb.core.contracts.retrieval.query import QueryGlossary
from llb.core.contracts.retrieval.prompt_system import (
    PromptSystemAnthology,
    PromptSystemCandidates,
    PromptSystemDocMetadata,
    PromptSystemManifest,
    PromptSystemMapping,
)
from llb.core.contracts.retrieval.store import (
    RagChunkRecord,
    RagStoreMetaRecord,
)

JSON_DOCUMENT = (
    FormatBinding(format="json", media_type="application/json", granularity="document"),
)
JSONL_ROWS = (FormatBinding(format="jsonl", media_type="application/x-ndjson", granularity="row"),)

READABLE_FOR_RELEASE = "Version 1 remains readable for this release line."


def retrieval_definitions() -> tuple[ContractDefinition, ...]:
    """Every retrieval, graph, and prompt-system family, in schema-id order."""
    return (
        ContractDefinition(
            schema_id="llb.graph-community-summaries",
            description="Diagnostic narrative summary per detected graph community.",
            current_version="1.0.0",
            models={"1.0.0": GraphCommunitySummaries},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
            legacy_document_field="summaries",
        ),
        ContractDefinition(
            schema_id="llb.graph-edge",
            description="One subject-relation-object fact and the source span evidencing it.",
            current_version="1.0.0",
            models={"1.0.0": GraphEdgeRecord},
            bindings=JSONL_ROWS,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.graph-node",
            description="One ontology-typed entity node with its exact-grounded mentions.",
            current_version="1.0.0",
            models={"1.0.0": GraphNodeRecord},
            bindings=JSONL_ROWS,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.graph-store-meta",
            description="What a persisted knowledge graph is, and what it was built from.",
            current_version="1.0.0",
            models={"1.0.0": GraphStoreMetaRecord},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.prompt-system-anthology",
            description="The salient passages a prompt-system package selected from its corpus.",
            current_version="1.0.0",
            models={"1.0.0": PromptSystemAnthology},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
            legacy_document_field="passages",
        ),
        ContractDefinition(
            schema_id="llb.prompt-system-candidates",
            description="The rendered, budget-fitted prompt-system candidates under review.",
            current_version="1.0.0",
            models={"1.0.0": PromptSystemCandidates},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
            legacy_document_field="candidates",
        ),
        ContractDefinition(
            schema_id="llb.prompt-system-doc-metadata",
            description="Per-source-document metadata a prompt-system package summarized.",
            current_version="1.0.0",
            models={"1.0.0": PromptSystemDocMetadata},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
            legacy_document_field="documents",
        ),
        ContractDefinition(
            schema_id="llb.prompt-system-manifest",
            description="A prompt-system package's identity, budget, and candidate roll-up.",
            current_version="1.0.0",
            models={"1.0.0": PromptSystemManifest},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.prompt-system-mapping",
            description="Salient term to the anthology passages that ground it.",
            current_version="1.0.0",
            models={"1.0.0": PromptSystemMapping},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
            legacy_document_field="mapping",
        ),
        ContractDefinition(
            schema_id="llb.query-glossary",
            description="Alias-expansion table a query-preparation run applies before retrieval.",
            current_version="1.0.0",
            models={"1.0.0": QueryGlossary},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.rag-chunk",
            description="One indexed retrieval unit: an offset-exact span and how it was cut.",
            current_version="1.0.0",
            models={"1.0.0": RagChunkRecord},
            bindings=JSONL_ROWS,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.rag-store-meta",
            description="What a persisted vector store is, and every knob a query applies.",
            current_version="1.0.0",
            models={"1.0.0": RagStoreMetaRecord},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.retrieval-comparison",
            description="Envelope of one machine-readable retrieval comparison sidecar.",
            current_version="1.0.0",
            models={"1.0.0": RetrievalComparisonSidecar},
            bindings=JSON_DOCUMENT,
            deprecation_policy=READABLE_FOR_RELEASE,
            legacy_version="1.0.0",
            legacy_document_field="report",
        ),
    )
