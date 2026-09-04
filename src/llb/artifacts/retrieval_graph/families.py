"""The registered retrieval, graph, and prompt-system contract families.

A built store, a built graph, and a prepared prompt-system package are the three artifacts every
later measurement is taken against, and each is a SET of files rather than one document. The
families below name the project-owned records in those sets; the vector index, the posting list,
and the query database are named as opaque members of the dataset instead, because their bytes
belong to FAISS, to this project's tokenizer version, and to DuckDB respectively.

Every family declares `legacy_version`, because every one of these artifacts already exists on
disk without an identity: a reader that knows WHICH file it opened supplies the version, exactly
as the data-prep families do.
"""

from llb.artifacts.definitions import ContractDefinition, MigrationStep
from llb.artifacts.retrieval_graph.migrations import store_meta_v1_to_v2
from llb.core.contracts.artifact_catalog import FormatBinding
from llb.core.contracts.retrieval_graph.calibration import RoutingCalibrationReport
from llb.core.contracts.retrieval_graph.comparison import RetrievalComparisonReport
from llb.core.contracts.retrieval_graph.graph import (
    CommunitySummariesDocument,
    GraphEdgeRow,
    GraphMetaDocument,
    GraphNodeRow,
)
from llb.core.contracts.retrieval_graph.prompt_system import (
    AnthologyDocument,
    DocMetadataDocument,
    GraphRagMappingDocument,
    PromptCandidatesDocument,
    PromptSystemManifestDocument,
)
from llb.core.contracts.retrieval_graph.stores import (
    ChunkRow,
    RagStoreMetaDocument,
    RagStoreMetaDocumentV1,
)

JSON_DOCUMENT = (
    FormatBinding(format="json", media_type="application/json", granularity="document"),
)
JSONL_ROWS = (FormatBinding(format="jsonl", media_type="application/x-ndjson", granularity="row"),)

_RELEASE_LINE = "Version 1 remains readable for this release line."


def retrieval_graph_definitions() -> tuple[ContractDefinition, ...]:
    """Every retrieval, graph, and prompt-system family, in schema-id order."""
    return (
        ContractDefinition(
            schema_id="llb.fusion-routing-calibration",
            description="Held-out calibration of the fusion router: every policy and the frozen one.",
            current_version="1.0.0",
            models={"1.0.0": RoutingCalibrationReport},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.graph-community-summaries",
            description="Per-community narrative summaries, tagged diagnostic and never span-scored.",
            current_version="1.0.0",
            models={"1.0.0": CommunitySummariesDocument},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.graph-edge",
            description="One subject-relation-object fact and the span that evidences it.",
            current_version="1.0.0",
            models={"1.0.0": GraphEdgeRow},
            bindings=JSONL_ROWS,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.graph-meta",
            description="The shape of one graph generation and the corpus it was built from.",
            current_version="1.0.0",
            models={"1.0.0": GraphMetaDocument},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.graph-node",
            description="One ontology-typed entity with its exact-grounded mention spans.",
            current_version="1.0.0",
            models={"1.0.0": GraphNodeRow},
            bindings=JSONL_ROWS,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.prompt-system-anthology",
            description="The passages a candidate prompt may attach, in selection order.",
            current_version="1.0.0",
            models={"1.0.0": AnthologyDocument},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.prompt-system-candidates",
            description="Every generated prompt candidate and the review status it carries.",
            current_version="1.0.0",
            models={"1.0.0": PromptCandidatesDocument},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.prompt-system-doc-metadata",
            description="Per-document summary the prompt template surfaces, in corpus order.",
            current_version="1.0.0",
            models={"1.0.0": DocMetadataDocument},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.prompt-system-manifest",
            description="The digests and budget that make a prompt-system run addressable.",
            current_version="1.0.0",
            models={"1.0.0": PromptSystemManifestDocument},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.prompt-system-mapping",
            description="Each salient term and the anthology passage ids that ground it.",
            current_version="1.0.0",
            models={"1.0.0": GraphRagMappingDocument},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.rag-chunk",
            description="One indexed unit or one parent: the offset-bearing record a lane returns.",
            current_version="1.0.0",
            models={"1.0.0": ChunkRow},
            bindings=JSONL_ROWS,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.rag-store-meta",
            description="What a vector store was built from, by which encoder, over which corpus.",
            current_version="2.0.0",
            models={"1.0.0": RagStoreMetaDocumentV1, "2.0.0": RagStoreMetaDocument},
            bindings=JSON_DOCUMENT,
            deprecation_policy=(
                "Version 1 is read-and-migrate: it predates the recorded opaque index members."
            ),
            migrations=(
                MigrationStep(
                    from_version="1.0.0",
                    to_version="2.0.0",
                    description=(
                        "State the duplicate-collapse knobs a reader defaulted and declare no "
                        "index members."
                    ),
                    transform=store_meta_v1_to_v2,
                ),
            ),
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.retrieval-comparison",
            description="Every compared retrieval lane, its paired reading, and the verdict.",
            current_version="1.0.0",
            models={"1.0.0": RetrievalComparisonReport},
            bindings=JSON_DOCUMENT,
            deprecation_policy=_RELEASE_LINE,
            legacy_version="1.0.0",
        ),
    )
