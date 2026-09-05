"""Describe a built store, a built graph, or a prompt-system package as one dataset.

A generation is a SET of files that only mean something together: the chunk rows index into the
vector matrix by build order, the metadata says which encoder produced that matrix, and the
posting list was written by one tokenizer version. Reading one member and trusting the rest is how
a generation gets half-migrated, so this module names every member of each, binds the
project-owned ones to their record contracts and the rest to their owner's format, and reads all
of them through those bindings.

Members are DISCOVERED: a flat store has no parents member and says so by omission, while a
member that is present and unreadable is a refusal.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from llb.artifacts.bundles import bound_version
from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.registry import ContractRegistry
from llb.artifacts.retrieval_graph.opaque import content_digest
from llb.artifacts.retrieval_graph.stores import read_store_meta
from llb.core.contracts.artifacts import (
    ARTIFACT_GRANULARITIES,
    ARTIFACT_MEDIA_TYPES,
    ArtifactFormat,
    ContractReference,
    DatasetManifest,
    DatasetMember,
    DatasetQualityCheck,
    OpaqueBinding,
    RecordGranularity,
)
from llb.core.contracts.retrieval_graph.graph import (
    COMMUNITY_SUMMARIES_SCHEMA_ID,
    GRAPH_EDGE_SCHEMA_ID,
    GRAPH_META_SCHEMA_ID,
    GRAPH_NODE_SCHEMA_ID,
)
from llb.core.contracts.retrieval_graph.prompt_system import (
    ANTHOLOGY_SCHEMA_ID,
    CANDIDATES_SCHEMA_ID,
    DOC_METADATA_SCHEMA_ID,
    MAPPING_SCHEMA_ID,
    PROMPT_SYSTEM_MANIFEST_SCHEMA_ID,
)
from llb.core.contracts.retrieval_graph.stores import CHUNK_SCHEMA_ID, STORE_META_SCHEMA_ID
from llb.graph.constants import (
    DUCKDB_FILE,
    EDGES_FILE,
    META_FILE as GRAPH_META_FILE,
    NODES_FILE,
    SUMMARIES_FILE,
)
from llb.prompt_system.layout import (
    ANTHOLOGY_FILE,
    CANDIDATES_FILE,
    MANIFEST_FILE,
    MAPPING_FILE,
    METADATA_FILE,
)
from llb.rag.vector_store.layout import (
    CHUNKS_FILE,
    META_FILE as STORE_META_FILE,
    PARENTS_FILE,
)

OPAQUE_MEDIA_TYPE = "application/octet-stream"

VECTOR_STORE_DATASET_ID = "llb-vector-store-generation"
GRAPH_STORE_DATASET_ID = "llb-graph-store-generation"
PROMPT_SYSTEM_DATASET_ID = "llb-prompt-system-package"


@dataclass(frozen=True)
class GenerationMember:
    """One member of a generation: where it lives, and what contract or owner it answers to."""

    member_id: str
    relative_path: str
    schema_id: str | None = None
    artifact_format: ArtifactFormat = "json"
    opaque_binding: OpaqueBinding | None = None
    # The digest the generation itself PUBLISHED for this member, where it recorded one. Binding
    # at the recorded value rather than at what the file hashes today is what makes the survey a
    # tamper check: a member rebuilt or swapped after publication no longer matches its store.
    declared_digest: str | None = None


VECTOR_STORE_MEMBERS: tuple[GenerationMember, ...] = (
    GenerationMember("store-meta", STORE_META_FILE, STORE_META_SCHEMA_ID),
    GenerationMember("chunks", CHUNKS_FILE, CHUNK_SCHEMA_ID, "jsonl"),
    GenerationMember("parents", PARENTS_FILE, CHUNK_SCHEMA_ID, "jsonl"),
)

GRAPH_STORE_MEMBERS: tuple[GenerationMember, ...] = (
    GenerationMember("graph-meta", GRAPH_META_FILE, GRAPH_META_SCHEMA_ID),
    GenerationMember("nodes", NODES_FILE, GRAPH_NODE_SCHEMA_ID, "jsonl"),
    GenerationMember("edges", EDGES_FILE, GRAPH_EDGE_SCHEMA_ID, "jsonl"),
    GenerationMember("community-summaries", SUMMARIES_FILE, COMMUNITY_SUMMARIES_SCHEMA_ID),
    GenerationMember(
        "graph-database",
        DUCKDB_FILE,
        artifact_format="opaque",
        opaque_binding=OpaqueBinding(
            owner="duckdb",
            format="duckdb-database",
            format_version="1",
            description="Materialized node/edge query engine; rebuilt in memory when absent.",
        ),
    ),
)

PROMPT_SYSTEM_MEMBERS: tuple[GenerationMember, ...] = (
    GenerationMember("manifest", MANIFEST_FILE, PROMPT_SYSTEM_MANIFEST_SCHEMA_ID),
    GenerationMember("anthology", ANTHOLOGY_FILE, ANTHOLOGY_SCHEMA_ID),
    GenerationMember("doc-metadata", METADATA_FILE, DOC_METADATA_SCHEMA_ID),
    GenerationMember("graph-rag-mapping", MAPPING_FILE, MAPPING_SCHEMA_ID),
    GenerationMember("candidates", CANDIDATES_FILE, CANDIDATES_SCHEMA_ID),
)

_QUALITY_CHECKS = (
    DatasetQualityCheck(
        check_id="member-contract-dispatch",
        kind="structural",
        description="Every present member resolves to its registered contract's current version.",
    ),
    DatasetQualityCheck(
        check_id="member-digest",
        kind="structural",
        description="Every member's content matches the digest recorded when it was described.",
    ),
    DatasetQualityCheck(
        check_id="opaque-member-owner",
        kind="structural",
        description="Every member this project does not define the bytes of names its owner.",
    ),
)


def vector_store_manifest(
    store_dir: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> DatasetManifest:
    """Describe one built vector-store generation, including its opaque index members."""
    root = Path(store_dir)
    return _manifest(
        root,
        VECTOR_STORE_DATASET_ID,
        "One vector-store generation: its rows, its metadata, and the indexes over them.",
        (*VECTOR_STORE_MEMBERS, *_declared_index_members(root, registry)),
        registry,
    )


def graph_store_manifest(
    graph_dir: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> DatasetManifest:
    """Describe one built graph generation: its nodes, edges, metadata, and summaries."""
    return _manifest(
        Path(graph_dir),
        GRAPH_STORE_DATASET_ID,
        "One graph generation: the entity and fact rows and everything that indexes them.",
        GRAPH_STORE_MEMBERS,
        registry,
    )


def prompt_system_manifest(
    package_dir: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> DatasetManifest:
    """Describe one prepared prompt-system package: the corpus inputs and the candidates."""
    return _manifest(
        Path(package_dir),
        PROMPT_SYSTEM_DATASET_ID,
        "One prompt-system package: the prepared corpus inputs and the reviewable candidates.",
        PROMPT_SYSTEM_MEMBERS,
        registry,
    )


def _declared_index_members(root: Path, registry: ContractRegistry) -> tuple[GenerationMember, ...]:
    """The opaque members the store's own metadata declares, bound by their owner's format.

    Discovered from the metadata rather than from the directory: which file the vector index lives
    in depends on the backend the store was built with, and the store is the thing that knows.
    """
    meta_path = root / STORE_META_FILE
    if not meta_path.is_file():
        return ()
    document = read_store_meta(meta_path, registry)
    return tuple(
        GenerationMember(
            member.member_id,
            member.path,
            artifact_format="opaque",
            opaque_binding=OpaqueBinding(
                owner=member.owner,
                format=member.format,
                format_version=member.format_version,
                description=member.description,
            ),
            declared_digest=member.digest,
        )
        for member in document.index_members
    )


def _manifest(
    root: Path,
    dataset_id: str,
    description: str,
    members: tuple[GenerationMember, ...],
    registry: ContractRegistry,
) -> DatasetManifest:
    described = [
        _member(root, member, registry)
        for member in members
        if (root / member.relative_path).exists()
    ]
    if not described:
        raise FileNotFoundError(f"{root}: no registered member of this generation is present")
    return DatasetManifest(
        schema_id="llb.dataset-manifest",
        schema_version="1.1.0",
        dataset_id=dataset_id,
        description=description,
        owner="loc-lm-bench maintainers",
        members=described,
        quality_checks=list(_QUALITY_CHECKS),
    )


def _member(root: Path, spec: GenerationMember, registry: ContractRegistry) -> DatasetMember:
    path = root / spec.relative_path
    if spec.schema_id is None:
        return DatasetMember(
            member_id=spec.member_id,
            path=spec.relative_path,
            format="opaque",
            media_type=OPAQUE_MEDIA_TYPE,
            granularity="opaque",
            digest=spec.declared_digest or content_digest(path),
            opaque_binding=spec.opaque_binding,
        )
    return DatasetMember(
        member_id=spec.member_id,
        path=spec.relative_path,
        format=spec.artifact_format,
        media_type=ARTIFACT_MEDIA_TYPES[spec.artifact_format],
        granularity=cast(RecordGranularity, ARTIFACT_GRANULARITIES[spec.artifact_format]),
        digest=content_digest(path),
        record_contract=ContractReference(
            schema_id=spec.schema_id,
            schema_version=bound_version(path, registry.definition(spec.schema_id)),
        ),
    )
