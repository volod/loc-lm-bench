"""Vector stores, knowledge graphs, and prompt-system packages described as datasets.

Each of the three is a directory whose members only mean something together: a store's chunk rows
index into the vector index by build order, a graph's edges reference its node ids, a package's
mapping cites its anthology's passage ids. The project-owned members are bound to their registered
contracts; the index, the postings file, the vector matrix, and the semantic-tree sidecar are
bound OPAQUELY, naming the owner and format version rather than pretending this project models
their bytes.
"""

from pathlib import Path

from llb.artifacts.datasets import MemberSpec, OpaqueMemberSpec, describe_dataset
from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.registry import ContractRegistry
from llb.conflicts.semantic_tree.node import TREE_VERSION
from llb.core.contracts.artifacts import DatasetManifest
from llb.core.store_generations import resolve_store_dir
from llb.graph.constants import (
    DUCKDB_FILE,
    EDGES_FILE,
    META_FILE as GRAPH_META_FILE,
    NODES_FILE,
    SUMMARIES_FILE,
)
from llb.prep.ontology.constants import EXTRACTION_FILENAME, ONTOLOGY_FILENAME
from llb.prompt_system.pipeline import (
    ANTHOLOGY_FILE,
    CANDIDATES_FILE,
    MANIFEST_FILE,
    MAPPING_FILE,
    METADATA_FILE,
)
from llb.rag.stores.base import VECTORS_FILE
from llb.rag.vector_store.build import CHUNKS_FILE, LEXICAL_FILE, META_FILE, PARENTS_FILE
from llb.rag.vector_store.lexical import LEXICAL_INDEX_VERSION
from llb.rag.vector_store.vector_index import (
    FAISS_INDEX_FILE,
    RAG_BACKEND_FAISS,
    RAG_BACKENDS,
)

STORE_DATASET_ID = "llb-vector-store"
GRAPH_DATASET_ID = "llb-knowledge-graph"
PROMPT_SYSTEM_DATASET_ID = "llb-prompt-system-package"

# The FAISS index this project writes is a flat inner-product index over normalized vectors; the
# file format is FAISS's own and its evolution is theirs, so the binding names it rather than
# modelling it.
FAISS_OWNER = "faiss"
FAISS_INDEX_FORMAT_VERSION = "IndexFlatIP/1"
NUMPY_OWNER = "numpy"
NUMPY_ARRAY_FORMAT_VERSION = "npy/1"
DUCKDB_OWNER = "duckdb"
DUCKDB_FORMAT_VERSION = "duckdb-database/1"
LEXICAL_OWNER = "llb.rag.vector_store.lexical_index"
SEMANTIC_TREE_OWNER = "llb.conflicts.semantic_tree"

STORE_MEMBERS: tuple[MemberSpec, ...] = (
    MemberSpec("store-meta", META_FILE, "llb.rag-store-meta"),
    MemberSpec("chunks", CHUNKS_FILE, "llb.rag-chunk", "jsonl"),
    MemberSpec("parents", PARENTS_FILE, "llb.rag-chunk", "jsonl"),
)

GRAPH_MEMBERS: tuple[MemberSpec, ...] = (
    MemberSpec("graph-meta", GRAPH_META_FILE, "llb.graph-store-meta"),
    MemberSpec("nodes", NODES_FILE, "llb.graph-node", "jsonl"),
    MemberSpec("edges", EDGES_FILE, "llb.graph-edge", "jsonl"),
    MemberSpec("community-summaries", SUMMARIES_FILE, "llb.graph-community-summaries"),
    # The build inputs persisted beside the store so an incremental refresh can chain from them;
    # they are the data-prep families, written here by the same record builders.
    MemberSpec("extraction", EXTRACTION_FILENAME, "llb.ontology-extraction", "jsonl"),
    MemberSpec("ontology", ONTOLOGY_FILENAME, "llb.ontology"),
)

PROMPT_SYSTEM_MEMBERS: tuple[MemberSpec, ...] = (
    MemberSpec("package-manifest", MANIFEST_FILE, "llb.prompt-system-manifest"),
    MemberSpec("anthology", ANTHOLOGY_FILE, "llb.prompt-system-anthology"),
    MemberSpec("doc-metadata", METADATA_FILE, "llb.prompt-system-doc-metadata"),
    MemberSpec("graph-rag-mapping", MAPPING_FILE, "llb.prompt-system-mapping"),
    MemberSpec("candidates", CANDIDATES_FILE, "llb.prompt-system-candidates"),
)

_SEMANTIC_TREE_MEMBERS = ("tree.json", "tree_meta.json", "projection.json")


def store_opaque_members() -> tuple[OpaqueMemberSpec, ...]:
    """The store members whose bytes another owner's format defines."""
    adapters = tuple(
        OpaqueMemberSpec(
            f"vector-index-{backend}",
            f"{backend}/{VECTORS_FILE}",
            NUMPY_OWNER,
            NUMPY_ARRAY_FORMAT_VERSION,
            f"Build-order float32 vector matrix the {backend} adapter rebuilds from.",
        )
        for backend in RAG_BACKENDS
        if backend != RAG_BACKEND_FAISS
    )
    trees = tuple(
        OpaqueMemberSpec(
            f"semantic-tree-{name.split('.', 1)[0].replace('_', '-')}",
            f"semantic_tree/{name}",
            SEMANTIC_TREE_OWNER,
            TREE_VERSION,
            "Conflict-detection semantic prefix-tree sidecar built over this store.",
        )
        for name in _SEMANTIC_TREE_MEMBERS
    )
    return (
        OpaqueMemberSpec(
            "vector-index-faiss",
            FAISS_INDEX_FILE,
            FAISS_OWNER,
            FAISS_INDEX_FORMAT_VERSION,
            "Flat inner-product index over the normalized chunk embeddings.",
        ),
        OpaqueMemberSpec(
            "lexical-index",
            LEXICAL_FILE,
            LEXICAL_OWNER,
            LEXICAL_INDEX_VERSION,
            "BM25 postings whose terms are this tokenizer version's output.",
        ),
        *adapters,
        *trees,
    )


GRAPH_OPAQUE_MEMBERS: tuple[OpaqueMemberSpec, ...] = (
    OpaqueMemberSpec(
        "graph-database",
        DUCKDB_FILE,
        DUCKDB_OWNER,
        DUCKDB_FORMAT_VERSION,
        "Materialized node/edge query engine, rebuildable from the node and edge rows.",
    ),
)


def store_dataset_manifest(
    index_dir: Path | str,
    registry: ContractRegistry = DEFAULT_REGISTRY,
    resolve_live: bool = True,
) -> DatasetManifest:
    """Describe a vector store, by default the live generation beneath `index_dir`.

    A publisher passes `resolve_live=False`: it has just written one exact directory -- a base
    rebuild, or a refresh's staging directory -- and must describe that one, not whichever
    generation happens to be live beside it.
    """
    return describe_dataset(
        _target(index_dir, META_FILE, resolve_live),
        STORE_DATASET_ID,
        "One vector store: its chunk rows, its metadata, and the indexes built over them.",
        STORE_MEMBERS,
        registry,
        store_opaque_members(),
    )


def graph_dataset_manifest(
    graph_dir: Path | str,
    registry: ContractRegistry = DEFAULT_REGISTRY,
    resolve_live: bool = True,
) -> DatasetManifest:
    """Describe a knowledge-graph store, by default the live generation beneath `graph_dir`."""
    return describe_dataset(
        _target(graph_dir, GRAPH_META_FILE, resolve_live),
        GRAPH_DATASET_ID,
        "One knowledge graph: its nodes, its facts, and the communities detected over them.",
        GRAPH_MEMBERS,
        registry,
        GRAPH_OPAQUE_MEMBERS,
    )


def prompt_system_dataset_manifest(
    run_dir: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> DatasetManifest:
    """Describe one prompt-system package directory."""
    return describe_dataset(
        Path(run_dir),
        PROMPT_SYSTEM_DATASET_ID,
        "One prompt-system package: its corpus inputs, its candidates, and its manifest.",
        PROMPT_SYSTEM_MEMBERS,
        registry,
    )


def _target(directory: Path | str, meta_filename: str, resolve_live: bool) -> Path:
    return resolve_store_dir(directory, meta_filename) if resolve_live else Path(directory)
