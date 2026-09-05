"""Knowledge-graph store contracts: the node and edge rows, the metadata, and the summaries.

A graph generation persists the same evidence a vector store does -- offset-bearing spans -- so
its rows are modelled at the same granularity: one node, one edge, one mention span each carrying
`doc_id` plus character offsets. The query engine is a DuckDB database built from those rows in
memory; where a generation does materialize one, the dataset manifest binds it as an opaque member
rather than modelling a format DuckDB owns.
"""

from typing import Literal

from pydantic import Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.retrieval_graph.common import RetrievalRow

GRAPH_NODE_SCHEMA_ID = "llb.graph-node"
GRAPH_EDGE_SCHEMA_ID = "llb.graph-edge"
GRAPH_META_SCHEMA_ID = "llb.graph-meta"
COMMUNITY_SUMMARIES_SCHEMA_ID = "llb.graph-community-summaries"


class GraphMentionRow(RetrievalRow):
    """An offset-bearing evidence span plus the section that contains it."""

    doc_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str
    section_title: str


class GraphNodeRow(ArtifactContract):
    """`nodes.jsonl`: one ontology-typed entity with its exact-grounded mention spans."""

    schema_id: Literal["llb.graph-node"]
    schema_version: Literal["1.0.0"]
    node_id: int
    name: str
    type: str
    confidence: float
    aliases: list[str] = Field(default_factory=list)
    mentions: list[GraphMentionRow] = Field(default_factory=list)
    # -1 is `llb.graph.model.NO_COMMUNITY`: a node community detection did not assign.
    community_id: int = -1


class GraphEdgeRow(ArtifactContract):
    """`edges.jsonl`: one subject-relation-object fact with the span that evidences it."""

    schema_id: Literal["llb.graph-edge"]
    schema_version: Literal["1.0.0"]
    edge_id: int
    src: int
    dst: int
    relation: str
    evidence: GraphMentionRow


class GraphMetaDocument(ArtifactContract):
    """`graph_meta.json`: the shape of one graph generation and the corpus it was built from."""

    schema_id: Literal["llb.graph-meta"]
    schema_version: Literal["1.0.0"]
    backend: str
    n_nodes: int = Field(ge=0)
    n_edges: int = Field(ge=0)
    n_communities: int = Field(ge=0)
    n_documents: int = Field(ge=0)
    khop_depth: int = Field(ge=0)
    # Per-document content hashes: the manifest-diff contract `refresh-index` diffs against.
    doc_fingerprints: dict[str, str] = Field(default_factory=dict)
    refreshed_from: str | None = None


class CommunitySummariesDocument(ArtifactContract):
    """`community_summaries.json`: one narrative summary per detected community.

    The summaries are tagged DIAGNOSTIC -- they are never returned by `retrieve` and never
    span-scored -- so the contract keeps them in their own member rather than folding them into
    the node rows, where a reader could mistake generated prose for evidence.
    """

    schema_id: Literal["llb.graph-community-summaries"]
    schema_version: Literal["1.0.0"]
    summaries: dict[str, str] = Field(default_factory=dict)
