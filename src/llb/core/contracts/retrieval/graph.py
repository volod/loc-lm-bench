"""Knowledge-graph store record contracts: nodes, edges, metadata, and community summaries.

The graph store persists `nodes.jsonl`, `edges.jsonl`, `graph_meta.json`, and the diagnostic
`community_summaries.json`. Every node mention and every edge evidence span is an exact-grounded
source span, which is what lets a serialized subgraph score on the same span metric the vector
path uses -- so the contract states those offsets rather than leaving them to a `dict` that
happened to carry them. Reading a graph store through these models needs neither DuckDB nor the
graph extra.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llb.core.contracts.artifacts import ArtifactContract

GRAPH_NODE_SCHEMA_ID = "llb.graph-node"
GRAPH_EDGE_SCHEMA_ID = "llb.graph-edge"
GRAPH_STORE_META_SCHEMA_ID = "llb.graph-store-meta"
GRAPH_COMMUNITY_SUMMARIES_SCHEMA_ID = "llb.graph-community-summaries"


class GraphRow(BaseModel):
    """Strict nested row shared by the graph record contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class GraphMentionRecord(GraphRow):
    """An offset-bearing evidence span plus the section that contains it."""

    doc_id: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str
    section_title: str

    @model_validator(mode="after")
    def _check_offsets(self) -> "GraphMentionRecord":
        if self.char_end < self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be >= char_start ({self.char_start})"
            )
        return self


class GraphNodeRecord(ArtifactContract):
    """One entity node: its ontology type, its aliases, and every mention that grounds it."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.graph-node"]
    schema_version: Literal["1.0.0"]
    node_id: int = Field(ge=0)
    name: str
    type: str
    confidence: float
    aliases: list[str] = Field(default_factory=list)
    mentions: list[GraphMentionRecord] = Field(default_factory=list)
    community_id: int


class GraphEdgeRecord(ArtifactContract):
    """One subject-relation-object fact, with the span that evidences it."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.graph-edge"]
    schema_version: Literal["1.0.0"]
    edge_id: int = Field(ge=0)
    src: int = Field(ge=0)
    dst: int = Field(ge=0)
    relation: str
    evidence: GraphMentionRecord


class GraphStoreMetaRecord(ArtifactContract):
    """`graph_meta.json`: what a persisted knowledge graph is, and what it was built from."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.graph-store-meta"]
    schema_version: Literal["1.0.0"]
    backend: str = Field(min_length=1)
    n_nodes: int = Field(ge=0)
    n_edges: int = Field(ge=0)
    n_communities: int = Field(ge=0)
    n_documents: int = Field(ge=0)
    khop_depth: int = Field(ge=0)
    doc_fingerprints: dict[str, str]
    refreshed_from: str | None = None


class GraphCommunitySummaries(ArtifactContract):
    """`community_summaries.json`: community id -> narrative summary.

    Tagged DIAGNOSTIC everywhere it is produced -- a summary is never returned by `retrieve` and
    never span-scored -- so the contract records what it is rather than promoting it to evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.graph-community-summaries"]
    schema_version: Literal["1.0.0"]
    role: Literal["diagnostic"] = "diagnostic"
    summaries: dict[str, str] = Field(default_factory=dict)
