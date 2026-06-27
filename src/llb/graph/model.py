"""Typed in-memory knowledge graph for the GraphRAG backend (GraphRAG backend).

Nodes are entities, edges are subject-relation-object facts -- both built from the ontology-assisted drafting
`DocExtraction` (no second extraction framework). Every node mention and every edge's evidence
keeps its `doc_id` + char offsets + exact text (a `GraphMention`), so a serialized subgraph or
community scores on the SAME source-span metric the FAISS path uses (source-span metric). The induced ontology
type/confidence, the containing `section_title`, and the detected `community_id` ride along as
typed properties, exactly the carry-through the spec calls for.

Plain dataclasses + TypedDicts so the construction, community detection, and retrieval strategies
are pure and unit-testable WITHOUT DuckDB (only the persisted `GraphStore` imports it).
"""

from dataclasses import dataclass, field

from typing_extensions import TypedDict

NO_COMMUNITY = -1  # a node not yet assigned to a community
UNKNOWN_CONFIDENCE = 0.0  # an entity whose type was not in the induced ontology


class GraphMention(TypedDict):
    """An offset-bearing evidence span (a `SourceSpanRecord` + its containing section)."""

    doc_id: str
    char_start: int
    char_end: int
    text: str
    section_title: str


@dataclass
class GraphNode:
    """An entity node: an ontology-typed concept with exact-grounded mention spans."""

    node_id: int
    name: str
    type: str  # induced ontology entity type (or the raw extracted type)
    confidence: float  # induced OntologyType.confidence for `type` (0.0 if not induced)
    aliases: list[str] = field(default_factory=list)
    mentions: list[GraphMention] = field(default_factory=list)
    community_id: int = NO_COMMUNITY


@dataclass
class GraphEdge:
    """A subject-relation-object fact: a directed edge with exact-grounded evidence."""

    edge_id: int
    src: int  # subject node_id
    dst: int  # object node_id
    relation: str
    evidence: GraphMention


@dataclass
class KnowledgeGraph:
    """The RAM-resident graph (a MEDIUM corpus is ~1e4-1e5 edges, so it fits)."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def node_by_id(self) -> dict[int, GraphNode]:
        return {n.node_id: n for n in self.nodes}

    def adjacency(self) -> dict[int, set[int]]:
        """Undirected neighbor map (k-hop expansion treats facts as undirected links)."""
        adj: dict[int, set[int]] = {n.node_id: set() for n in self.nodes}
        for edge in self.edges:
            if edge.src in adj and edge.dst in adj:
                adj[edge.src].add(edge.dst)
                adj[edge.dst].add(edge.src)
        return adj

    def community_members(self) -> dict[int, list[int]]:
        """community_id -> member node ids (deterministic insertion order)."""
        members: dict[int, list[int]] = {}
        for node in self.nodes:
            members.setdefault(node.community_id, []).append(node.node_id)
        return members
