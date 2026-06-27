"""GraphRAG backend -- GraphRAG knowledge-graph + narrative retrieval backend.

A graph retrieval backend behind the RAG-store seam (`--retrieval-backend graph`), built by
REUSING the ontology-assisted drafting extraction (`DocExtraction` entities + SRO facts) as nodes/edges -- no second
extraction framework. Store = DuckDB (already a dependency; the abandoned Kuzu pick was dropped):
node/edge JSONL persistence loaded into an in-memory DuckDB engine that carries local k-hop
(recursive CTE) and community grouping (`WHERE community_id`). Two span-preserving strategies
share the one backend, recorded per run as `retrieval_strategy`:

  - local_khop        -- entity-link the question, expand k hops, serialize the subgraph.
  - global_community  -- map the question to communities (offline label propagation), serialize
                         each community's member nodes/edges WITH their offsets (the narrative
                         layer).

Both keep `doc_id` + char offsets so the source-span metric span metric still applies; an optional LLM community
summary is a tagged DIAGNOSTIC (`summary.summarize_communities`), never span-scored.
"""

from llb.graph.build import build_graph
from llb.graph.community import assign_communities, detect_communities
from llb.graph.constants import (
    BACKEND_GRAPH,
    STRATEGIES,
    STRATEGY_GLOBAL_COMMUNITY,
    STRATEGY_LOCAL_KHOP,
)
from llb.graph.model import GraphEdge, GraphMention, GraphNode, KnowledgeGraph
from llb.graph.store import GraphStore

__all__ = [
    "BACKEND_GRAPH",
    "STRATEGIES",
    "STRATEGY_GLOBAL_COMMUNITY",
    "STRATEGY_LOCAL_KHOP",
    "GraphEdge",
    "GraphMention",
    "GraphNode",
    "KnowledgeGraph",
    "GraphStore",
    "build_graph",
    "assign_communities",
    "detect_communities",
]
