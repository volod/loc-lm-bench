"""Question linking + span-preserving serialization for the GraphRAG strategies (GraphRAG backend).

Pure, dependency-free (no DuckDB, no embedder): the graph STORE owns persistence and the graph
queries (k-hop via recursive CTE, community grouping via `WHERE community_id`); this module owns
"what is relevant to the question" (lexical entity linking) and "render a node/edge set back to
offset-bearing context". Both retrieval strategies serialize node MENTIONS and edge EVIDENCE with
their exact `doc_id` + char offsets, so the result scores on the SAME source-span metric (source-span metric)
the FAISS path uses -- the un-grounded abstraction (an LLM community summary) is kept out of here
entirely and recorded only as a tagged diagnostic.
"""

from llb.core.contracts.rag import ChunkRecord
from llb.graph.constants import (
    KIND_EDGE_FACT,
    KIND_NODE_MENTION,
)
from llb.graph.model import GraphMention, KnowledgeGraph


# Small UA/EN stopword set so question linking keys on content words, not function words.


def _record(
    mention: GraphMention, chunk_id: str, kind: str, score: float, **meta: object
) -> ChunkRecord:
    record: ChunkRecord = {
        "doc_id": mention["doc_id"],
        "char_start": mention["char_start"],
        "char_end": mention["char_end"],
        "text": mention["text"],
        "chunk_id": chunk_id,
        "retrieval_score": round(score, 4),
        "metadata": {"kind": kind, "section_title": mention["section_title"], **meta},
    }
    return record


def serialize_subgraph(
    graph: KnowledgeGraph, node_relevance: dict[int, float], k: int
) -> list[ChunkRecord]:
    """Render the member nodes/edges to ranked, deduplicated, offset-bearing context chunks.

    `node_relevance` maps each MEMBER node id to its relevance (seed proximity for local_khop,
    question link score for global_community). Node mentions and the evidence of edges whose BOTH
    endpoints are members are emitted, ranked by relevance, deduplicated by exact span, capped at
    `k`. Empty member set -> no context (the eval graph then records a retrieval_miss).
    """
    by_id = graph.node_by_id()
    members = set(node_relevance)
    scored: list[tuple[float, ChunkRecord]] = []

    for node_id in members:
        node = by_id[node_id]
        relevance = node_relevance[node_id]
        for i, mention in enumerate(node.mentions):
            scored.append(
                (
                    relevance,
                    _record(
                        mention,
                        f"node{node_id}:m{i}",
                        KIND_NODE_MENTION,
                        relevance,
                        node=node.name,
                        node_type=node.type,
                        confidence=node.confidence,
                        community_id=node.community_id,
                    ),
                )
            )
    for edge in graph.edges:
        if edge.src in members and edge.dst in members:
            relevance = (node_relevance[edge.src] + node_relevance[edge.dst]) / 2.0
            scored.append(
                (
                    relevance,
                    _record(
                        edge.evidence,
                        f"edge{edge.edge_id}",
                        KIND_EDGE_FACT,
                        relevance,
                        relation=edge.relation,
                        community_id=by_id[edge.src].community_id,
                    ),
                )
            )

    return _rank_dedup(scored, k)


def _rank_dedup(scored: list[tuple[float, ChunkRecord]], k: int) -> list[ChunkRecord]:
    """Sort by relevance (then a stable span key), drop duplicate spans, cap at k, assign ranks."""
    scored.sort(
        key=lambda sr: (
            -sr[0],
            sr[1]["doc_id"],
            sr[1]["char_start"],
            sr[1]["char_end"],
        )
    )
    out: list[ChunkRecord] = []
    seen: set[tuple[str, int, int]] = set()
    for _score, record in scored:
        marker = (record["doc_id"], record["char_start"], record["char_end"])
        if marker in seen:
            continue
        seen.add(marker)
        record["rank"] = len(out) + 1
        out.append(record)
        if len(out) >= k:
            break
    return out
