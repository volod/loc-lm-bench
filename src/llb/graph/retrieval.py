"""Question linking + span-preserving serialization for the GraphRAG strategies (GraphRAG backend).

Pure, dependency-free (no DuckDB, no embedder): the graph STORE owns persistence and the graph
queries (k-hop via recursive CTE, community grouping via `WHERE community_id`); this module owns
"what is relevant to the question" (lexical entity linking) and "render a node/edge set back to
offset-bearing context". Both retrieval strategies serialize node MENTIONS and edge EVIDENCE with
their exact `doc_id` + char offsets, so the result scores on the SAME source-span metric (source-span metric)
the FAISS path uses -- the un-grounded abstraction (an LLM community summary) is kept out of here
entirely and recorded only as a tagged diagnostic.

Node relevance is a coarse LEVEL -- a hop distance, a link score, a community-member floor -- and
every mention of a node inherits it, so the rank-k cut routinely falls inside a block of spans the
lane scored identically and the cut is then settled by a document id. Within a level the spans are
therefore ordered by their own question affinity (`llb.graph.span_affinity`), placed strictly
inside the gap down to the next distinct level: the refinement reorders only spans the lane had
NOT scored apart, and the score it publishes is the score it ranked on, so a measurement floor
reads the real ranking.
"""

from typing import NamedTuple

from llb.core.contracts.rag import ChunkRecord
from llb.graph.constants import (
    KIND_EDGE_FACT,
    KIND_NODE_MENTION,
    SPAN_AFFINITY_BAND,
)
from llb.graph.model import GraphMention, KnowledgeGraph
from llb.graph.span_affinity import QuestionKeys, span_affinity

# The lowest relevance level has no level below it to band against, so its affinity is spread over
# the gap down to zero -- which no serialized candidate can reach, every member scoring above it.
_NO_LOWER_LEVEL = 0.0


class _Candidate(NamedTuple):
    """One emitted span before its level-internal affinity refinement decides its final score."""

    relevance: float
    mention: GraphMention
    chunk_id: str
    kind: str
    meta: dict[str, object]


def serialize_subgraph(
    graph: KnowledgeGraph, node_relevance: dict[int, float], k: int, question: str = ""
) -> list[ChunkRecord]:
    """Render the member nodes/edges to ranked, deduplicated, offset-bearing context chunks.

    `node_relevance` maps each MEMBER node id to its relevance (seed proximity for local_khop,
    question link score for global_community). Node mentions and the evidence of edges whose BOTH
    endpoints are members are emitted, ranked by relevance, deduplicated by exact span, capped at
    `k`. Empty member set -> no context (the eval graph then records a retrieval_miss).

    `question` orders the spans WITHIN one relevance level by their own affinity to it; an empty
    question (no query text to key on) leaves every level's spans exactly tied, as before.
    """
    candidates = _candidates(graph, node_relevance)
    return _rank_dedup(_scored_records(candidates, QuestionKeys(question)), k)


def _candidates(graph: KnowledgeGraph, node_relevance: dict[int, float]) -> list[_Candidate]:
    """Every member node's mentions plus the evidence of every edge internal to the member set."""
    by_id = graph.node_by_id()
    members = set(node_relevance)
    candidates: list[_Candidate] = []
    for node_id in members:
        node = by_id[node_id]
        for i, mention in enumerate(node.mentions):
            candidates.append(
                _Candidate(
                    node_relevance[node_id],
                    mention,
                    f"node{node_id}:m{i}",
                    KIND_NODE_MENTION,
                    {
                        "node": node.name,
                        "node_type": node.type,
                        "confidence": node.confidence,
                        "community_id": node.community_id,
                    },
                )
            )
    for edge in graph.edges:
        if edge.src in members and edge.dst in members:
            candidates.append(
                _Candidate(
                    (node_relevance[edge.src] + node_relevance[edge.dst]) / 2.0,
                    edge.evidence,
                    f"edge{edge.edge_id}",
                    KIND_EDGE_FACT,
                    {"relation": edge.relation, "community_id": by_id[edge.src].community_id},
                )
            )
    return candidates


def _scored_records(
    candidates: list[_Candidate], question: QuestionKeys
) -> list[tuple[float, ChunkRecord]]:
    """Refine each span's score by its question affinity, strictly inside its relevance level.

    A span at relevance `r` is placed in `[r - SPAN_AFFINITY_BAND * (r - next_lower), r]`, so its
    whole band sits above the next distinct level: the refinement can reorder spans the lane scored
    identically and nothing else. The score is published unrounded, because rounding a published
    score is itself a way to manufacture the ties this term exists to break.

    A question with no content tokens carries no signal to refine by, so no band is opened at all
    and every span keeps its level's relevance exactly.
    """
    below = _next_lower_level({candidate.relevance for candidate in candidates})
    scored: list[tuple[float, ChunkRecord]] = []
    for candidate in candidates:
        band = SPAN_AFFINITY_BAND * (candidate.relevance - below[candidate.relevance])
        affinity = span_affinity(question, candidate.mention) if question.size else 1.0
        score = candidate.relevance - band * (1.0 - affinity)
        scored.append((score, _record(candidate, score)))
    return scored


def _next_lower_level(levels: set[float]) -> dict[float, float]:
    """Each distinct relevance level mapped to the next distinct level below it."""
    ordered = sorted(levels)
    return {level: (ordered[i - 1] if i else _NO_LOWER_LEVEL) for i, level in enumerate(ordered)}


def _record(candidate: _Candidate, score: float) -> ChunkRecord:
    mention = candidate.mention
    record: ChunkRecord = {
        "doc_id": mention["doc_id"],
        "char_start": mention["char_start"],
        "char_end": mention["char_end"],
        "text": mention["text"],
        "chunk_id": candidate.chunk_id,
        "retrieval_score": float(score),
        "metadata": {
            "kind": candidate.kind,
            "section_title": mention["section_title"],
            **candidate.meta,
        },
    }
    return record


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
