"""Question linking + span-preserving serialization for the GraphRAG strategies (GraphRAG backend).

Pure, dependency-free (no DuckDB, no embedder): the graph STORE owns persistence and the graph
queries (k-hop via recursive CTE, community grouping via `WHERE community_id`); this module owns
"what is relevant to the question" (lexical entity linking) and "render a node/edge set back to
offset-bearing context". Both retrieval strategies serialize node MENTIONS and edge EVIDENCE with
their exact `doc_id` + char offsets, so the result scores on the SAME source-span metric (source-span metric)
the FAISS path uses -- the un-grounded abstraction (an LLM community summary) is kept out of here
entirely and recorded only as a tagged diagnostic.
"""

import re

from llb.contracts import ChunkRecord
from llb.graph.constants import (
    CONFIDENCE_TIE_BOOST,
    KIND_EDGE_FACT,
    KIND_NODE_MENTION,
    MIN_STEM_LEN,
    STEM_MATCH_WEIGHT,
)
from llb.graph.model import GraphMention, KnowledgeGraph

_TOKEN = re.compile(r"\w+", re.UNICODE)

# Small UA/EN stopword set so question linking keys on content words, not function words.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "what",
        "which",
        "who",
        "whom",
        "where",
        "when",
        "how",
        "why",
        "did",
        "do",
        "does",
        "що",
        "як",
        "хто",
        "де",
        "коли",
        "чому",
        "який",
        "яка",
        "яке",
        "які",
        "це",
        "та",
        "і",
        "й",
        "у",
        "в",
        "на",
        "з",
        "до",
        "за",
        "по",
        "про",
        "the",
        "for",
        "with",
    }
)


def tokenize(text: str) -> set[str]:
    """Content tokens of `text` (casefolded, >=2 chars, stopwords dropped)."""
    return {
        tok
        for raw in _TOKEN.findall(text)
        if len(tok := raw.casefold()) >= 2 and tok not in _STOPWORDS
    }


def morph_key(token: str) -> str:
    """Morphological linking key: the leading `MIN_STEM_LEN` chars of a token.

    A Ukrainian name and its inflected forms differ only in the ENDING, so collapsing each token
    to its leading stem lets an inflected question form (e.g. "Франка" genitive) still link the
    "Франко" node. Tokens shorter than the stem length key on themselves (they only match exactly).
    This is the dependency-free morphology seam -- no lemmatizer, no embedder.
    """
    return token[:MIN_STEM_LEN] if len(token) >= MIN_STEM_LEN else token


def _node_surface_tokens(graph: KnowledgeGraph) -> dict[int, set[str]]:
    """Per-node bag of tokens drawn from its NAME + aliases.

    Entity linking keys on the entity's own surface forms, not the surrounding mention text:
    a mention span can carry neighboring relation verbs ("X wrote Y"), and matching those would
    wrongly link unrelated entities that share a common verb.
    """
    surface: dict[int, set[str]] = {}
    for node in graph.nodes:
        toks = tokenize(node.name)
        for alias in node.aliases:
            toks |= tokenize(alias)
        surface[node.node_id] = toks
    return surface


def node_link_scores(graph: KnowledgeGraph, question: str) -> dict[int, float]:
    """Lexical link score per node: question tokens it covers, with confidence as a tiny boost.

    Exact token overlap scores full weight; a morphological (shared-stem) match that is NOT already
    an exact hit scores `STEM_MATCH_WEIGHT` (so an inflected question form still links the node, but
    exact hits always rank first). A node with neither kind of overlap is omitted (not linked).
    """
    q_tokens = tokenize(question)
    if not q_tokens:
        return {}
    q_keys = {morph_key(tok) for tok in q_tokens}
    surface = _node_surface_tokens(graph)
    confidence = {n.node_id: n.confidence for n in graph.nodes}
    scores: dict[int, float] = {}
    for node_id, toks in surface.items():
        exact = len(q_tokens & toks)
        # surface stems matched by a question stem, beyond the ones already counted as exact hits
        stem_only = max(len(q_keys & {morph_key(tok) for tok in toks}) - exact, 0)
        if exact or stem_only:
            scores[node_id] = (
                exact + STEM_MATCH_WEIGHT * stem_only + CONFIDENCE_TIE_BOOST * confidence[node_id]
            )
    return scores


def link_seed_nodes(graph: KnowledgeGraph, question: str, n_seeds: int) -> list[int]:
    """Top-`n_seeds` question-linked node ids (the entity-link step of local_khop)."""
    scores = node_link_scores(graph, question)
    ranked = sorted(scores, key=lambda nid: (-scores[nid], nid))
    return ranked[:n_seeds]


def link_communities(graph: KnowledgeGraph, question: str, n_communities: int) -> list[int]:
    """Top-`n_communities` community ids, scored by their members' total question link score."""
    scores = node_link_scores(graph, question)
    if not scores:
        return []
    by_node = {n.node_id: n for n in graph.nodes}
    community_score: dict[int, float] = {}
    for node_id, score in scores.items():
        cid = by_node[node_id].community_id
        community_score[cid] = community_score.get(cid, 0.0) + score
    ranked = sorted(community_score, key=lambda cid: (-community_score[cid], cid))
    return ranked[:n_communities]


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
