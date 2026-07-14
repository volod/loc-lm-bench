"""Focused linking implementation."""

import re

from llb.graph.constants import (
    CONFIDENCE_TIE_BOOST,
    MIN_STEM_LEN,
    STEM_MATCH_WEIGHT,
)

from llb.graph.model import KnowledgeGraph

_TOKEN = re.compile(r"\w+", re.UNICODE)

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
