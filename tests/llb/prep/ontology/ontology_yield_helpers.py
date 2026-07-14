"""draft-yield-quality-max: coverage targets, multi-hop paths, dedup, and question-type labels.

Every LLM/embedder call is an injected fake, so coverage-target sampling, 2-hop graph-path walking,
multi-span multi-hop grounding, near-duplicate suppression, and the per-question-type report are all
exercised deterministically with no server, key, or GPU.
"""

import json

from llb.goldset.schema import GoldItem, SourceSpan
from llb.graph.model import GraphEdge, GraphMention, GraphNode, KnowledgeGraph
from llb.prep.ontology.constants import (
    PROVENANCE_KIND,
)
from llb.prep.ontology.models import (
    DraftSeed,
)

CHAIN_DOC = "# Мережа\n\nAlpha керує Beta. Beta належить Gamma.\n"


def _span(doc_id: str, char_start: int, text: str) -> SourceSpan:
    return SourceSpan(
        doc_id=doc_id, char_start=char_start, char_end=char_start + len(text), text=text
    )


def _relation_pool(n: int) -> list[DraftSeed]:
    """A pool of `n` seeds, each in a DISTINCT relation bucket (shared doc/section/difficulty)."""
    return [
        DraftSeed(
            doc_id="a.md",
            kind="fact",
            section_title="S",
            difficulty="medium",
            strata={"doc": "a.md", "section": "S", "difficulty": "medium", "relation": f"r{i}"},
            evidence=_span("a.md", 0, "abc"),
        )
        for i in range(n)
    ]


# --- coverage-target sampling + exhaustion report --------------------------------------------


# --- question-type taxonomy ------------------------------------------------------------------


# --- multi-hop graph-path seeds --------------------------------------------------------------


def _chain_graph() -> KnowledgeGraph:
    e1 = CHAIN_DOC.index("Alpha керує Beta")
    e2 = CHAIN_DOC.index("Beta належить Gamma")
    m1: GraphMention = {
        "doc_id": "chain.md",
        "char_start": e1,
        "char_end": e1 + len("Alpha керує Beta"),
        "text": "Alpha керує Beta",
        "section_title": "Мережа",
    }
    m2: GraphMention = {
        "doc_id": "chain.md",
        "char_start": e2,
        "char_end": e2 + len("Beta належить Gamma"),
        "text": "Beta належить Gamma",
        "section_title": "Мережа",
    }
    nodes = [
        GraphNode(node_id=0, name="Alpha", type="ORG", confidence=1.0),
        GraphNode(node_id=1, name="Beta", type="ORG", confidence=1.0),
        GraphNode(node_id=2, name="Gamma", type="ORG", confidence=1.0),
    ]
    edges = [
        GraphEdge(edge_id=0, src=0, dst=1, relation="керує", evidence=m1),
        GraphEdge(edge_id=1, src=1, dst=2, relation="належить", evidence=m2),
    ]
    return KnowledgeGraph(nodes=nodes, edges=edges)


# --- near-duplicate suppression --------------------------------------------------------------


class FakeEmbedder:
    """Deterministic per-text unit vectors: identical text -> identical vector (cos 1),
    distinct text -> orthogonal vector (cos 0)."""

    def __init__(self) -> None:
        self._basis: dict[str, int] = {}

    def _index(self, text: str) -> int:
        return self._basis.setdefault(text, len(self._basis))

    def embed(self, texts: list[str]) -> list[list[float]]:
        indices = [self._index(text) for text in texts]
        width = len(self._basis)
        return [[1.0 if i == idx else 0.0 for i in range(width)] for idx in indices]


def _item(item_id: str, question: str) -> GoldItem:
    return GoldItem(
        id=item_id,
        question=question,
        reference_answer="x",
        source_doc_id="a.md",
        source_spans=[_span("a.md", 0, "abc")],
        provenance=PROVENANCE_KIND,
        split="final",
    )


# --- refine labels + per-question-type retrieval fraction ------------------------------------


class FakeNeedleRetriever:
    def __init__(self, hits: dict[str, list[dict[str, object]]]):
        self._hits = hits

    def retrieve(self, question: str, k: int) -> list[dict[str, object]]:
        return self._hits.get(question, [])[:k]


# --- full flow: multi-hop + coverage-target + dedup over a fake endpoint ----------------------


def _chain_extraction_json() -> str:
    return json.dumps(
        {
            "entities": [
                {"name": "Alpha", "type": "ORG", "mentions": ["Alpha"]},
                {"name": "Beta", "type": "ORG", "mentions": ["Beta"]},
                {"name": "Gamma", "type": "ORG", "mentions": ["Gamma"]},
            ],
            "facts": [
                {
                    "subject": "Alpha",
                    "relation": "керує",
                    "object": "Beta",
                    "evidence": "Alpha керує Beta",
                },
                {
                    "subject": "Beta",
                    "relation": "належить",
                    "object": "Gamma",
                    "evidence": "Beta належить Gamma",
                },
            ],
        }
    )


def _chain_endpoint(prompt: str) -> str:
    if "будує онтологію" in prompt:
        return _chain_extraction_json()
    if "багатокрокових (multi-hop)" in prompt:
        return json.dumps(
            {
                "question": "Кому належить компанія, якою керує Alpha?",
                "reference_answer": "Кінцевою організацією є Gamma.",
            }
        )
    if "укладач набору запитань" in prompt:
        return json.dumps(
            {
                "question": "Яка організація згадана поряд з Alpha?",
                "reference_answer": "Згаданою організацією є Beta.",
                "answer_span": "Beta",
            }
        )
    return "{}"
