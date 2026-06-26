"""Build the knowledge graph from the M4.4 extraction (Milestone 6, construction stage).

REUSES the M4.4 `DocExtraction` (entities + SRO facts) -- no second extraction framework. Entity
mentions become nodes; SRO facts become directed edges between the subject/object nodes. Each
mention and evidence span keeps its `doc_id` + char offsets + exact text, and is tagged with the
induced ontology `type`/`confidence` and the containing `section_title`. A fact whose subject or
object is not a known entity gets a lightweight fact-only node, so no grounded fact is dropped.

Pure + deterministic (sorted iteration), so the same corpus always yields the same graph; only
community ids are filled later (`assign_communities`).
"""

import logging

from llb.goldset.schema import SourceSpan
from llb.graph.model import (
    UNKNOWN_CONFIDENCE,
    GraphEdge,
    GraphMention,
    GraphNode,
    KnowledgeGraph,
)
from llb.prep.ontology.entity_types import DEFAULT_ENTITY_TYPE
from llb.prep.ontology.induce import induce_ontology
from llb.prep.ontology.inventory import section_at
from llb.prep.ontology.models import (
    DocExtraction,
    DocRecord,
    OntologyCandidate,
    Section,
)

_LOG = logging.getLogger(__name__)


def _norm(name: str) -> str:
    """Case/space-insensitive key used to link fact endpoints to entity nodes."""
    return " ".join(name.split()).casefold()


def _mention(span: SourceSpan, sections: list[Section]) -> GraphMention:
    return {
        "doc_id": span.doc_id,
        "char_start": span.char_start,
        "char_end": span.char_end,
        "text": span.text,
        "section_title": section_at(sections, span.char_start),
    }


class _GraphBuilder:
    """Accumulates nodes/edges while linking fact endpoints to entity nodes by normalized name."""

    def __init__(
        self,
        sections_by_doc: dict[str, list[Section]],
        type_confidence: dict[str, float],
    ) -> None:
        self._sections_by_doc = sections_by_doc
        self._type_confidence = type_confidence
        self._by_key: dict[str, GraphNode] = {}  # normalized name -> node
        self.nodes: list[GraphNode] = []
        self.edges: list[GraphEdge] = []

    def _sections(self, doc_id: str) -> list[Section]:
        return self._sections_by_doc.get(doc_id, [])

    def _ensure_node(self, name: str, etype: str, aliases: list[str]) -> GraphNode:
        key = _norm(name)
        node = self._by_key.get(key)
        if node is None:
            node = GraphNode(
                node_id=len(self.nodes),
                name=name,
                type=etype,
                confidence=self._type_confidence.get(etype, UNKNOWN_CONFIDENCE),
                aliases=list(aliases),
            )
            self._by_key[key] = node
            self.nodes.append(node)
            return node
        for alias in aliases:  # merge any new aliases onto the existing node
            if alias not in node.aliases:
                node.aliases.append(alias)
        return node

    def add_entity(
        self, name: str, etype: str, aliases: list[str], mentions: list[SourceSpan]
    ) -> None:
        node = self._ensure_node(name, etype, aliases)
        seen = {(m["char_start"], m["char_end"], m["doc_id"]) for m in node.mentions}
        for span in mentions:
            marker = (span.char_start, span.char_end, span.doc_id)
            if marker not in seen:
                seen.add(marker)
                node.mentions.append(_mention(span, self._sections(span.doc_id)))

    def add_fact(self, subject: str, relation: str, obj: str, evidence: SourceSpan) -> None:
        mention = _mention(evidence, self._sections(evidence.doc_id))
        src = self._ensure_node(subject, DEFAULT_ENTITY_TYPE, [])
        dst = self._ensure_node(obj, DEFAULT_ENTITY_TYPE, [])
        # a fact-only endpoint still needs grounding -> attach the evidence as a mention
        for node in (src, dst):
            if not node.mentions:
                node.mentions.append(mention)
        self.edges.append(
            GraphEdge(
                edge_id=len(self.edges),
                src=src.node_id,
                dst=dst.node_id,
                relation=relation,
                evidence=mention,
            )
        )

    def graph(self) -> KnowledgeGraph:
        return KnowledgeGraph(nodes=self.nodes, edges=self.edges)


def build_graph(
    extractions: list[DocExtraction],
    docs: list[DocRecord],
    ontology: OntologyCandidate | None = None,
) -> KnowledgeGraph:
    """Construct the knowledge graph from per-document extractions + their inventoried docs.

    `ontology` carries the induced type confidences onto the nodes; it is induced from the
    extractions when not supplied (the same constrained ontology the M4.4 pipeline produces).
    """
    induced = ontology if ontology is not None else induce_ontology(extractions)
    type_confidence = {t.name: t.confidence for t in induced.entity_types}
    sections_by_doc = {doc.doc_id: doc.sections for doc in docs}

    builder = _GraphBuilder(sections_by_doc, type_confidence)
    # entities first (so facts link onto typed entity nodes), deterministic order
    for extraction in extractions:
        for entity in extraction.entities:
            builder.add_entity(entity.name, entity.type, entity.aliases, entity.mentions)
    for extraction in extractions:
        for fact in extraction.facts:
            builder.add_fact(fact.subject, fact.relation, fact.object, fact.evidence)

    graph = builder.graph()
    _LOG.info(
        "[graph] built %d nodes, %d edges from %d documents",
        len(graph.nodes),
        len(graph.edges),
        len(docs),
    )
    return graph
