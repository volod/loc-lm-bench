"""A planted graph whose correct node clustering is known, plus a deterministic node embedder.

The plant is the fragmentation the pass exists for, written out on purpose: each true entity is
several nodes -- a full name, a shorter form, an inflected form, an initialism -- carrying the
alias overlap, the document co-occurrence, and the shared mention subject a real fragmented entity
carries. It deliberately does NOT plant a form nothing else mentions (an epithet on its own node,
agreeing with its entity on nothing but the type and the document): resolving that is coreference,
which this pass is explicitly not.

The distractors are the hard half -- same type, same document, names sharing a leading word, and a
different entity every one of them -- so a run that merges everything fails the fixture just as
loudly as one that merges nothing.
"""

import hashlib
import math

import pytest

from llb.graph.model import GraphEdge, GraphMention, GraphNode, KnowledgeGraph

# (canonical name, its other surface forms, entity type, document, what its mentions are about).
PLANTED_ENTITIES = (
    ("Іван Франко", ("Франко", "Франка"), "PERSON", "doc-a", "українська поезія"),
    ("Леся Українка", ("Українка", "Українки"), "PERSON", "doc-a", "драматургія"),
    ("Тарас Шевченко", ("Шевченко", "Шевченка"), "PERSON", "doc-b", "народна пісня"),
    ("Збройні Сили України", ("ЗСУ",), "ORG", "doc-b", "оборона держави"),
    ("Міністерство оборони України", ("МОУ",), "ORG", "doc-c", "військове управління"),
    ("Львівський національний університет", ("ЛНУ",), "ORG", "doc-c", "вища освіта"),
)

# (name, entity type, document, what it is about).
PLANTED_DISTRACTORS = (
    ("Додаток 27", "WORK", "doc-a", "форма звіту"),
    ("Додаток 57", "WORK", "doc-a", "форма накладної"),
    ("Додаток 26", "WORK", "doc-b", "форма акта"),
    ("Акт списання запасів", "WORK", "doc-b", "списання майна"),
    ("Акт приймання-передачі", "WORK", "doc-c", "передача майна"),
    ("Наказ Міністра оборони", "LAW", "doc-c", "нормативний документ"),
    ("Наказ Генерального штабу", "LAW", "doc-a", "бойова підготовка"),
    ("Положення про облік", "LAW", "doc-b", "бухгалтерський облік"),
    ("Порядок списання", "LAW", "doc-c", "рух матеріальних засобів"),
    ("Категорія якості", "MISC", "doc-a", "технічний стан обладнання"),
    ("Діапазон від 1 до 5", "MISC", "doc-b", "шкала оцінювання"),
    ("Технічний стан", "MISC", "doc-c", "придатність обладнання"),
    ("Інвентарний номер", "MISC", "doc-a", "ідентифікація предмета"),
    ("Матеріальна відповідальність", "MISC", "doc-b", "відповідальна особа"),
    ("Військова частина", "MISC", "doc-c", "організаційна структура"),
    ("Особовий склад", "MISC", "doc-a", "штатна чисельність"),
)


def _mention(doc_id: str, offset: int, text: str) -> GraphMention:
    return {
        "doc_id": doc_id,
        "char_start": offset,
        "char_end": offset + len(text),
        "text": text,
        "section_title": "Розділ 1",
    }


class PlantedGraph:
    """The planted graph plus the truth a test reads it against."""

    def __init__(self) -> None:
        self.graph = KnowledgeGraph()
        self.truth: dict[int, str] = {}  # node id -> the true entity it belongs to
        self._offset = 0
        self._plant()

    def _add(self, name: str, etype: str, doc: str, aliases, topic: str, truth: str) -> GraphNode:
        # The subject is repeated so the mention embedding is dominated by what the mention is
        # ABOUT rather than by the surface form, which is the signal the cosine ladder prices.
        text = f"{name} -- {topic}. Тема: {topic}."
        node = GraphNode(
            node_id=len(self.graph.nodes),
            name=name,
            type=etype,
            confidence=0.9,
            aliases=list(aliases),
            mentions=[_mention(doc, self._offset, text)],
        )
        self._offset += len(text) + 1
        self.graph.nodes.append(node)
        self.truth[node.node_id] = truth
        return node

    def _plant(self) -> None:
        for canonical, forms, etype, doc, topic in PLANTED_ENTITIES:
            # The canonical node keeps every surface form as an alias and each fragment keeps the
            # canonical as its own -- what the builder produces when one extraction names an
            # entity in full and another names it by a form the normalized key does not match.
            self._add(canonical, etype, doc, forms, topic, canonical)
            for form in forms:
                self._add(form, etype, doc, [canonical], topic, canonical)
        for name, etype, doc, topic in PLANTED_DISTRACTORS:
            self._add(name, etype, doc, [], topic, name)
        # Edges connect DISTRACTORS only. A fragmented entity's own pieces are exactly what the
        # graph does not know are related -- if an edge already joined them, a k-hop expansion
        # would reach the sibling's mentions without any merge, and the fixture would be measuring
        # the expansion rather than the resolution.
        distractors = [node for node in self.graph.nodes if self.truth[node.node_id] == node.name]
        for left, right in zip(distractors, distractors[1:]):
            self.graph.edges.append(
                GraphEdge(
                    edge_id=len(self.graph.edges),
                    src=left.node_id,
                    dst=right.node_id,
                    relation="згадується_з",
                    evidence=left.mentions[0],
                )
            )

    @property
    def truth_groups(self) -> dict[str, set[int]]:
        groups: dict[str, set[int]] = {}
        for node_id, entity in self.truth.items():
            groups.setdefault(entity, set()).add(node_id)
        return groups

    @property
    def fragmented_groups(self) -> dict[str, set[int]]:
        """The true entities that actually fragmented -- the merges a run has to find."""
        return {name: ids for name, ids in self.truth_groups.items() if len(ids) > 1}


class HashedNodeEmbedder:
    """A deterministic character-trigram embedder: no model, no GPU, stable across hosts.

    Two fragments of one entity share the subject their mentions are about, so their vectors are
    close; two distractors sharing only a leading word are not. That is enough for the cosine
    ladder to carry real signal in a fixture run, and it is the same trick the drafting dedup
    tests use for questions.
    """

    dimension = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        folded = " ".join(text.split()).casefold()
        for index in range(max(len(folded) - 2, 1)):
            gram = folded[index : index + 3]
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest()
            values[int.from_bytes(digest, "big") % self.dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


@pytest.fixture
def planted() -> PlantedGraph:
    return PlantedGraph()


@pytest.fixture
def node_embedder() -> HashedNodeEmbedder:
    return HashedNodeEmbedder()
