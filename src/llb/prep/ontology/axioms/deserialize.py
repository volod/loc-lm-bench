"""Read a Turtle axiom file back into the typed `AxiomSet` the checker runs on.

Reading is driven by the `owl:Axiom` annotation blocks: each one names the constraint triple it
annotates (`owl:annotatedSource/Property/Target`), so one block is one axiom, and the axiom id,
the Ukrainian gloss, and the sign-off travel with it. A constraint triple with no annotation block
is IGNORED -- an axiom with no id is one nobody can accept, reject, or cite in a violation.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from llb.prep.ontology.axioms.constants import (
    DISJOINT_TYPES,
    DOMAIN,
    MAX_CARDINALITY,
    RANGE,
)
from llb.prep.ontology.axioms.models import Axiom, AxiomSet
from llb.prep.ontology.axioms.turtle import (
    Bnode,
    Iri,
    Literal,
    Term,
    Triple,
    collection_items,
    parse_turtle,
)
from llb.prep.ontology.axioms.vocab import (
    DCTERMS_CREATOR,
    OWL_ANNOTATED_PROPERTY,
    OWL_ANNOTATED_SOURCE,
    OWL_ANNOTATED_TARGET,
    DCTERMS_DATE,
    KIND_BY_CHARACTERISTIC,
    KIND_BY_PREDICATE,
    OWL_AXIOM,
    OWL_MAX_CARDINALITY,
    OWL_ON_PROPERTY,
    OWL_ONTOLOGY,
    OWL_OBJECT_PROPERTY,
    OWL_UNION_OF,
    OWL_VERSION_INFO,
    RDFS_COMMENT,
    RDFS_LABEL,
    RDF_TYPE,
    type_name,
)

DEFAULT_VERSION = "unversioned"


class AxiomFileError(ValueError):
    """The Turtle parses, but it does not describe an axiom set this checker can run."""


class _Store:
    """Triple lookups by subject and by (subject, predicate)."""

    def __init__(self, triples: list[Triple]) -> None:
        self.triples = triples
        self._by_sp: dict[tuple[Term, str], list[Term]] = defaultdict(list)
        for subject, predicate, obj in triples:
            self._by_sp[(subject, predicate.value)].append(obj)

    def objects(self, subject: Term, predicate: str) -> list[Term]:
        return self._by_sp[(subject, predicate)]

    def one(self, subject: Term, predicate: str) -> Term | None:
        found = self.objects(subject, predicate)
        return found[0] if found else None

    def text(self, subject: Term, predicate: str) -> str:
        found = self.one(subject, predicate)
        return found.value if isinstance(found, Literal) else ""

    def subjects_typed(self, class_iri: str) -> list[Term]:
        return [s for s, p, o in self.triples if p.value == RDF_TYPE and o == Iri(class_iri)]


def _version(store: _Store) -> str:
    for subject in store.subjects_typed(OWL_ONTOLOGY):
        if version := store.text(subject, OWL_VERSION_INFO):
            return version
    return DEFAULT_VERSION


def _relation_labels(store: _Store) -> dict[Term, str]:
    return {
        subject: store.text(subject, RDFS_LABEL)
        for subject in store.subjects_typed(OWL_OBJECT_PROPERTY)
    }


def _types_of(store: _Store, target: Term) -> list[str]:
    """The entity types a domain/range target names: one class, or a `owl:unionOf` list."""
    if isinstance(target, Iri):
        return [type_name(target.value)]
    union = store.one(target, OWL_UNION_OF)
    if union is None:
        raise AxiomFileError(f"domain/range target {target!r} is neither a class nor a union")
    members = collection_items(store.triples, union)
    return [type_name(item.value) for item in members if isinstance(item, Iri)]


def _relation_of(store: _Store, term: Term, labels: dict[Term, str]) -> str:
    relation = labels.get(term, "")
    if not relation:
        raise AxiomFileError(f"no rdfs:label for the relation {term!r}; declare it as a property")
    return relation


def _kind(predicate: str, target: Term) -> str:
    if predicate == RDF_TYPE:
        if not isinstance(target, Iri) or target.value not in KIND_BY_CHARACTERISTIC:
            raise AxiomFileError(f"annotated rdf:type target {target!r} is not an axiom class")
        return KIND_BY_CHARACTERISTIC[target.value]
    if predicate not in KIND_BY_PREDICATE:
        raise AxiomFileError(f"annotated predicate {predicate!r} expresses no axiom class")
    return KIND_BY_PREDICATE[predicate]


@dataclass(frozen=True)
class _Payload:
    """The kind-specific fields of one axiom, read off its annotated constraint triple."""

    relation: str | None = None
    entity_types: list[str] = field(default_factory=list)
    max_count: int | None = None


def _payload(
    store: _Store, kind: str, source: Term, target: Term, labels: dict[Term, str]
) -> _Payload:
    """The kind-specific fields (`relation`, `entity_types`, `max_count`) of one axiom."""
    if kind in (DOMAIN, RANGE):
        return _Payload(
            relation=_relation_of(store, source, labels), entity_types=_types_of(store, target)
        )
    if kind == DISJOINT_TYPES:
        pair = [t for t in (source, target) if isinstance(t, Iri)]
        return _Payload(entity_types=[type_name(t.value) for t in pair])
    if kind == MAX_CARDINALITY:
        on_property = store.one(target, OWL_ON_PROPERTY)
        bound = store.text(target, OWL_MAX_CARDINALITY)
        if on_property is None or not bound.isdigit():
            raise AxiomFileError(
                f"restriction {target!r} needs owl:onProperty + owl:maxCardinality"
            )
        return _Payload(relation=_relation_of(store, on_property, labels), max_count=int(bound))
    return _Payload(relation=_relation_of(store, source, labels))


def _axiom(store: _Store, node: Term, labels: dict[Term, str]) -> Axiom:
    source = store.one(node, OWL_ANNOTATED_SOURCE)
    predicate = store.one(node, OWL_ANNOTATED_PROPERTY)
    target = store.one(node, OWL_ANNOTATED_TARGET)
    label = store.text(node, RDFS_LABEL)
    if source is None or not isinstance(predicate, Iri) or target is None or not label:
        raise AxiomFileError(f"owl:Axiom block {node!r} is missing source/property/target/label")
    kind = _kind(predicate.value, target)
    payload = _payload(store, kind, source, target, labels)
    return Axiom(
        axiom_id=label,
        kind=kind,
        gloss=store.text(node, RDFS_COMMENT),
        signed_by=store.text(node, DCTERMS_CREATOR) or None,
        signed_on=store.text(node, DCTERMS_DATE) or None,
        relation=payload.relation,
        entity_types=payload.entity_types,
        max_count=payload.max_count,
    )


def load_turtle(text: str) -> AxiomSet:
    """Parse a Turtle axiom document into the typed set the checker runs on."""
    _prefixes, triples = parse_turtle(text)
    store = _Store(triples)
    labels = _relation_labels(store)
    blocks = [node for node in store.subjects_typed(OWL_AXIOM) if isinstance(node, Bnode)]
    return AxiomSet(
        version=_version(store),
        axioms=[_axiom(store, node, labels) for node in blocks],
    )
