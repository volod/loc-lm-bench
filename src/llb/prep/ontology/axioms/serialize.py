"""Render an `AxiomSet` as RDFS/OWL Turtle.

The emitted document has three parts a reader can skim in order: the closed entity-type
vocabulary as `owl:Class` declarations, the relations the axioms constrain as
`owl:ObjectProperty` declarations carrying their Ukrainian surface as `rdfs:label`, and then one
block per axiom -- the standard OWL constraint followed by its `owl:Axiom` annotation (id, gloss,
and the sign-off, when a reviewer has given one).
"""

from llb.prep.ontology.axioms.constants import (
    DISJOINT_TYPES,
    DOMAIN,
    MAX_CARDINALITY,
    RANGE,
)
from llb.prep.ontology.axioms.models import Axiom, AxiomSet
from llb.prep.ontology.axioms.turtle import escape_literal
from llb.prep.ontology.axioms.vocab import (
    NS,
    CHARACTERISTIC_BY_KIND,
    ONTOLOGY_IRI,
    PREFIXES,
    RDFS_DOMAIN,
    RDFS_RANGE,
    RDFS_SUBCLASS_OF,
    RDF_TYPE,
    RELATION_PREFIX,
    OWL_DISJOINT_WITH,
    OWL_THING,
    compact,
    slug,
    type_iri,
)
from llb.prep.ontology.axioms.vocab import XSD_NON_NEGATIVE_INTEGER
from llb.prep.ontology.extraction.entity_types import ENTITY_TYPE_GLOSSES

LANG_UK = "@uk"


def _literal(value: str, *, lang: str = "") -> str:
    return f'"{escape_literal(value)}"{lang}'


class RelationNames:
    """Assigns each relation surface a unique ASCII IRI local name, deterministically."""

    def __init__(self, relations: list[str]) -> None:
        self._iri: dict[str, str] = {}
        used: set[str] = set()
        for relation in relations:
            base = RELATION_PREFIX + slug(relation)
            name, index = base, 1
            while name in used:
                index += 1
                name = f"{base}_{index}"
            used.add(name)
            self._iri[relation] = name

    def iri(self, relation: str) -> str:
        """The `llb:`-prefixed name for a relation surface."""
        return f"llb:{self._iri[relation]}"

    def full(self, relation: str) -> str:
        """The absolute IRI for a relation surface (what an RDF library needs)."""
        return NS + self._iri[relation]

    def items(self) -> list[tuple[str, str]]:
        """(surface, prefixed name) pairs in declaration order."""
        return [(relation, f"llb:{name}") for relation, name in self._iri.items()]


def relation_names(axiom_set: AxiomSet) -> RelationNames:
    """The relation IRI assignment for a whole set -- shared by the writer and the cross-check."""
    ordered: list[str] = []
    for axiom in axiom_set.axioms:
        if axiom.relation and axiom.relation not in ordered:
            ordered.append(axiom.relation)
    return RelationNames(ordered)


def _annotation(axiom: Axiom, source: str, predicate: str, target: str) -> list[str]:
    """The standard `owl:Axiom` block carrying identity, gloss, and sign-off."""
    lines = [
        "[] a owl:Axiom ;",
        f"  owl:annotatedSource {source} ;",
        f"  owl:annotatedProperty {compact(predicate)} ;",
        f"  owl:annotatedTarget {target} ;",
        f"  rdfs:label {_literal(axiom.axiom_id)}",
    ]
    if axiom.gloss:
        lines.append(f"  ; rdfs:comment {_literal(axiom.gloss, lang=LANG_UK)}")
    if axiom.signed_by:
        lines.append(f"  ; dcterms:creator {_literal(axiom.signed_by)}")
    if axiom.signed_on:
        lines.append(f"  ; dcterms:date {_literal(axiom.signed_on)}")
    lines.append("  .")
    return lines


def _type_target(axiom: Axiom, node: str) -> tuple[str, list[str]]:
    """The domain/range target: a bare class for one type, a named union node for several."""
    if len(axiom.entity_types) == 1:
        return compact(type_iri(axiom.entity_types[0])), []
    members = " ".join(compact(type_iri(t)) for t in axiom.entity_types)
    return node, [f"{node} a owl:Class ;", f"  owl:unionOf ( {members} ) .", ""]


def _axiom_lines(axiom: Axiom, relations: RelationNames) -> list[str]:
    """The constraint triples plus the annotation block for one axiom."""
    node = f"_:{slug(axiom.axiom_id)}"
    if axiom.kind in CHARACTERISTIC_BY_KIND:
        subject = relations.iri(axiom.relation or "")
        target = compact(CHARACTERISTIC_BY_KIND[axiom.kind])
        return [f"{subject} a {target} .", *_annotation(axiom, subject, RDF_TYPE, target)]
    if axiom.kind in (DOMAIN, RANGE):
        subject = relations.iri(axiom.relation or "")
        predicate = RDFS_DOMAIN if axiom.kind == DOMAIN else RDFS_RANGE
        target, extra = _type_target(axiom, node)
        return [
            f"{subject} {compact(predicate)} {target} .",
            *extra,
            *_annotation(axiom, subject, predicate, target),
        ]
    if axiom.kind == DISJOINT_TYPES:
        left, right = (compact(type_iri(t)) for t in axiom.entity_types)
        return [
            f"{left} owl:disjointWith {right} .",
            *_annotation(axiom, left, OWL_DISJOINT_WITH, right),
        ]
    assert axiom.kind == MAX_CARDINALITY  # the only remaining class; validated on the model
    bound = f'"{axiom.max_count}"^^{compact(XSD_NON_NEGATIVE_INTEGER)}'
    return [
        f"{compact(OWL_THING)} rdfs:subClassOf {node} .",
        f"{node} a owl:Restriction ;",
        f"  owl:onProperty {relations.iri(axiom.relation or '')} ;",
        f"  owl:maxCardinality {bound} .",
        "",
        *_annotation(axiom, compact(OWL_THING), RDFS_SUBCLASS_OF, node),
    ]


def axiom_turtle(axiom: Axiom) -> str:
    """One axiom rendered alone -- what the sign-off worksheet shows beside the gloss."""
    return "\n".join(_axiom_lines(axiom, RelationNames([axiom.relation or ""])))


def dump_turtle(axiom_set: AxiomSet, header: list[str] | None = None) -> str:
    """The whole constraint set as a Turtle document, with an optional comment header."""
    relations = relation_names(axiom_set)
    lines = [f"# {line}" if line else "#" for line in header or []]
    lines += [f"@prefix {prefix}: <{namespace}> ." for prefix, namespace in PREFIXES]
    lines += [
        "",
        f"<{ONTOLOGY_IRI}> a owl:Ontology ;",
        f"  owl:versionInfo {_literal(axiom_set.version)} .",
        "",
        "# --- the closed entity-type vocabulary (entity_types.py) ---",
    ]
    lines += [
        f"llb:{name} a owl:Class ; rdfs:label {_literal(name)} ; "
        f"rdfs:comment {_literal(gloss, lang=LANG_UK)} ."
        for name, gloss in ENTITY_TYPE_GLOSSES
    ]
    lines += ["", "# --- the relations these axioms constrain ---"]
    lines += [
        f"{iri} a owl:ObjectProperty ; rdfs:label {_literal(surface, lang=LANG_UK)} ."
        for surface, iri in relations.items()
    ]
    for axiom in axiom_set.axioms:
        lines += ["", f"# {axiom.kind}: {axiom.axiom_id}", *_axiom_lines(axiom, relations)]
    return "\n".join(lines) + "\n"
