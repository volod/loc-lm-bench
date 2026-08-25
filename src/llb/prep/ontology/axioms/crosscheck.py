"""Hold the in-repo checker to OWL semantics with a real reasoner -- in CI, never in the answer path.

The shipped checker is pure Python because a scoring host must never need an RDF stack to build a
graph. That leaves one risk: the checker could be self-consistently WRONG about what an axiom
means. This module removes it. It writes the same axioms and the same ledger as RDF, computes the
OWL 2 RL closure with `owlrl`, and evaluates each rule's antecedent over the closure -- the same
condition the rule uses to declare the graph inconsistent. A disagreement is a bug in the in-repo
checker, never a reason to relax the comparison.

It covers the classes whose OWL reading IS an inconsistency condition (`REASONER_KINDS`). The
open-world classes are excluded, and excluded from the RDF graph too: an axiom the cross-check
does not compare would still change the closure of the ones it does.

Needs the optional `[ontology]` extra; without it the run records that the cross-check did not run
rather than silently passing.
"""

import logging
from itertools import combinations
from typing import TYPE_CHECKING

from llb.prep.ontology.axioms.constants import (
    ASYMMETRIC,
    DISJOINT_TYPES,
    FUNCTIONAL,
    INVERSE_FUNCTIONAL,
    IRREFLEXIVE,
    REASONER_KINDS,
    REASONER_MISSING,
)
from llb.prep.ontology.axioms.ledger import Ledger
from llb.prep.ontology.axioms.models import Axiom, AxiomSet, CrosscheckResult, ValidationReport
from llb.prep.ontology.axioms.keys import comparison_key, pair_key
from llb.prep.ontology.axioms.serialize import RelationNames, dump_turtle, relation_names
from llb.prep.ontology.axioms.vocab import NS, type_iri
from llb.prep.ontology.naming import normalize_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rdflib import Graph, URIRef

_LOG = logging.getLogger(__name__)
INDIVIDUAL_PREFIX = NS + "individual_"


def _reasoner_subset(axiom_set: AxiomSet) -> AxiomSet:
    """The axioms the cross-check compares -- and the only ones its RDF graph may carry."""
    return AxiomSet(
        version=axiom_set.version,
        axioms=[a for a in axiom_set.axioms if a.kind in REASONER_KINDS],
    )


class _Individuals:
    """Mints one IRI per folded endpoint name and remembers the way back."""

    def __init__(self) -> None:
        self._iri: dict[str, str] = {}
        self.name: dict[str, str] = {}

    def iri(self, raw: str) -> str:
        key = normalize_name(raw)
        if key not in self._iri:
            minted = f"{INDIVIDUAL_PREFIX}{len(self._iri)}"
            self._iri[key] = minted
            self.name[minted] = key
        return self._iri[key]


def _add_facts(
    graph: "Graph",
    ledger: Ledger,
    relations: RelationNames,
    constrained: set[str],
    inds: _Individuals,
) -> None:
    """Every ledger fact whose relation an axiom in the subset constrains, as one RDF triple."""
    from rdflib import URIRef

    for relation in sorted(constrained):
        predicate = URIRef(relations.full(relation))
        for fact in ledger.relation_facts(relation):
            graph.add((URIRef(inds.iri(fact.subject)), predicate, URIRef(inds.iri(fact.object))))


def _add_types(graph: "Graph", ledger: Ledger, types: set[str], inds: _Individuals) -> None:
    """Every ASSERTED type the subset's disjointness axioms can speak about."""
    from rdflib import URIRef
    from rdflib.namespace import RDF

    for _key, assertions in ledger.typed_names():
        for assertion in assertions:
            if assertion.entity_type in types:
                graph.add(
                    (
                        URIRef(inds.iri(assertion.name)),
                        RDF.type,
                        URIRef(type_iri(assertion.entity_type)),
                    )
                )


def _build_graph(subset: AxiomSet, ledgers: list[Ledger]) -> "tuple[Graph, _Individuals]":
    """Write the axioms and every ledger fact/type this subset can constrain as RDF."""
    from rdflib import Graph

    graph = Graph()
    graph.parse(data=dump_turtle(subset), format="turtle")
    relations = relation_names(subset)
    constrained = {a.relation for a in subset.axioms if a.relation}
    types = {t for a in subset.axioms for t in a.entity_types}
    individuals = _Individuals()
    for ledger in ledgers:
        _add_facts(graph, ledger, relations, constrained, individuals)
        _add_types(graph, ledger, types, individuals)
    return graph, individuals


def _close(graph: "Graph") -> None:
    """Materialize the OWL 2 RL closure in place."""
    from owlrl import DeductiveClosure, OWLRL_Semantics

    DeductiveClosure(OWLRL_Semantics).expand(graph)


def _cardinality_keys(
    graph: "Graph", axiom: Axiom, predicate: "URIRef", names: dict[str, str], invert: bool
) -> set[str]:
    """`prp-fp` / `prp-ifp`: two values the closure declares `owl:sameAs` under one anchor."""
    from rdflib.namespace import OWL

    grouped: dict[str, list[str]] = {}
    for subject, obj in graph.subject_objects(predicate):
        anchor, value = (obj, subject) if invert else (subject, obj)
        grouped.setdefault(str(anchor), []).append(str(value))
    keys: set[str] = set()
    for held, values in grouped.items():
        for left, right in combinations(sorted(set(values)), 2):
            if (_ref(left), OWL.sameAs, _ref(right)) not in graph:
                continue
            pairs = [_pair(held, left, names, invert), _pair(held, right, names, invert)]
            keys.add(comparison_key(axiom.axiom_id, pairs))
    return keys


def _name(names: dict[str, str], iri: str) -> str:
    """The ledger name behind a minted IRI; an unknown IRI stays itself so it surfaces loudly."""
    return names.get(iri, iri)


def _ref(iri: str) -> "URIRef":
    from rdflib import URIRef

    return URIRef(iri)


def _pair(anchor: str, value: str, names: dict[str, str], invert: bool) -> tuple[str, str]:
    left, right = (value, anchor) if invert else (anchor, value)
    return names.get(left, left), names.get(right, right)


def _direction_keys(
    graph: "Graph", axiom: Axiom, predicate: "URIRef", names: dict[str, str]
) -> set[str]:
    """`prp-irp` / `prp-asyp`: the self-loop, or the pair the closure holds in both directions."""
    edges = {(str(s), str(o)) for s, o in graph.subject_objects(predicate)}
    keys: set[str] = set()
    for src, dst in edges:
        if axiom.kind == IRREFLEXIVE and src == dst:
            keys.add(comparison_key(axiom.axiom_id, [(_name(names, src), _name(names, dst))]))
        if axiom.kind == ASYMMETRIC and src != dst and (dst, src) in edges:
            keys.add(
                comparison_key(axiom.axiom_id, [(names[src], names[dst]), (names[dst], names[src])])
            )
    return keys


def _disjoint_keys(graph: "Graph", axiom: Axiom, names: dict[str, str]) -> set[str]:
    """`cax-dw`: one individual the closure types with both members of a disjoint pair."""
    from rdflib.namespace import RDF

    left, right = (_ref(type_iri(t)) for t in axiom.entity_types)
    holders = {str(s) for s in graph.subjects(RDF.type, left)} & {
        str(s) for s in graph.subjects(RDF.type, right)
    }
    return {
        comparison_key(
            axiom.axiom_id,
            [
                (_name(names, iri), axiom.entity_types[0]),
                (_name(names, iri), axiom.entity_types[1]),
            ],
        )
        for iri in holders
    }


def _reasoner_keys(graph: "Graph", subset: AxiomSet, names: dict[str, str]) -> set[str]:
    relations = relation_names(subset)
    keys: set[str] = set()
    for axiom in subset.axioms:
        if axiom.kind == DISJOINT_TYPES:
            keys |= _disjoint_keys(graph, axiom, names)
            continue
        predicate = _ref(relations.full(axiom.relation or ""))
        if axiom.kind in (FUNCTIONAL, INVERSE_FUNCTIONAL):
            keys |= _cardinality_keys(
                graph, axiom, predicate, names, axiom.kind == INVERSE_FUNCTIONAL
            )
        else:
            keys |= _direction_keys(graph, axiom, predicate, names)
    return keys


def _checker_keys(report: ValidationReport, kinds: tuple[str, ...]) -> set[str]:
    return {
        pair_key(violation)
        for ledger in report.ledgers
        for violation in ledger.violations
        if violation.kind in kinds
    }


def crosscheck_report(
    axiom_set: AxiomSet, ledgers: list[Ledger], report: ValidationReport
) -> CrosscheckResult:
    """Compare the in-repo checker's violation set with the OWL 2 RL closure's."""
    subset = _reasoner_subset(axiom_set)
    try:
        graph, individuals = _build_graph(subset, ledgers)
    except ImportError:
        _LOG.warning("[axioms] %s", REASONER_MISSING)
        return CrosscheckResult(ran=False, reason=REASONER_MISSING, kinds=list(REASONER_KINDS))
    _close(graph)
    reasoner = _reasoner_keys(graph, subset, individuals.name)
    checker = _checker_keys(report, REASONER_KINDS)
    return CrosscheckResult(
        ran=True,
        kinds=list(REASONER_KINDS),
        checker_only=sorted(checker - reasoner),
        reasoner_only=sorted(reasoner - checker),
    )
