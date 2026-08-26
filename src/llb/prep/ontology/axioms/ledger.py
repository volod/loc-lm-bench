"""The read-side view of an extraction ledger that the axiom checker runs over.

The checker asks three questions of a ledger and nothing else: which facts carry a relation, what
types a name was asserted to have, and where each of those assertions is grounded. Building that
index once keeps every axiom class a single pass over a dict rather than a re-scan of the ledger.
"""

from collections import defaultdict

from pydantic import BaseModel

from llb.goldset.schema import SourceSpan
from llb.prep.ontology.axioms.models import ViolationFact
from llb.prep.ontology.models import DocExtraction, SROFact
from llb.prep.ontology.naming import normalize_name, normalize_relation

# The pseudo-relation a type assertion is reported under, so a disjointness or domain violation
# renders in the same three-column shape as a fact violation.
TYPE_RELATION = "rdf:type"


class TypeAssertion(BaseModel):
    """One "this name is of this type" claim, with the mention span that grounds it."""

    name: str
    entity_type: str
    evidence: SourceSpan

    def as_fact(self) -> ViolationFact:
        """Render the assertion in the violation's fact shape (subject / relation / object)."""
        return ViolationFact(
            subject=self.name,
            relation=TYPE_RELATION,
            object=self.entity_type,
            evidence=self.evidence,
        )


class Ledger:
    """A relation- and name-indexed view over one corpus's `DocExtraction` records."""

    def __init__(self, extractions: list[DocExtraction]) -> None:
        self.n_docs = len(extractions)
        self.facts: list[SROFact] = [f for e in extractions for f in e.facts]
        self.n_entities = sum(len(e.entities) for e in extractions)
        self._by_relation: dict[str, list[SROFact]] = defaultdict(list)
        for fact in self.facts:
            self._by_relation[normalize_relation(fact.relation)].append(fact)
        self._types: dict[str, list[TypeAssertion]] = defaultdict(list)
        for extraction in extractions:
            for entity in extraction.entities:
                self._add_entity_type(entity.name, entity.type, entity.mentions)

    def _add_entity_type(self, name: str, entity_type: str, mentions: list[SourceSpan]) -> None:
        if not mentions:  # an ungrounded entity cannot be cited in a violation, so it is not one
            return
        key = normalize_name(name)
        if any(a.entity_type == entity_type for a in self._types[key]):
            return  # the same type asserted again adds no evidence a reviewer has not seen
        self._types[key].append(
            TypeAssertion(name=name, entity_type=entity_type, evidence=mentions[0])
        )

    @property
    def n_facts(self) -> int:
        return len(self.facts)

    @property
    def n_relations(self) -> int:
        return len(self._by_relation)

    def relation_facts(self, relation: str) -> list[SROFact]:
        """Every fact whose relation surface folds to `relation`, in ledger order."""
        return self._by_relation.get(normalize_relation(relation), [])

    def types_of(self, name: str) -> list[TypeAssertion]:
        """Every type asserted for the folded name; empty for an untyped fact-only endpoint."""
        return self._types.get(normalize_name(name), [])

    def typed_names(self) -> list[tuple[str, list[TypeAssertion]]]:
        """(folded name, assertions) pairs in a deterministic order."""
        return sorted(self._types.items())
