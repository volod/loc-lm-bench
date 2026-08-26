"""Typed models for the axiom layer: the constraints, the violations, and the per-axiom evidence.

An `Axiom` is a claim about the domain, so it carries a Ukrainian gloss and a sign-off field
beside its logical content -- a reviewer accepts or rejects the SENTENCE, not the triple. A
`Violation` carries every fact it rests on with the fact's exact `SourceSpan`, so a reviewer
adjudicates it without re-reading the corpus.
"""

from pydantic import BaseModel, Field, model_validator

from llb.goldset.schema import SourceSpan
from llb.prep.ontology.axioms.constants import (
    AXIOM_KINDS,
    DISJOINT_TYPES,
    DOMAIN,
    MAX_CARDINALITY,
    RANGE,
    RELATION_KINDS,
)
from llb.prep.ontology.extraction.entity_types import ENTITY_TYPES


class Axiom(BaseModel):
    """One constraint over the closed vocabulary or an induced relation.

    `relation` is the Ukrainian relation surface as the extractor emits it (matched
    whitespace/case-insensitively); `entity_types` is the allowed set for `domain`/`range` and the
    pair for `disjoint_types`; `max_count` is the bound for `max_cardinality`.
    """

    axiom_id: str
    kind: str
    relation: str | None = None
    entity_types: list[str] = Field(default_factory=list)
    max_count: int | None = Field(default=None, ge=1)
    gloss: str = ""  # the domain sentence a reviewer accepts or rejects, in Ukrainian
    signed_by: str | None = None
    signed_on: str | None = None  # ISO date; both set together by the sign-off lane

    @property
    def signed(self) -> bool:
        """A signed axiom is one a named reviewer dated -- never one the corpus voted for."""
        return bool(self.signed_by and self.signed_on)

    @model_validator(mode="after")
    def _check_shape(self) -> "Axiom":
        if self.kind not in AXIOM_KINDS:
            raise ValueError(f"unknown axiom kind {self.kind!r}; expected one of {AXIOM_KINDS}")
        if self.kind in RELATION_KINDS and not self.relation:
            raise ValueError(f"{self.kind} axiom {self.axiom_id!r} needs a relation")
        if self.kind == DISJOINT_TYPES and len(self.entity_types) != 2:
            raise ValueError(f"{self.kind} axiom {self.axiom_id!r} needs exactly two entity types")
        if self.kind in (DOMAIN, RANGE) and not self.entity_types:
            raise ValueError(f"{self.kind} axiom {self.axiom_id!r} needs at least one entity type")
        unknown = [t for t in self.entity_types if t not in ENTITY_TYPES]
        if unknown:
            raise ValueError(
                f"axiom {self.axiom_id!r} names types outside the vocabulary: {unknown}"
            )
        if self.kind == MAX_CARDINALITY and self.max_count is None:
            raise ValueError(f"{self.kind} axiom {self.axiom_id!r} needs max_count")
        return self


class AxiomSet(BaseModel):
    """The constraint set as a whole, with the version an operator points the checker at."""

    version: str
    axioms: list[Axiom] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "AxiomSet":
        ids = [axiom.axiom_id for axiom in self.axioms]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate axiom ids: {duplicates}")
        return self

    @property
    def signed(self) -> list[Axiom]:
        """The subset an answer gate may enable -- every other axiom is still a candidate."""
        return [axiom for axiom in self.axioms if axiom.signed]


class ViolationFact(BaseModel):
    """One fact (or type assertion) a violation rests on, with its exact evidence span."""

    subject: str
    relation: str
    object: str
    evidence: SourceSpan


class Violation(BaseModel):
    """One broken axiom, naming the axiom and every fact that breaks it."""

    axiom_id: str
    kind: str
    relation: str | None = None
    subject: str | None = None
    detail: str  # one sentence: what the ledger asserts that the axiom forbids
    facts: list[ViolationFact] = Field(default_factory=list)

    def key(self) -> tuple[str, str, str, str]:
        """Deterministic sort/compare key -- also what the reasoner cross-check matches on."""
        endpoints = "|".join(sorted(f"{f.subject}->{f.object}" for f in self.facts))
        return (self.axiom_id, self.kind, self.subject or "", endpoints)


class AxiomStat(BaseModel):
    """What one axiom cost and bought on one ledger.

    `checked` counts the UNITS the axiom was evaluated over (subject groups for a cardinality
    class, facts for a per-fact class, entity names for disjointness), not the facts behind them,
    so `violating / checked` is a base rate a reviewer can read.
    """

    axiom_id: str
    kind: str
    relation: str | None = None
    entity_types: list[str] = Field(default_factory=list)
    checked: int = 0
    supporting: int = 0
    violating: int = 0
    unchecked: int = 0  # units skipped for an untyped endpoint (domain/range only)
    violations: int = 0  # emitted rows; a unit with several conflicting pairs emits several

    @property
    def rate(self) -> float:
        """Violating units over checked units; 0.0 when the axiom had nothing to check."""
        return self.violating / self.checked if self.checked else 0.0


class AxiomEvidence(BaseModel):
    """Per-axiom worksheet input for the sign-off lane: what supports it, what contradicts it."""

    axiom_id: str
    kind: str
    gloss: str = ""
    turtle: str = ""  # the axiom rendered on its own, for a reviewer who reads OWL
    stat: AxiomStat
    supporting: list[ViolationFact] = Field(default_factory=list)
    contradicting: list[Violation] = Field(default_factory=list)


class LedgerReport(BaseModel):
    """The result of checking one extraction ledger against the whole axiom set."""

    label: str
    source: str  # project-relative path when the ledger lives inside the repo, else its name
    n_docs: int = 0
    n_entities: int = 0
    n_facts: int = 0
    n_relations: int = 0
    stats: list[AxiomStat] = Field(default_factory=list)
    violations: list[Violation] = Field(default_factory=list)


class CrosscheckResult(BaseModel):
    """Whether the `owlrl` reasoner reproduced the in-repo checker's violation set."""

    ran: bool
    reason: str = ""  # why it did not run, when it did not
    kinds: list[str] = Field(default_factory=list)  # the classes OWL RL decides
    checker_only: list[str] = Field(default_factory=list)  # violation keys the reasoner missed
    reasoner_only: list[str] = Field(default_factory=list)  # violation keys the checker missed

    @property
    def agrees(self) -> bool:
        return self.ran and not self.checker_only and not self.reasoner_only


class ValidationReport(BaseModel):
    """Everything one `validate-ontology-axioms` run produced."""

    axioms_source: str
    axioms_version: str
    n_axioms: int
    n_signed: int
    ledgers: list[LedgerReport] = Field(default_factory=list)
    evidence: list[AxiomEvidence] = Field(default_factory=list)
    crosscheck: CrosscheckResult | None = None

    @property
    def n_violations(self) -> int:
        return sum(len(ledger.violations) for ledger in self.ledgers)
