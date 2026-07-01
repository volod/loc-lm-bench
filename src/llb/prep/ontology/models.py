"""Typed intermediate artifacts for the ontology-assisted drafting pipeline.

Every grounded artifact reuses the canonical `SourceSpan` (doc id + char offsets + exact
text), so extraction, the induced ontology, and the drafted gold items all link back to the
SAME exact-evidence representation the validator already checks. Pydantic enforces the shapes
and gives free JSON (de)serialization for the provenance bundle.
"""

from pydantic import BaseModel, Field

from llb.goldset.schema import SourceSpan


class Section(BaseModel):
    """A titled region of a document (markdown heading or paragraph block)."""

    title: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)


class DocRecord(BaseModel):
    """Stage 1: one inventoried source document with offset-preserving metadata."""

    doc_id: str  # corpus-relative path; the id the spans index into
    text: str
    sha256: str
    n_chars: int
    sections: list[Section] = Field(default_factory=list)


class Entity(BaseModel):
    """A named entity with a type, surface aliases, and exact-grounded mention spans."""

    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    mentions: list[SourceSpan] = Field(default_factory=list)


class Event(BaseModel):
    """An event description backed by an exact evidence span."""

    description: str
    evidence: SourceSpan


class Claim(BaseModel):
    """A claim/assertion backed by an exact evidence span."""

    text: str
    evidence: SourceSpan


class SROFact(BaseModel):
    """A subject-relation-object fact backed by an exact evidence span."""

    subject: str
    relation: str
    object: str
    evidence: SourceSpan


class DocExtraction(BaseModel):
    """Stage 2: everything extracted from one document, all spans exact-grounded."""

    doc_id: str
    entities: list[Entity] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    facts: list[SROFact] = Field(default_factory=list)


class OntologyType(BaseModel):
    """One induced type (entity type or relation), with support count, confidence, examples."""

    name: str
    count: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    examples: list[str] = Field(default_factory=list)


class OntologyCandidate(BaseModel):
    """Stage 3: the constrained candidate ontology induced over all documents."""

    entity_types: list[OntologyType] = Field(default_factory=list)
    relation_types: list[OntologyType] = Field(default_factory=list)


class DraftSeed(BaseModel):
    """Stage 4: a stratified target for one drafted QA item.

    `strata` records which coverage buckets the seed fills; `evidence` is the span the drafter
    must keep the answer inside; exactly one focus payload is set.
    """

    doc_id: str
    kind: str  # "fact" | "entity" | "claim" | "event"
    section_title: str
    difficulty: str  # "easy" | "medium" | "hard"
    strata: dict[str, str] = Field(default_factory=dict)
    evidence: SourceSpan
    fact: SROFact | None = None
    entity: Entity | None = None
    claim: Claim | None = None
    event: Event | None = None
