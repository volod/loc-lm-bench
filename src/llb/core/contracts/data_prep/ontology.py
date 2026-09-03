"""Ontology drafting bundle contracts: provenance, the induced ontology, and extraction rows.

`provenance.json` is the dictionary that exposed the gap this migration closes: it was assembled
as a bare `dict[str, object]`, so nothing checked that a bundle carried its corpus binding, its
document rows, or its endpoint record until a much later reader missed a key. Two versions are
registered because the binding is younger than the bundle: a pre-binding bundle names no corpus
version and its document rows carry no acquisition provenance, and it migrates to the current form
by stating those absences instead of omitting them.
"""

from typing import Literal

from pydantic import Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject
from llb.core.contracts.data_prep.corpus import DataPrepRow
from llb.core.contracts.data_prep.goldset import SourceSpanRecord


class ProvenanceDocumentV1(DataPrepRow):
    """A pre-binding document row: local identity only."""

    doc_id: str = Field(min_length=1)
    sha256: str = Field(min_length=1)
    n_chars: int = Field(ge=0)


class ProvenanceDocument(ProvenanceDocumentV1):
    """A current document row: local identity plus the acquisition provenance it carries."""

    source_uri: str | None = None
    capture_time: str | None = None
    capture_id: str | None = None
    payload_digest: str | None = None
    licence: str | None = None
    acquisition_run_id: str | None = None
    revision_of: str | None = None


class CorpusVersionRecord(DataPrepRow):
    """Portable identity of the corpus version a bundle was drafted from."""

    corpus_fingerprint: str = Field(min_length=1)
    acquisition_run_ids: list[str] = Field(default_factory=list)


class _ProvenanceFields(ArtifactContract):
    """Fields both provenance versions share.

    A bundle that aborted on budget writes the same record with `abort` set and the drafting
    results absent, which is why the result fields are optional rather than defaulted: an absent
    stage count is a run that never reached the stage, not a stage that produced nothing.
    """

    kind: Literal["ontology-drafted"]
    synthetic: bool
    endpoint: JsonObject = Field(default_factory=dict)
    settings: JsonObject = Field(default_factory=dict)
    elapsed_s: float = Field(ge=0.0)
    cost: JsonObject = Field(default_factory=dict)
    status: str | None = None
    abort: JsonObject | None = None
    prompts: dict[str, str] | None = None
    seed: int | None = None
    stages: JsonObject | None = None
    labels: JsonObject | None = None
    ontology: JsonObject | None = None
    n_items: int | None = None
    seed_coverage: JsonObject | None = None
    dedup: JsonObject | None = None
    multi_hop_carry_forward: JsonObject | None = None
    applied_feedback: JsonObject | None = None
    multi_hop_path_strata: JsonObject | None = None


class OntologyProvenanceV1(_ProvenanceFields):
    """The pre-binding bundle record: no corpus version, document rows without acquisition."""

    schema_id: Literal["llb.ontology-provenance"]
    schema_version: Literal["1.0.0"]
    documents: list[ProvenanceDocumentV1] | None = None


class OntologyProvenance(_ProvenanceFields):
    """The current bundle record: the corpus version it drafted from, stated per document."""

    schema_id: Literal["llb.ontology-provenance"]
    schema_version: Literal["2.0.0"]
    corpus_version: CorpusVersionRecord | None = None
    documents: list[ProvenanceDocument] | None = None


class OntologyTypeRecord(DataPrepRow):
    """One induced type with its support count, confidence, and examples."""

    name: str = Field(min_length=1)
    count: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    examples: list[str] = Field(default_factory=list)


class OntologyDocument(ArtifactContract):
    """`ontology.json`: the constrained candidate ontology induced over the corpus."""

    schema_id: Literal["llb.ontology"]
    schema_version: Literal["1.0.0"]
    entity_types: list[OntologyTypeRecord] = Field(default_factory=list)
    relation_types: list[OntologyTypeRecord] = Field(default_factory=list)


class EntityRecord(DataPrepRow):
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    mentions: list[SourceSpanRecord] = Field(default_factory=list)


class EventRecord(DataPrepRow):
    description: str
    evidence: SourceSpanRecord


class ClaimRecord(DataPrepRow):
    text: str
    evidence: SourceSpanRecord


class FactRecord(DataPrepRow):
    subject: str
    relation: str
    object: str
    evidence: SourceSpanRecord


class OntologyExtractionRow(ArtifactContract):
    """One `extraction.jsonl` row: everything extracted from one document, spans exact-grounded."""

    schema_id: Literal["llb.ontology-extraction"]
    schema_version: Literal["1.0.0"]
    doc_id: str = Field(min_length=1)
    entities: list[EntityRecord] = Field(default_factory=list)
    events: list[EventRecord] = Field(default_factory=list)
    claims: list[ClaimRecord] = Field(default_factory=list)
    facts: list[FactRecord] = Field(default_factory=list)
