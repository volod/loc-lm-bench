"""Prompt-system package record contracts.

A prompt-system run directory IS the package: the anthology it selected, the per-document
metadata it summarized, the knowledge-graph-to-RAG mapping that grounds each salient term, the
rendered candidates an operator reviews, and the manifest that makes every score traceable back
to a corpus digest, a template revision, and a context budget. Each of the five was a bare JSON
array or object before; each is now a document with its own identity, so a package written by a
newer build is refused rather than half-read.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llb.core.contracts.artifacts import ArtifactContract

PROMPT_SYSTEM_ANTHOLOGY_SCHEMA_ID = "llb.prompt-system-anthology"
PROMPT_SYSTEM_DOC_METADATA_SCHEMA_ID = "llb.prompt-system-doc-metadata"
PROMPT_SYSTEM_MAPPING_SCHEMA_ID = "llb.prompt-system-mapping"
PROMPT_SYSTEM_CANDIDATES_SCHEMA_ID = "llb.prompt-system-candidates"
PROMPT_SYSTEM_MANIFEST_SCHEMA_ID = "llb.prompt-system-manifest"


class PromptSystemRow(BaseModel):
    """Strict nested row shared by the prompt-system package contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PassageRecord(PromptSystemRow):
    """One selected anthology passage, with the exact source span it was cut from."""

    passage_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str


class DocMetadataRecord(PromptSystemRow):
    """What the package says about one source document."""

    doc_id: str = Field(min_length=1)
    title: str
    n_chars: int = Field(ge=0)
    n_paragraphs: int = Field(ge=0)
    top_terms: list[str] = Field(default_factory=list)


class TemplateFieldsRecord(PromptSystemRow):
    """The operator-editable knobs one candidate was rendered from."""

    role: str
    instruction: str
    metadata_density: str
    graph_reference_style: str
    anthology_size: int = Field(ge=0)
    knowledge_tree_depth: int = Field(ge=0)
    knowledge_tree_budget: int = Field(ge=0)


class DroppedSectionRecord(PromptSystemRow):
    """One section of the budget fit: what it kept and what did not fit."""

    section: str
    n_kept: int = Field(ge=0)
    n_dropped: int = Field(ge=0)
    dropped_ids: list[str] = Field(default_factory=list)
    dropped_tokens: int = Field(ge=0)


class DroppedContextRecord(PromptSystemRow):
    """The whole budget fit for one candidate."""

    budget_tokens: int = Field(ge=0)
    used_tokens: int = Field(ge=0)
    sections: list[DroppedSectionRecord] = Field(default_factory=list)


class KnowledgeTreeRecord(PromptSystemRow):
    """The knowledge-tree block a candidate carried, and where it came from.

    `baseline_prompt_system_id` names the no-tree control this candidate is read against; it is
    absent for a candidate the tuning grid did not pair with one.
    """

    depth: int = Field(ge=0)
    budget_tokens: int = Field(ge=0)
    used_tokens: int = Field(ge=0)
    kept_lines: int = Field(ge=0)
    dropped_lines: int = Field(ge=0)
    source_kind: str
    source_digest: str
    baseline_prompt_system_id: str | None = None


class NoKnowledgeTreeRecord(PromptSystemRow):
    """The EMPTY object a candidate with no knowledge tree has always written.

    Every tree-enabled grid renders no-tree controls, so this is the common form in every package
    written before this contract, not an edge case. It is a declared alternative rather than a
    reader's fixup, so the generated JSON Schema accepts it too and an outside reader needs no
    Python of ours to validate an archived package.
    """


class PromptCandidateRecord(PromptSystemRow):
    """One reviewable candidate: its id, its knobs, its rendered prompts, and its status."""

    prompt_system_id: str = Field(min_length=1)
    fields: TemplateFieldsRecord
    system_prompt: str
    additional_prompt: str
    dropped_context: DroppedContextRecord
    used_tokens: int = Field(ge=0)
    status: str = Field(min_length=1)
    note: str = ""
    knowledge_tree: KnowledgeTreeRecord | NoKnowledgeTreeRecord | None = None


class CandidateSummaryRecord(PromptSystemRow):
    """The manifest's one-line view of a candidate, the row a board groups scores by."""

    prompt_system_id: str = Field(min_length=1)
    anthology_size: int = Field(ge=0)
    metadata_density: str
    graph_reference_style: str
    used_tokens: int = Field(ge=0)
    knowledge_tree_depth: int = Field(ge=0)
    knowledge_tree_budget: int = Field(ge=0)
    knowledge_tree_used_tokens: int = Field(ge=0)
    status: str = Field(min_length=1)


class KnowledgeTreeSourceRecord(PromptSystemRow):
    """Which knowledge-tree source the package's candidates were rendered from."""

    kind: str = Field(min_length=1)
    digest: str = Field(min_length=1)


class PromptSystemAnthology(ArtifactContract):
    """`anthology.json`: the selected passages, in selection order."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.prompt-system-anthology"]
    schema_version: Literal["1.0.0"]
    passages: list[PassageRecord] = Field(default_factory=list)


class PromptSystemDocMetadata(ArtifactContract):
    """`doc_metadata.json`: one row per source document."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.prompt-system-doc-metadata"]
    schema_version: Literal["1.0.0"]
    documents: list[DocMetadataRecord] = Field(default_factory=list)


class PromptSystemMapping(ArtifactContract):
    """`graph_rag_mapping.json`: salient term -> the anthology passages that ground it."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.prompt-system-mapping"]
    schema_version: Literal["1.0.0"]
    mapping: dict[str, list[str]] = Field(default_factory=dict)


class PromptSystemCandidates(ArtifactContract):
    """`candidates.json`: the review surface, and the file a review session writes back."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.prompt-system-candidates"]
    schema_version: Literal["1.0.0"]
    candidates: list[PromptCandidateRecord] = Field(default_factory=list)


class PromptSystemManifest(ArtifactContract):
    """`manifest.json`: the package's identity, its budget, and its candidate roll-up."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.prompt-system-manifest"]
    schema_version: Literal["1.0.0"]
    method: str = Field(min_length=1)
    corpus_digest: str = Field(min_length=1)
    mapping_digest: str = Field(min_length=1)
    tokenizer: str = Field(min_length=1)
    context_window: int = Field(ge=0)
    prompt_budget_tokens: int = Field(ge=0)
    reserved_tokens: int = Field(ge=0)
    n_candidates: int = Field(ge=0)
    knowledge_tree_source: KnowledgeTreeSourceRecord | None = None
    candidates: list[CandidateSummaryRecord] = Field(default_factory=list)
