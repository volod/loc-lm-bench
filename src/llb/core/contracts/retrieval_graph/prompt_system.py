"""Prompt-system package contracts: the prepared corpus, the candidates, and the run manifest.

A prompt-system package is what makes a benchmark score traceable back to the exact corpus,
template, and budget that produced it. Its five members are prepared together and reviewed
together, so each is a registered record rather than a shape that happens to survive whichever
reader opens it next.
"""

from typing import Literal

from pydantic import Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.retrieval_graph.common import RetrievalRow

ANTHOLOGY_SCHEMA_ID = "llb.prompt-system-anthology"
DOC_METADATA_SCHEMA_ID = "llb.prompt-system-doc-metadata"
MAPPING_SCHEMA_ID = "llb.prompt-system-mapping"
CANDIDATES_SCHEMA_ID = "llb.prompt-system-candidates"
PROMPT_SYSTEM_MANIFEST_SCHEMA_ID = "llb.prompt-system-manifest"


class PassageRow(RetrievalRow):
    """One selected anthology passage, its source span preserved exactly."""

    passage_id: str
    doc_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str


class DocMetadataRow(RetrievalRow):
    """One document's summary as the prompt template surfaces it."""

    doc_id: str
    title: str
    n_chars: int = Field(ge=0)
    n_paragraphs: int = Field(ge=0)
    top_terms: list[str] = Field(default_factory=list)


class AnthologyDocument(ArtifactContract):
    """`anthology.json`: the passages a candidate prompt may attach, in selection order."""

    schema_id: Literal["llb.prompt-system-anthology"]
    schema_version: Literal["1.0.0"]
    passages: list[PassageRow] = Field(default_factory=list)


class DocMetadataDocument(ArtifactContract):
    """`doc_metadata.json`: the per-document summary block, in corpus order."""

    schema_id: Literal["llb.prompt-system-doc-metadata"]
    schema_version: Literal["1.0.0"]
    documents: list[DocMetadataRow] = Field(default_factory=list)


class GraphRagMappingDocument(ArtifactContract):
    """`graph_rag_mapping.json`: each salient term and the passage ids that ground it."""

    schema_id: Literal["llb.prompt-system-mapping"]
    schema_version: Literal["1.0.0"]
    mapping: dict[str, list[str]] = Field(default_factory=dict)


class TemplateFieldsRow(RetrievalRow):
    """The operator-editable knobs one candidate was rendered from."""

    role: str
    instruction: str
    metadata_density: str
    graph_reference_style: str
    anthology_size: int = Field(ge=0)
    knowledge_tree_depth: int = Field(default=0, ge=0)
    knowledge_tree_budget: int = Field(default=0, ge=0)


class DroppedSectionRow(RetrievalRow):
    """What one attached-context section lost when it was fitted to the budget."""

    section: str
    n_kept: int = Field(ge=0)
    n_dropped: int = Field(ge=0)
    dropped_ids: list[str] = Field(default_factory=list)
    dropped_tokens: int = Field(ge=0)


class DroppedContextRow(RetrievalRow):
    """The budget the candidate was fitted to and everything it could not carry."""

    budget_tokens: int = Field(ge=0)
    used_tokens: int = Field(ge=0)
    sections: list[DroppedSectionRow] = Field(default_factory=list)


class KnowledgeTreeRow(RetrievalRow):
    """The knowledge-tree block a candidate carried, and where its lines came from.

    `budget_tokens` is what the render was actually given -- the requested budget capped by the
    prompt budget -- so `requested_budget_tokens` states the ask beside it. Every tree candidate
    also names the no-tree control it varies (`baseline_prompt_system_id`), which is what makes it
    readable as a comparison rather than as a lone candidate.
    """

    depth: int = Field(ge=0)
    budget_tokens: int = Field(ge=0)
    used_tokens: int = Field(ge=0)
    kept_lines: int = Field(ge=0)
    dropped_lines: int = Field(ge=0)
    source_kind: str
    source_digest: str
    requested_budget_tokens: int = Field(ge=0)
    baseline_prompt_system_id: str


class PromptCandidateRow(RetrievalRow):
    """One reviewable candidate: its rendered prompts, its budget accounting, its status."""

    prompt_system_id: str
    fields: TemplateFieldsRow
    system_prompt: str
    additional_prompt: str
    dropped_context: DroppedContextRow
    used_tokens: int = Field(ge=0)
    status: str
    note: str = ""
    # Absent (an empty object) on a candidate rendered without a knowledge tree.
    knowledge_tree: KnowledgeTreeRow | None = None


class PromptCandidatesDocument(ArtifactContract):
    """`candidates.json`: every generated candidate and the review status it carries."""

    schema_id: Literal["llb.prompt-system-candidates"]
    schema_version: Literal["1.0.0"]
    candidates: list[PromptCandidateRow] = Field(default_factory=list)


class CandidateSummaryRow(RetrievalRow):
    """The manifest's index line for one candidate: enough to pick one without opening it."""

    prompt_system_id: str
    anthology_size: int = Field(ge=0)
    metadata_density: str
    graph_reference_style: str
    used_tokens: int = Field(ge=0)
    knowledge_tree_depth: int = Field(ge=0)
    knowledge_tree_budget: int = Field(ge=0)
    knowledge_tree_used_tokens: int = Field(ge=0)
    status: str


class KnowledgeTreeSourceRow(RetrievalRow):
    """Which artifact the knowledge tree was rendered from, and its digest."""

    kind: str
    digest: str


class PromptSystemManifestDocument(ArtifactContract):
    """`manifest.json`: the digests and budget that make every run of this package addressable."""

    schema_id: Literal["llb.prompt-system-manifest"]
    schema_version: Literal["1.0.0"]
    method: str
    corpus_digest: str
    mapping_digest: str
    tokenizer: str
    context_window: int = Field(ge=0)
    prompt_budget_tokens: int = Field(ge=0)
    reserved_tokens: int = Field(ge=0)
    n_candidates: int = Field(ge=0)
    knowledge_tree_source: KnowledgeTreeSourceRow | None = None
    candidates: list[CandidateSummaryRow] = Field(default_factory=list)
