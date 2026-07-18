"""RAG prompt-system comparison prompt-template generator -- assemble a budget-fitted RAG prompt system from the corpus.

Builds candidate system + additional prompts that embed the anthology, a metadata summary, and the
graph/RAG mapping references in a structured template with EDITABLE fields (role / instruction /
metadata density / graph-reference style / anthology size). The rendered `PromptPackage` is usable
unchanged by BOTH the baseline RAG path (as the system + attached-context prompt) and the agentic
harness lane (prepended to the task prompt), so the SAME prompt system can be measured across models
and harnesses without touching objective scoring. The attached context is trimmed to the model's
`ContextBudget`, and the dropped-context report rides along for the review loop.
"""

from dataclasses import dataclass
from typing import Callable

from llb.prompt_system.budget import (
    ContextBudget,
    DroppedContextReport,
    SectionItem,
    Tokenizer,
    fit_sections,
)
from llb.prompt_system.corpus import CorpusPackage, DocMetadata, Passage
from llb.prompts.registry import render_text, render_text_map

# Metadata density: how much per-document summary the attached context carries.
METADATA_NONE = "none"
METADATA_COMPACT = "compact"
METADATA_FULL = "full"
METADATA_DENSITIES = (METADATA_NONE, METADATA_COMPACT, METADATA_FULL)

# Graph-reference style: how the knowledge-graph-to-RAG mapping is surfaced.
GRAPH_NONE = "none"
GRAPH_INLINE = "inline"
GRAPH_APPENDIX = "appendix"
GRAPH_STYLES = (GRAPH_NONE, GRAPH_INLINE, GRAPH_APPENDIX)

SECTION_ANTHOLOGY = "anthology"
SECTION_METADATA = "metadata"
SECTION_GRAPH = "graph"

_DEFAULT_ROLE = render_text("prompt_system.default_role")
_DEFAULT_INSTRUCTION = render_text("prompt_system.default_instruction")
_HEADINGS = render_text_map("prompt_system.headings")


@dataclass(slots=True)
class TemplateFields:
    """The operator-editable knobs the prompt-tuning loop searches over."""

    role: str = _DEFAULT_ROLE
    instruction: str = _DEFAULT_INSTRUCTION
    metadata_density: str = METADATA_COMPACT
    graph_reference_style: str = GRAPH_INLINE
    anthology_size: int = 8
    knowledge_tree_depth: int = 0
    knowledge_tree_budget: int = 0

    def validate(self) -> None:
        if self.metadata_density not in METADATA_DENSITIES:
            raise ValueError(f"unknown metadata_density: {self.metadata_density!r}")
        if self.graph_reference_style not in GRAPH_STYLES:
            raise ValueError(f"unknown graph_reference_style: {self.graph_reference_style!r}")
        if self.anthology_size < 0:
            raise ValueError("anthology_size must be >= 0")
        tree_enabled = self.knowledge_tree_depth > 0 or self.knowledge_tree_budget > 0
        if tree_enabled and (self.knowledge_tree_depth <= 0 or self.knowledge_tree_budget <= 0):
            raise ValueError("knowledge-tree depth and budget must both be > 0")


@dataclass(slots=True)
class PromptPackage:
    """A rendered, budget-fitted prompt system, usable by the RAG and agentic harness lanes alike."""

    system_prompt: str
    additional_prompt: str
    fields: TemplateFields
    dropped_context: DroppedContextReport
    used_tokens: int = 0

    def as_prefix(self) -> str:
        """A single prompt prefix (system + attached context) for the single-string agentic lane."""
        return f"{self.system_prompt}\n\n{self.additional_prompt}".strip()

    def apply(self, prompt: str) -> str:
        """Prepend the prompt system to an existing task/agent prompt (scoring unchanged)."""
        return f"{self.as_prefix()}\n\n{prompt}"


def _metadata_items(metadata: list[DocMetadata], density: str) -> list[SectionItem]:
    if density == METADATA_NONE:
        return []
    if density == METADATA_COMPACT:
        if not metadata:
            return []
        summary = "; ".join(f"{m['title']} ({m['doc_id']})" for m in metadata)
        return [
            {
                "item_id": "metadata::compact",
                "text": render_text("prompt_system.metadata_compact", {"summary": summary}),
            }
        ]
    return [
        {
            "item_id": f"metadata::{m['doc_id']}",
            "text": render_text(
                "prompt_system.metadata_full_item",
                {
                    "title": m["title"],
                    "doc_id": m["doc_id"],
                    "n_chars": m["n_chars"],
                    "top_terms": ", ".join(m["top_terms"]),
                },
            ),
        }
        for m in metadata
    ]


def _graph_items(mapping: dict[str, list[str]], style: str) -> list[SectionItem]:
    if style == GRAPH_NONE or not mapping:
        return []
    return [
        {
            "item_id": f"graph::{term}",
            "text": render_text(
                "prompt_system.graph_item",
                {"term": term, "passage_ids": ", ".join(passage_ids)},
            ),
        }
        for term, passage_ids in mapping.items()
    ]


def _anthology_items(anthology: list[Passage], size: int) -> list[SectionItem]:
    return [
        {
            "item_id": p["passage_id"],
            "text": render_text(
                "prompt_system.anthology_item",
                {"passage_id": p["passage_id"], "text": p["text"]},
            ),
        }
        for p in anthology[: max(0, size)]
    ]


def build_sections(
    corpus: CorpusPackage, fields: TemplateFields
) -> list[tuple[str, list[SectionItem]]]:
    """Ordered (most-important-first) sections the budget controller trims to fit a model."""
    return [
        (SECTION_ANTHOLOGY, _anthology_items(corpus.anthology, fields.anthology_size)),
        (SECTION_GRAPH, _graph_items(corpus.graph_rag_mapping, fields.graph_reference_style)),
        (SECTION_METADATA, _metadata_items(corpus.metadata, fields.metadata_density)),
    ]


def _render_additional_prompt(kept: dict[str, list[SectionItem]], graph_style: str) -> str:
    blocks: list[str] = []
    anthology = kept.get(SECTION_ANTHOLOGY, [])
    if anthology:
        body = "\n\n".join(item["text"] for item in anthology)
        blocks.append(_section_block(_HEADINGS["anthology"], body))
    metadata = kept.get(SECTION_METADATA, [])
    if metadata:
        body = "\n".join(item["text"] for item in metadata)
        blocks.append(_section_block(_HEADINGS["metadata"], body))
    graph = kept.get(SECTION_GRAPH, [])
    if graph:
        body = "\n".join(item["text"] for item in graph)
        heading = _HEADINGS["graph_appendix" if graph_style == GRAPH_APPENDIX else "graph_inline"]
        blocks.append(_section_block(heading, body))
    return "\n\n".join(blocks)


def _section_block(heading: str, body: str) -> str:
    return render_text("prompt_system.section_block", {"heading": heading, "body": body})


def render_package(
    corpus: CorpusPackage,
    fields: TemplateFields,
    budget: ContextBudget,
    tokenizer: Tokenizer,
    *,
    knowledge_tree_text: str = "",
) -> PromptPackage:
    """Render a budget-fitted `PromptPackage` from the corpus inputs and editable template fields."""
    fields.validate()
    sections = build_sections(corpus, fields)
    tree_tokens = tokenizer.count(knowledge_tree_text)
    fit = fit_sections(sections, max(0, budget.prompt_budget - tree_tokens), tokenizer)
    system_prompt = render_text(
        "prompt_system.system_prompt",
        {"role": fields.role, "instruction": fields.instruction},
    )
    if knowledge_tree_text:
        system_prompt = f"{system_prompt}\n\n{knowledge_tree_text}"
    additional_prompt = _render_additional_prompt(fit.kept, fields.graph_reference_style)
    return PromptPackage(
        system_prompt=system_prompt,
        additional_prompt=additional_prompt,
        fields=fields,
        dropped_context=fit.report,
        used_tokens=fit.used_tokens + tree_tokens,
    )


def wrap_complete(complete: Callable[[str], str], package: PromptPackage) -> Callable[[str], str]:
    """Wrap a candidate `complete` (prompt -> text) so the prompt system is prepended to EVERY call.

    The same `PromptPackage` drives the baseline RAG path and agentic harnesses without touching the
    objective scorer; it only adds grounding context to the prompt each harness builds.
    """

    def wrapped(prompt: str) -> str:
        return complete(package.apply(prompt))

    return wrapped
