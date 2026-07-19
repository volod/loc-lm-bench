"""RAG prompt-system comparison prompt-system identity + run artifacts -- digests that make every run addressable.

A prompt SYSTEM is identified by the corpus it was built from and the template fields used, so two
runs are comparable iff they share a `prompt_system_id`. The digests (corpus / mapping / template)
plus the tokenizer + context budget are recorded in the run manifest, so a board can group scores by
prompt-system id and an operator can always trace a score back to its exact corpus + template +
budget inputs. Pure + dependency-free (hashlib + json).
"""

import hashlib
import json
from typing import Any

from typing_extensions import NotRequired, TypedDict

from llb.prompt_system.budget import ContextBudget
from llb.prompt_system.corpus import CorpusPackage
from llb.prompt_system.template import TemplateFields

_DIGEST_LEN = 12


def _digest(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:_DIGEST_LEN]


def corpus_digest(corpus: CorpusPackage) -> str:
    """Stable digest over the prepared corpus inputs (anthology + metadata + mapping + terms)."""
    return _digest(
        {
            "anthology": corpus.anthology,
            "metadata": corpus.metadata,
            "mapping": corpus.graph_rag_mapping,
            "terms": corpus.salient_terms,
        }
    )


def mapping_digest(corpus: CorpusPackage) -> str:
    """Digest of just the knowledge-graph-to-RAG mapping (its own provenance line in the manifest)."""
    return _digest(corpus.graph_rag_mapping)


def template_digest(fields: TemplateFields) -> str:
    """Digest over the editable template fields (the prompt template revision)."""
    payload = {
        "role": fields.role,
        "instruction": fields.instruction,
        "metadata_density": fields.metadata_density,
        "graph_reference_style": fields.graph_reference_style,
        "anthology_size": fields.anthology_size,
    }
    if fields.knowledge_tree_depth > 0:
        payload["knowledge_tree_depth"] = fields.knowledge_tree_depth
        payload["knowledge_tree_budget"] = fields.knowledge_tree_budget
    return _digest(payload)


def prompt_system_id(
    corpus: CorpusPackage, fields: TemplateFields, *, knowledge_tree_text: str = ""
) -> str:
    """The comparison key: same corpus + same template fields -> same prompt-system id."""
    payload = {"corpus": corpus_digest(corpus), "template": template_digest(fields)}
    if knowledge_tree_text:
        payload["knowledge_tree"] = _digest(knowledge_tree_text)
    return _digest(payload)


class PromptSystemProvenance(TypedDict):
    """The manifest block that makes a run prompt-system-addressable (board axis + traceability)."""

    prompt_system_id: str
    corpus_digest: str
    mapping_digest: str
    template_revision: str
    tokenizer: str
    context_window: int
    prompt_budget_tokens: int
    knowledge_tree: NotRequired[dict[str, object]]


def prompt_system_provenance(
    corpus: CorpusPackage,
    fields: TemplateFields,
    budget: ContextBudget,
    *,
    tokenizer: str,
) -> PromptSystemProvenance:
    """Build the manifest provenance block for a run that used this prompt system + budget."""
    return {
        "prompt_system_id": prompt_system_id(corpus, fields),
        "corpus_digest": corpus_digest(corpus),
        "mapping_digest": mapping_digest(corpus),
        "template_revision": template_digest(fields),
        "tokenizer": tokenizer,
        "context_window": budget.context_window,
        "prompt_budget_tokens": budget.prompt_budget,
    }
