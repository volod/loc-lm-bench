"""M7.3 human-assisted RAG prompt-system generation and tuning.

Operator-facing tooling that turns a supplied text corpus into candidate RAG prompt systems, fits
them to a model's context budget, exposes them for human review, searches prompt-system variants,
and records every run prompt-system-addressably so scores compare by prompt-system id across models
and harnesses. Heavy nothing: the whole package is pure + deterministic with an injectable tokenizer.
"""

from llb.prompt_system.budget import (
    CharRatioTokenizer,
    ContextBudget,
    Tokenizer,
    plan_budget,
)
from llb.prompt_system.corpus import CorpusPackage, build_corpus_package, read_corpus
from llb.prompt_system.manifest import prompt_system_id, prompt_system_provenance
from llb.prompt_system.template import PromptPackage, TemplateFields, render_package, wrap_complete

__all__ = [
    "CharRatioTokenizer",
    "ContextBudget",
    "CorpusPackage",
    "PromptPackage",
    "TemplateFields",
    "Tokenizer",
    "build_corpus_package",
    "plan_budget",
    "prompt_system_id",
    "prompt_system_provenance",
    "read_corpus",
    "render_package",
    "wrap_complete",
]
