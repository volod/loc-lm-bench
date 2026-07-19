"""RAG prompt-system comparison prompt-system pipeline -- corpus -> candidates -> reviewable, manifest-addressable run.

Ties the RAG prompt-system comparison stages together: read a caller-provided corpus, build the anthology / metadata /
graph-RAG mapping, plan the per-model context budget, search the template-variant grid, render
budget-fitted candidates, and persist everything under `$DATA_DIR/prompt-system/<run_timestamp>/`
(anthology, metadata, mapping, candidates, and a manifest carrying the corpus / mapping / template
digests + tokenizer + context budget). The artifacts are the operator's review surface and the
provenance the benchmark records per run.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from llb.bench.common import new_run_timestamp
from llb.core.paths import resolve_data_dir
from llb.prompt_system.budget import (
    DEFAULT_ANSWER_TOKENS,
    CharRatioTokenizer,
    ContextBudget,
    Tokenizer,
    plan_budget,
)
from llb.prompt_system.corpus import CorpusPackage, build_corpus_package, read_corpus
from llb.prompt_system.manifest import (
    corpus_digest,
    mapping_digest,
)
from llb.prompt_system.knowledge_tree_render import DEFAULT_TREE_BUDGETS, DEFAULT_TREE_DEPTHS
from llb.prompt_system.knowledge_tree_source import load_knowledge_tree_source
from llb.prompt_system.review import PromptCandidate, candidate_to_dict, save_candidates
from llb.prompt_system.template import GRAPH_STYLES, TemplateFields
from llb.prompt_system.tuning import (
    generate_candidates,
    variant_grid,
    with_knowledge_tree_variants,
)

_LOG = logging.getLogger(__name__)

METHOD = "prompt-system"
ANTHOLOGY_FILE = "anthology.json"
METADATA_FILE = "doc_metadata.json"
MAPPING_FILE = "graph_rag_mapping.json"
CANDIDATES_FILE = "candidates.json"
MANIFEST_FILE = "manifest.json"

DEFAULT_QUESTION_TOKENS = 64
DEFAULT_CHUNK_TOKENS = 1024


@dataclass(slots=True)
class PromptSystemRun:
    """The result of a prepared prompt-system run (in-memory objects + the written run dir)."""

    corpus: CorpusPackage
    candidates: list[PromptCandidate]
    budget: ContextBudget
    tokenizer_name: str
    run_dir: Path


def prepare_prompt_system(
    corpus_root: Path | str,
    *,
    data_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    base_fields: TemplateFields | None = None,
    context_window: int = 8192,
    tokenizer: Tokenizer | None = None,
    tokenizer_name: str = "char-ratio",
    question_tokens: int = DEFAULT_QUESTION_TOKENS,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    transcript_tokens: int = 0,
    answer_tokens: int = DEFAULT_ANSWER_TOKENS,
    max_passages: int = 12,
    anthology_sizes: list[int] | None = None,
    graph_styles: list[str] | None = None,
    metadata_densities: list[str] | None = None,
    ontology_bundle: Path | str | None = None,
    graph_dir: Path | str | None = None,
    knowledge_tree_depths: list[int] | None = None,
    knowledge_tree_budgets: list[int] | None = None,
    persist: bool = True,
) -> PromptSystemRun:
    """Prepare + persist a reviewable prompt-system run from a corpus directory."""
    docs = read_corpus(corpus_root)
    corpus = build_corpus_package(docs, max_passages=max_passages)
    budget = plan_budget(
        context_window,
        question_tokens=question_tokens,
        chunk_tokens=chunk_tokens,
        transcript_tokens=transcript_tokens,
        answer_tokens=answer_tokens,
    )
    tok = tokenizer or CharRatioTokenizer()
    grid = variant_grid(
        base_fields or TemplateFields(),
        anthology_sizes=anthology_sizes or [max(1, max_passages // 2), max_passages],
        graph_styles=graph_styles or list(GRAPH_STYLES),
        metadata_densities=metadata_densities,
    )
    tree_source = None
    if ontology_bundle is not None or graph_dir is not None:
        tree_source = load_knowledge_tree_source(
            ontology_bundle=ontology_bundle,
            graph_dir=graph_dir,
        )
        grid = with_knowledge_tree_variants(
            grid,
            depths=knowledge_tree_depths or list(DEFAULT_TREE_DEPTHS),
            budgets=knowledge_tree_budgets or list(DEFAULT_TREE_BUDGETS),
        )
    candidates = generate_candidates(
        corpus,
        grid,
        budget,
        tok,
        knowledge_tree_source=tree_source,
    )

    run_dir = (
        Path(out_dir)
        if out_dir is not None
        else resolve_data_dir(data_dir) / METHOD / new_run_timestamp()[1]
    )
    if persist:
        _write_run(run_dir, corpus, candidates, budget, tokenizer_name)
        _LOG.info(
            "[prompt-system] %d candidates over %d docs -> %s",
            len(candidates),
            len(docs),
            run_dir / MANIFEST_FILE,
        )
    return PromptSystemRun(
        corpus=corpus,
        candidates=candidates,
        budget=budget,
        tokenizer_name=tokenizer_name,
        run_dir=run_dir,
    )


def _write_run(
    run_dir: Path,
    corpus: CorpusPackage,
    candidates: list[PromptCandidate],
    budget: ContextBudget,
    tokenizer_name: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / ANTHOLOGY_FILE, corpus.anthology)
    _write_json(run_dir / METADATA_FILE, corpus.metadata)
    _write_json(run_dir / MAPPING_FILE, corpus.graph_rag_mapping)
    save_candidates(candidates, run_dir / CANDIDATES_FILE)
    manifest = {
        "method": METHOD,
        "corpus_digest": corpus_digest(corpus),
        "mapping_digest": mapping_digest(corpus),
        "tokenizer": tokenizer_name,
        "context_window": budget.context_window,
        "prompt_budget_tokens": budget.prompt_budget,
        "reserved_tokens": budget.reserved,
        "n_candidates": len(candidates),
        "knowledge_tree_source": _knowledge_tree_source_manifest(candidates),
        "candidates": [
            {
                "prompt_system_id": c.prompt_system_id,
                "anthology_size": c.fields.anthology_size,
                "metadata_density": c.fields.metadata_density,
                "graph_reference_style": c.fields.graph_reference_style,
                "used_tokens": c.used_tokens,
                "knowledge_tree_depth": c.fields.knowledge_tree_depth,
                "knowledge_tree_budget": c.fields.knowledge_tree_budget,
                "knowledge_tree_used_tokens": c.knowledge_tree.get("used_tokens", 0),
                "status": c.status,
            }
            for c in candidates
        ],
    }
    _write_json(run_dir / MANIFEST_FILE, manifest)


def _knowledge_tree_source_manifest(candidates: list[PromptCandidate]) -> dict[str, object] | None:
    tree = next(
        (candidate.knowledge_tree for candidate in candidates if candidate.knowledge_tree), None
    )
    if tree is None:
        return None
    return {
        "kind": tree["source_kind"],
        "digest": tree["source_digest"],
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# Re-export so the CLI persists reviewed candidates with the same serializer.
__all__ = [
    "METHOD",
    "PromptSystemRun",
    "prepare_prompt_system",
    "candidate_to_dict",
]
