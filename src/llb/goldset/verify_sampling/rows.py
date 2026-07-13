"""Gold-item and chain worksheet-row construction."""

import json
from collections.abc import Sequence
from pathlib import Path

from llb.goldset.chains import ChainItem, chain_stratum_key
from llb.goldset.schema import GoldItem
from llb.goldset.verify_base import CORPUS_DIRNAME, KIND_CHAINS, KIND_GOLDSET
from llb.goldset.verify_sampling.context import (
    corpus_text,
    corpus_window,
    load_cross_check,
    load_retrieval_ranks,
    page_citation_for_span,
)
from llb.goldset.verify_sampling.strata import stratum_key

PageCache = dict[str, tuple[str | None, list[dict[str, object]]] | None]


def gold_row(
    item: GoldItem,
    corpus_root: Path,
    cache: dict[str, str | None],
    verdict: dict[str, object],
    *,
    synthetic: bool,
    retrieval_rank: int | None = None,
    page_cache: PageCache | None = None,
) -> dict[str, str]:
    span = item.source_spans[0]
    text = corpus_text(corpus_root, span.doc_id, cache)
    context = (
        corpus_window(text, span.char_start, span.char_end) if text is not None else "(missing doc)"
    )
    page = (
        page_citation_for_span(corpus_root, span.doc_id, span.char_start, span.char_end, page_cache)
        if page_cache is not None
        else ""
    )

    def flag(key: str) -> str:
        value = verdict.get(key)
        return "" if value is None else ("true" if value else "false")

    return {
        "item_kind": KIND_GOLDSET,
        "item_id": item.id,
        "provenance": item.provenance,
        "split": item.split,
        "source_doc_id": item.source_doc_id,
        "synthetic": "true" if synthetic else "false",
        "stratum": stratum_key(item),
        "question": item.question,
        "reference_answer": item.reference_answer,
        "span_doc_id": span.doc_id,
        "span_text": span.text,
        "context": context,
        "retrieval_rank": "" if retrieval_rank is None else str(retrieval_rank),
        "page_citation": page,
        "chain_steps": "",
        "cc_grounded": flag("grounded"),
        "cc_non_circular": flag("non_circular"),
        "cc_supported": flag("supported"),
        "cc_answerable": flag("answerable"),
        "cc_note": str(verdict.get("note", "")),
    }


def chain_step_contexts(
    chain: ChainItem, corpus_root: Path, cache: dict[str, str | None], page_cache: PageCache
) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    for step in chain.steps:
        span = step.source_spans[0]
        text = corpus_text(corpus_root, span.doc_id, cache)
        context = (
            corpus_window(text, span.char_start, span.char_end)
            if text is not None
            else "(missing doc)"
        )
        steps.append(
            {
                "order": str(step.order),
                "question": step.question,
                "reference_answer": step.reference_answer,
                "dependency_note": step.dependency_note,
                "span_doc_id": span.doc_id,
                "span_text": span.text,
                "context": context,
                "page_citation": page_citation_for_span(
                    corpus_root, span.doc_id, span.char_start, span.char_end, page_cache
                ),
            }
        )
    return steps


def chain_row(
    chain: ChainItem, corpus_root: Path, cache: dict[str, str | None], page_cache: PageCache
) -> dict[str, str]:
    steps = chain_step_contexts(chain, corpus_root, cache, page_cache)
    final = steps[-1] if steps else {}
    return {
        "item_kind": KIND_CHAINS,
        "item_id": chain.chain_id,
        "provenance": chain.provenance,
        "split": chain.split,
        "source_doc_id": chain.steps[0].source_doc_id if chain.steps else "",
        "synthetic": "false",
        "stratum": chain_stratum_key(chain),
        "question": " -> ".join(step.question for step in chain.steps),
        "reference_answer": chain.steps[-1].reference_answer if chain.steps else "",
        "span_doc_id": final.get("span_doc_id", ""),
        "span_text": final.get("span_text", ""),
        "context": final.get("context", ""),
        "retrieval_rank": "",
        "page_citation": final.get("page_citation", ""),
        "chain_steps": json.dumps(steps, ensure_ascii=False),
    }


def sample_gold_rows(
    bundle: Path, sample: Sequence[GoldItem], *, synthetic: bool
) -> list[dict[str, str]]:
    verdicts = load_cross_check(bundle)
    ranks = load_retrieval_ranks(bundle)
    corpus_root = bundle / CORPUS_DIRNAME
    cache: dict[str, str | None] = {}
    page_cache: PageCache = {}
    return [
        gold_row(
            item,
            corpus_root,
            cache,
            verdicts.get(item.id, {}),
            synthetic=synthetic,
            retrieval_rank=ranks.get(item.id),
            page_cache=page_cache,
        )
        for item in sample
    ]


def sample_chain_rows(bundle: Path, sample: Sequence[ChainItem]) -> list[dict[str, str]]:
    corpus_root = bundle / CORPUS_DIRNAME
    cache: dict[str, str | None] = {}
    page_cache: PageCache = {}
    return [chain_row(chain, corpus_root, cache, page_cache) for chain in sample]
