"""Draft + ground multi-hop chain questions from graph-path seeds (yield-max).

Each `MultiHopSeed` is a 2-hop `A -r1-> B -r2-> C` chain with an exact evidence span per hop. The
drafter is shown both facts and asked for ONE question that needs both; the two evidence spans
become the item's grounded source spans, so a multi-hop item carries >= MULTI_HOP_MIN_SPANS spans
and passes span-exact validation by construction. The reference answer must also contain the
verbatim bridge or end entity, which makes the answer itself span-checkable against the chain.
Grounding reuses the same exact-then-normalized match as the flat drafter, so a span is re-verified
against the copied corpus doc before it is emitted.
"""

import json
import logging
from typing import Any

from llb.goldset.schema import GoldItem, SourceSpan, Split
from llb.prep.frontier import ground_span, parse_json_block
from llb.prep.frontier_telemetry import DraftBudgetExceeded, LLMComplete
from llb.prep.ontology.constants import (
    DRAFT_CONTEXT_RADIUS,
    MULTI_HOP_DIFFICULTY,
    MULTI_HOP_ID_PREFIX,
    MULTI_HOP_MIN_SPANS,
    PROVENANCE_KIND,
    QUESTION_TYPE_MULTI_HOP,
)
from llb.prep.ontology.draft import context_window
from llb.prep.ontology.language import is_ukrainian_dominant
from llb.prep.ontology.models import DocRecord, ItemLabels, MultiHopSeed, MultiHopStep
from llb.prompts.registry import render_text

_LOG = logging.getLogger(__name__)


def _chain_line(seed: MultiHopSeed) -> str:
    return (
        "; ".join(f"{step.subject} -- {step.relation} -> {step.object}" for step in seed.steps)
        + "."
    )


def chain_context(doc_texts: dict[str, str], seed: MultiHopSeed) -> str:
    """Concatenate a bounded context window around each hop's evidence span (deduplicated)."""
    windows: list[str] = []
    seen: set[tuple[str, int, int]] = set()
    for step in seed.steps:
        text = doc_texts.get(step.evidence.doc_id)
        if text is None:
            continue
        key = (step.evidence.doc_id, step.evidence.char_start, step.evidence.char_end)
        if key in seen:
            continue
        seen.add(key)
        windows.append(
            context_window(
                text, step.evidence.char_start, step.evidence.char_end, DRAFT_CONTEXT_RADIUS
            )
        )
    return "\n---\n".join(windows)


def multi_hop_prompt(seed: MultiHopSeed, context: str) -> str:
    """One multi-hop UA question whose answer needs both facts of the chain."""
    return render_text(
        "prep.ontology.multi_hop",
        {"chain_line": _chain_line(seed), "context": context},
    )


def draft_multi_hop(
    complete: LLMComplete, docs: list[DocRecord], seeds: list[MultiHopSeed]
) -> list[dict[str, Any] | None]:
    """Draft one raw multi-hop QA dict per seed (None on failure), aligned index-for-index."""
    doc_texts = {doc.doc_id: doc.text for doc in docs}
    drafts: list[dict[str, Any] | None] = []
    for seed in seeds:
        context = chain_context(doc_texts, seed)
        try:
            payload = parse_json_block(complete(multi_hop_prompt(seed, context)))
        except json.JSONDecodeError:
            _LOG.warning("[ontology] unparseable multi-hop draft; skipping")
            drafts.append(None)
            continue
        except DraftBudgetExceeded:
            raise
        except Exception as exc:  # endpoint/transport error -> skip this seed, keep going
            _LOG.warning("[ontology] multi-hop draft call failed: %s", exc)
            drafts.append(None)
            continue
        drafts.append(payload if isinstance(payload, dict) else None)
    return drafts


def _ground_step_span(doc_texts: dict[str, str], step: MultiHopStep) -> SourceSpan | None:
    """Re-verify a hop's evidence against the copied corpus doc; drop it if it no longer resolves."""
    text = doc_texts.get(step.evidence.doc_id)
    if text is None:
        return None
    exact = text[step.evidence.char_start : step.evidence.char_end]
    if exact == step.evidence.text:
        return SourceSpan(**step.evidence.model_dump())
    grounded = ground_span(text, step.evidence.text)
    if grounded is None:
        return None
    start, exact_text = grounded
    return SourceSpan(
        doc_id=step.evidence.doc_id,
        char_start=start,
        char_end=start + len(exact_text),
        text=exact_text,
    )


def build_multi_hop_items(
    docs: list[DocRecord],
    seeds: list[MultiHopSeed],
    drafts: list[dict[str, Any] | None],
    *,
    split: Split = "final",
) -> tuple[list[GoldItem], dict[str, ItemLabels]]:
    """Turn drafted multi-hop chains into grounded, multi-span gold items + their labels.

    An item is emitted only when the draft has a question and reference answer AND at least
    `MULTI_HOP_MIN_SPANS` of its hops re-ground to distinct exact spans -- otherwise the chain is
    dropped (never mis-grounded). Labels tag every survivor `multi-hop` / hard.
    """
    doc_texts = {doc.doc_id: doc.text for doc in docs}
    items: list[GoldItem] = []
    labels: dict[str, ItemLabels] = {}
    n_dropped = 0
    for i, (seed, draft) in enumerate(zip(seeds, drafts)):
        item = _multi_hop_item(doc_texts, seed, draft, i, split)
        if item is None:
            n_dropped += 1
            continue
        items.append(item)
        labels[item.id] = ItemLabels(
            question_type=QUESTION_TYPE_MULTI_HOP, difficulty=MULTI_HOP_DIFFICULTY
        )
    _LOG.info("[ontology] multi-hop: %d chain items kept (%d dropped)", len(items), n_dropped)
    return items, labels


def _distinct_step_spans(doc_texts: dict[str, str], seed: MultiHopSeed) -> list[SourceSpan]:
    """The seed's hop spans that re-ground exactly, with byte-identical repeats removed."""
    spans: list[SourceSpan] = []
    span_keys: set[tuple[str, int, int]] = set()
    for step in seed.steps:
        span = _ground_step_span(doc_texts, step)
        if span is None:
            continue
        key = (span.doc_id, span.char_start, span.char_end)
        if key in span_keys:
            continue
        span_keys.add(key)
        spans.append(span)
    return spans


def _multi_hop_item(
    doc_texts: dict[str, str],
    seed: MultiHopSeed,
    draft: dict[str, Any] | None,
    index: int,
    split: Split,
) -> GoldItem | None:
    """One grounded multi-span GoldItem, or None when the draft/grounding is insufficient."""
    if draft is None:
        return None
    question = str(draft.get("question", "")).strip()
    reference = str(draft.get("reference_answer", "")).strip()
    if not question or not reference:
        return None
    if not all(is_ukrainian_dominant(text) for text in (question, reference)):
        return None
    if not any(entity and entity in reference for entity in (seed.bridge, seed.end)):
        return None
    spans = _distinct_step_spans(doc_texts, seed)
    if len(spans) < MULTI_HOP_MIN_SPANS:
        return None
    return GoldItem(
        id=f"{spans[0].doc_id}-{MULTI_HOP_ID_PREFIX}-{index}",
        question=question,
        reference_answer=reference,
        source_doc_id=spans[0].doc_id,
        source_spans=spans,
        provenance=PROVENANCE_KIND,
        verified=False,
        split=split,
    )
