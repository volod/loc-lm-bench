"""Build canonical chain-of-questions items from ontology graph paths."""

import logging

from llb.goldset.chains import ChainItem, ChainStep
from llb.goldset.schema import SourceSpan, Split
from llb.prep.frontier import ground_span
from llb.prep.ontology.constants import CHAIN_ID_PREFIX, PROVENANCE_KIND
from llb.prep.ontology.models import DocRecord, MultiHopSeed, MultiHopStep

_LOG = logging.getLogger(__name__)


def _ground_step_span(doc_texts: dict[str, str], step: MultiHopStep) -> SourceSpan | None:
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


def _question_for_step(step: MultiHopStep, *, order: int) -> str:
    if order == 1:
        return f'Який факт у джерелі пов\'язує "{step.subject}" і "{step.object}"?'
    return (
        f"З урахуванням попереднього кроку, який факт у джерелі пов'язує "
        f'"{step.subject}" і "{step.object}"?'
    )


def _dependency_for_step(seed: MultiHopSeed, *, order: int) -> str:
    if order == 1:
        return ""
    previous = seed.steps[order - 2]
    return (
        f'Крок {order - 1} задає контекст для теми "{seed.bridge}" через зв\'язок '
        f'"{previous.subject}" -> "{previous.object}".'
    )


def build_chain_items(
    docs: list[DocRecord],
    seeds: list[MultiHopSeed],
    *,
    split: Split = "final",
) -> list[ChainItem]:
    """Turn graph-path seeds into exact-grounded chain items.

    The model-facing multi-hop drafter produces one flat QA item. This builder instead keeps
    each hop as its own reviewable step so the context-policy benchmark can test stepwise
    context accumulation. A seed is dropped unless every hop re-grounds to a distinct span.
    """
    doc_texts = {doc.doc_id: doc.text for doc in docs}
    chains: list[ChainItem] = []
    n_dropped = 0
    for i, seed in enumerate(seeds):
        steps: list[ChainStep] = []
        seen_spans: set[tuple[str, int, int]] = set()
        for order, hop in enumerate(seed.steps, 1):
            span = _ground_step_span(doc_texts, hop)
            if span is None:
                break
            key = (span.doc_id, span.char_start, span.char_end)
            if key in seen_spans:
                break
            seen_spans.add(key)
            steps.append(
                ChainStep(
                    order=order,
                    question=_question_for_step(hop, order=order),
                    reference_answer=span.text,
                    source_doc_id=span.doc_id,
                    source_spans=[span],
                    dependency_note=_dependency_for_step(seed, order=order),
                )
            )
        if len(steps) != len(seed.steps):
            n_dropped += 1
            continue
        chains.append(
            ChainItem(
                chain_id=f"{CHAIN_ID_PREFIX}-{i:04d}",
                steps=steps,
                provenance=PROVENANCE_KIND,
                verified=False,
                split=split,
            )
        )
    _LOG.info("[ontology] chains: %d chain items kept (%d dropped)", len(chains), n_dropped)
    return chains
