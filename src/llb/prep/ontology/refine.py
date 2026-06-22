"""Stage 6 -- exact-ground, deduplicate, and reject unsupported or circular drafts.

Grounding reuses `frontier.build_drafted_items` (so the answer span is re-located in the doc by
exact-then-normalized match and dropped if absent -- "unsupported" cannot survive). On top of
that:

  - circular items are rejected: a draft whose question already contains the answer text (a
    give-away) or whose question equals its reference answer teaches nothing about retrieval;
  - duplicates are dropped: the same normalized question on a doc, or the same answer span on a
    doc, is kept once.

The survivors are canonical `GoldItem`s tagged `provenance="ontology-drafted", verified=False`.
"""

import logging
from typing import Any

from llb.goldset.schema import GoldItem, Split
from llb.prep.frontier import build_drafted_items
from llb.prep.ontology.constants import PROVENANCE_KIND
from llb.prep.ontology.models import DocRecord

_LOG = logging.getLogger(__name__)


def _normalize_question(question: str) -> str:
    return " ".join(question.split()).casefold()


def is_circular(question: str, reference_answer: str, span_text: str) -> bool:
    """True when the question gives the answer away (so it tests phrasing, not retrieval)."""
    q_norm = _normalize_question(question)
    if not q_norm:
        return True
    if q_norm == _normalize_question(reference_answer):
        return True
    return _normalize_question(span_text) in q_norm


def refine_drafts(
    docs: list[DocRecord], drafts: list[dict[str, Any]], *, split: Split = "final"
) -> list[GoldItem]:
    """Ground + dedup + reject circular drafts into canonical ontology-drafted gold items."""
    by_id = {doc.doc_id: doc for doc in docs}
    # group drafts per doc so build_drafted_items can re-ground against the right text
    per_doc: dict[str, list[dict[str, Any]]] = {}
    for draft in drafts:
        per_doc.setdefault(str(draft.get("doc_id", "")), []).append(draft)

    seen_questions: set[tuple[str, str]] = set()
    seen_spans: set[tuple[str, int, int]] = set()
    kept: list[GoldItem] = []
    n_circular = 0
    n_dup = 0
    for doc_id, doc_drafts in per_doc.items():
        doc = by_id.get(doc_id)
        if doc is None:
            continue
        grounded = build_drafted_items(
            doc_id,
            doc.text,
            doc_drafts,
            split,
            provenance=PROVENANCE_KIND,
            id_prefix="onto",
        )
        for item in grounded:
            span = item.source_spans[0]
            if is_circular(item.question, item.reference_answer, span.text):
                n_circular += 1
                continue
            q_key = (item.source_doc_id, _normalize_question(item.question))
            s_key = (item.source_doc_id, span.char_start, span.char_end)
            if q_key in seen_questions or s_key in seen_spans:
                n_dup += 1
                continue
            seen_questions.add(q_key)
            seen_spans.add(s_key)
            kept.append(item)
    _LOG.info(
        "[ontology] stage 6: %d items kept (%d circular, %d duplicate rejected)",
        len(kept),
        n_circular,
        n_dup,
    )
    return kept
