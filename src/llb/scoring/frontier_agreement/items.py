"""Calibration-worksheet inputs for the frontier-judge agreement lane.

One `AgreementItem` per worksheet row carries what the agreement report needs: the judged
(question, answer, contexts) triple plus the two reference ratings already in the ledger --
the human rating and the local judge's rating. Grounding contexts come from the gold set's
source spans so the frontier judge scores the same evidence the local judge saw, with
retrieval held out of the comparison.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from llb.core.contracts.judging import JudgeInputRecord
from llb.goldset.schema import GoldItem, load_goldset
from llb.judge.calibration_worksheet import load_worksheet

_LOG = logging.getLogger(__name__)

GOLD_CONTEXT_WINDOW_CHARS = 1200
"""Source-document chars laid around each gold span as the judge's grounding context."""

CORPUS_SUBDIR = "corpus"
"""Conventional corpus directory beside a gold set JSONL."""


@dataclass(frozen=True)
class AgreementItem:
    """One judgeable calibration row plus the ratings it is compared against."""

    item_id: str
    question: str
    answer: str
    contexts: list[str]
    human_rating: float | None
    local_rating: float | None

    def judge_record(self) -> JudgeInputRecord:
        """The record handed to any judge lane (frontier or local)."""
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": list(self.contexts),
        }


def _optional_float(raw: str) -> float | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _window(text: str, char_start: int, char_end: int, window: int) -> str:
    """Return `window` chars of `text` centered on the span, clipped to the document."""
    span_len = max(0, char_end - char_start)
    pad = max(0, (window - span_len) // 2)
    start = max(0, char_start - pad)
    end = min(len(text), char_end + pad)
    return text[start:end].strip()


def _doc_text(corpus_root: Path, doc_id: str, cache: dict[str, str | None]) -> str | None:
    if doc_id not in cache:
        path = corpus_root / doc_id
        cache[doc_id] = path.read_text(encoding="utf-8") if path.is_file() else None
        if cache[doc_id] is None:
            _LOG.warning("[frontier-judge] missing corpus doc %s under %s", doc_id, corpus_root)
    return cache[doc_id]


def _gold_contexts(
    item: GoldItem,
    corpus_root: Path | None,
    cache: dict[str, str | None],
    window: int,
) -> list[str]:
    """Grounding contexts for one gold item: span windows when the corpus resolves, else spans."""
    contexts: list[str] = []
    for span in item.source_spans:
        text = None if corpus_root is None else _doc_text(corpus_root, span.doc_id, cache)
        chunk = span.text if text is None else _window(text, span.char_start, span.char_end, window)
        if chunk and chunk not in contexts:
            contexts.append(chunk)
    return contexts


def resolve_corpus_root(goldset: Path, corpus_root: Path | None) -> Path | None:
    """Explicit corpus root, else the conventional `corpus/` beside the gold set."""
    if corpus_root is not None:
        return corpus_root
    candidate = Path(goldset).parent / CORPUS_SUBDIR
    return candidate if candidate.is_dir() else None


def load_agreement_items(
    worksheet: Path,
    *,
    goldset: Path | None = None,
    corpus_root: Path | None = None,
    window: int = GOLD_CONTEXT_WINDOW_CHARS,
    limit: int | None = None,
) -> list[AgreementItem]:
    """Load judgeable calibration rows from a filled worksheet.

    Rows without a `model_answer` are skipped: there is nothing for a judge to score, and an
    empty answer would enter every correlation as a constant zero. When `goldset` is omitted
    the reference answer stands in as the only context, which weakens faithfulness judging --
    pass the gold set whenever the agreement evidence is meant to be durable.
    """
    rows, _ = load_worksheet(Path(worksheet))
    gold_by_id: dict[str, GoldItem] = {}
    resolved_root: Path | None = None
    if goldset is not None:
        gold_by_id = {item.id: item for item in load_goldset(goldset)}
        resolved_root = resolve_corpus_root(Path(goldset), corpus_root)
        if resolved_root is None:
            _LOG.warning("[frontier-judge] no corpus root; grounding on gold span text alone")
    else:
        _LOG.warning("[frontier-judge] no gold set; grounding on the reference answer alone")

    cache: dict[str, str | None] = {}
    items: list[AgreementItem] = []
    skipped = 0
    for row in rows:
        answer = (row.get("model_answer") or "").strip()
        if not answer:
            skipped += 1
            continue
        item_id = row.get("item_id", "")
        gold = gold_by_id.get(item_id)
        if gold is not None:
            contexts = _gold_contexts(gold, resolved_root, cache, window)
        else:
            contexts = [c for c in [(row.get("reference_answer") or "").strip()] if c]
        items.append(
            AgreementItem(
                item_id=item_id,
                question=(row.get("question") or "").strip(),
                answer=answer,
                contexts=contexts,
                human_rating=_optional_float(row.get("human_rating", "")),
                local_rating=_optional_float(row.get("judge_rating", "")),
            )
        )
    if skipped:
        _LOG.info("[frontier-judge] skipped %d worksheet rows without a model_answer", skipped)
    if limit is not None:
        items = items[:limit]
    return items
