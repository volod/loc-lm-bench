"""Focused squad records implementation."""

import ast
import hashlib
import json
import logging
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import SquadAnswers, SquadRecord
from llb.goldset.schema import GoldItem, Provenance
from llb.goldset.splits import assign_splits

_LOG = logging.getLogger(__name__)


def _doc_id(context: str) -> str:
    digest = hashlib.sha1(context.encode("utf-8")).hexdigest()[:12]
    return f"squad/{digest}.txt"


def _flatten_squad_paragraph(context: str, qa: Mapping[str, Any]) -> SquadRecord:
    answers = qa.get("answers", [])
    return {
        "id": qa.get("id"),
        "context": context,
        "question": qa["question"],
        "answers": {
            "text": [a["text"] for a in answers],
            "answer_start": [a.get("answer_start") for a in answers],
        },
    }


def _flatten_squad_articles(data: object) -> list[SquadRecord]:
    if isinstance(data, dict):
        articles = [data]
    elif isinstance(data, list):
        articles = data
    else:
        raise ValueError("unrecognized SQuAD data field (expected an article or list)")
    records: list[SquadRecord] = []
    for article in articles:
        for para in article.get("paragraphs", []):
            context = para["context"]
            for qa in para.get("qas", []):
                records.append(_flatten_squad_paragraph(context, qa))
    return records


def normalize(raw: object) -> list[SquadRecord]:
    """Accept flattened records or nested SQuAD; return a list of flattened records."""
    if isinstance(raw, dict) and "data" in raw:
        return _flatten_squad_articles(raw["data"])
    if isinstance(raw, list):
        return cast(list[SquadRecord], raw)
    raise ValueError("unrecognized SQuAD JSON shape (expected a list or a {'data': ...} object)")


def _answer_char_start(context: str, answer: str, starts: list[int | None]) -> int | None:
    start = starts[0] if starts else None
    if isinstance(start, int) and context[start : start + len(answer)] == answer:
        return start
    found = context.find(answer)
    return found if found >= 0 else None


def _squad_record_to_gold(
    rec: SquadRecord,
    index: int,
    *,
    lang: str,
    provenance: Provenance,
    verified: bool,
) -> tuple[str, str, JsonObject] | None:
    """Map one SQuAD record to (doc_id, context, raw gold item dict), or None if skipped."""
    context = rec["context"]
    answers = rec.get("answers") or {}
    texts = answers.get("text") or []
    if not texts:
        return None
    answer = texts[0]
    start = _answer_char_start(context, answer, answers.get("answer_start") or [])
    if start is None:
        return None
    doc_id = _doc_id(context)
    item_id = str(rec.get("id") or f"squad-uk-{index:05d}")
    raw_item: JsonObject = {
        "id": item_id,
        "lang": lang,
        "question": rec["question"],
        "reference_answer": answer,
        "source_doc_id": doc_id,
        "source_spans": [
            {
                "doc_id": doc_id,
                "char_start": start,
                "char_end": start + len(answer),
                "text": answer,
            }
        ],
        "provenance": provenance,
        "verified": verified,
    }
    return doc_id, context, raw_item


def squad_to_gold(
    records: list[SquadRecord],
    lang: str = "uk",
    provenance: Provenance = "public-reused",
    verified: bool = False,
    max_items: int | None = None,
    seed: int = 13,
) -> tuple[dict[str, str], list[GoldItem], int]:
    docs: dict[str, str] = {}
    raw_items: list[JsonObject] = []
    skipped = 0
    for i, rec in enumerate(records):
        if max_items is not None and len(raw_items) >= max_items:
            break
        mapped = _squad_record_to_gold(rec, i, lang=lang, provenance=provenance, verified=verified)
        if mapped is None:
            skipped += 1
            continue
        doc_id, context, raw_item = mapped
        docs[doc_id] = context
        raw_items.append(raw_item)
    split_map = assign_splits([r["id"] for r in raw_items], seed=seed)
    items = [GoldItem.model_validate({**r, "split": split_map[r["id"]]}) for r in raw_items]
    return docs, items, skipped


def load_squad_json(path: Path) -> list[SquadRecord]:
    return normalize(json.loads(Path(path).read_text(encoding="utf-8")))


def coerce_answers(row: Mapping[str, Any]) -> SquadAnswers:
    """Normalize a row's answers into {text: [...], answer_start: [...]}.

    Handles three shapes seen in the wild: a real dict (HF squad), a dict serialized as
    a Python-repr string (HPLT/ua-squad), and flat answer_text/answer_start columns.
    """
    ans = row.get("answers")
    if isinstance(ans, str):
        try:
            ans = ast.literal_eval(ans)
        except (ValueError, SyntaxError):
            ans = None
    if isinstance(ans, dict) and ans.get("text"):
        return {
            "text": list(ans.get("text") or []),
            "answer_start": list(ans.get("answer_start") or []),
        }
    text = row.get("answer_text")
    if isinstance(text, str) and text:
        start = row.get("answer_start")
        starts: list[int | None]
        try:
            starts = (
                [int(float(start))] if isinstance(start, str | int | float) and start != "" else []
            )
        except (TypeError, ValueError):
            starts = []
        return {"text": [text], "answer_start": starts}
    return {"text": [], "answer_start": []}


def hf_rows_to_records(rows: Iterable[Mapping[str, Any]], dataset_id: str) -> Iterator[SquadRecord]:
    """Flatten either regular QA rows or article rows from a nested SQuAD dataset."""
    for i, row in enumerate(rows):
        if "data" in row:
            yield from normalize({"data": row["data"]})
            continue
        if "context" not in row or "question" not in row:
            continue
        yield {
            "id": row.get("id") or f"{dataset_id.replace('/', '_')}-{i:05d}",
            "context": row["context"],
            "question": row["question"],
            "answers": coerce_answers(row),
        }
