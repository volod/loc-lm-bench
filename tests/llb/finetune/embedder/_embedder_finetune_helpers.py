"""A tiny corpus + gold set whose spans really index it, shared by the embedder fine-tune tests.

Pure: the documents are short Ukrainian paragraphs and every gold span is computed from the text
by `str.index`, so a chunker change moves the chunks without invalidating the labels.
"""

import json
from pathlib import Path

from llb.goldset.schema import GoldItem

DOCS = {
    "ua/norm.txt": (
        "Нормандія розташована у Франції. Нормани дали назву цьому регіону у десятому столітті. "
        "Регіон лежить на узбережжі Ла-Маншу."
    ),
    "ua/calif.txt": (
        "Каліфорнія розташована на заході США. Лос-Анджелес і Сан-Дієго -- найбільші міста штату. "
        "На сході лежить Колорадська пустеля."
    ),
    "ua/complex.txt": (
        "Обчислювальна складність вивчає ресурси алгоритмів. Клас P містить задачі, "
        "розв'язні за поліноміальний час."
    ),
}


def write_corpus(root: Path) -> Path:
    """Write the fixture documents under `root/corpus` and return that directory."""
    corpus = root / "corpus"
    for doc_id, text in DOCS.items():
        path = corpus / doc_id
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return corpus


def gold_item(item_id: str, doc_id: str, question: str, evidence: str, split: str) -> GoldItem:
    """A verified gold item whose single span is `evidence` at its real offset in `doc_id`."""
    text = DOCS[doc_id]
    start = text.index(evidence)
    return GoldItem(
        id=item_id,
        question=question,
        reference_answer=evidence,
        source_doc_id=doc_id,
        source_spans=[
            {
                "doc_id": doc_id,
                "char_start": start,
                "char_end": start + len(evidence),
                "text": evidence,
            }
        ],
        provenance="human-authored",
        verified=True,
        split=split,
    )


def default_items() -> list[GoldItem]:
    """One item per split, so a leak is visible as a wrong id rather than a wrong count."""
    return [
        gold_item(
            "tuning-1",
            "ua/norm.txt",
            "Де розташована Нормандія?",
            "Нормандія розташована у Франції",
            "tuning",
        ),
        gold_item(
            "tuning-2",
            "ua/calif.txt",
            "Які міста найбільші у Каліфорнії?",
            "Лос-Анджелес і Сан-Дієго",
            "tuning",
        ),
        gold_item(
            "calibration-1",
            "ua/complex.txt",
            "Що вивчає обчислювальна складність?",
            "ресурси алгоритмів",
            "calibration",
        ),
        gold_item(
            "final-1",
            "ua/complex.txt",
            "Який клас містить поліноміальні задачі?",
            "Клас P",
            "final",
        ),
    ]


def write_goldset(root: Path, items: list[GoldItem] | None = None) -> Path:
    """Write a gold set JSONL beside the corpus and return its path."""
    path = root / "goldset.jsonl"
    rows = items if items is not None else default_items()
    path.write_text(
        "".join(json.dumps(item.model_dump(), ensure_ascii=False) + "\n" for item in rows),
        encoding="utf-8",
    )
    return path
