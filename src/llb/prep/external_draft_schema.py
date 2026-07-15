"""Focused external draft schema implementation."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from llb.core.contracts.common import ValidationReport
from llb.goldset.schema import GoldItem, Provenance, SourceSpan
from llb.prep.frontier import ground_span
from llb.prep.ontology.constants import DEFAULT_QUESTION_TYPE, QUESTION_TYPES
from llb.prep.ontology.coverage import classify_difficulty
from llb.prep.ontology.models import ItemLabels
from llb.prep.ontology.question_types import classify_question_type

PROVENANCE_EXTERNAL: Provenance = "frontier-drafted"

DATA_CLASSIFICATION_OPEN = "open"

DIFFICULTIES = ("easy", "medium", "hard")

SIDECAR_FILENAME = "external_provenance.json"

GOLDSET_FILENAME = "goldset.jsonl"

CORPUS_DIRNAME = "corpus"

PROVENANCE_FILENAME = "provenance.json"

ITEM_PROVENANCE_FILENAME = "item_provenance.jsonl"

IMPORT_REPORT_FILENAME = "import_report.json"


@dataclass
class ImportReport:
    """Counts and per-row drop/repair reasons for one import (written beside the bundle)."""

    loaded: int = 0
    kept: int = 0
    dropped: list[dict[str, str]] = field(default_factory=list)
    repaired: list[dict[str, str]] = field(default_factory=list)

    def drop(self, item_id: str, reason: str) -> None:
        self.dropped.append({"id": item_id, "reason": reason})

    def repair(self, item_id: str, what: str) -> None:
        self.repaired.append({"id": item_id, "repair": what})

    def to_dict(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "kept": self.kept,
            "counts": {"dropped": len(self.dropped), "repaired": len(self.repaired)},
            "dropped": self.dropped,
            "repaired": self.repaired,
        }


def load_sidecar(sidecar: Path) -> dict[str, Any]:
    """Load + enforce the data-classification sidecar; a missing or non-open sidecar aborts.

    Raises SystemExit (no bundle written) so restricted/private material can never be imported as
    if it were cleared for third-party processing.
    """
    if not Path(sidecar).is_file():
        raise SystemExit(
            f"[import-external-draft] required sidecar {sidecar} is absent; an external draft "
            f"cannot be imported without its {SIDECAR_FILENAME} data-classification record."
        )
    try:
        data = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[import-external-draft] cannot read sidecar {sidecar}: {exc}") from exc
    if not isinstance(data, dict) or data.get("data_classification") != DATA_CLASSIFICATION_OPEN:
        raise SystemExit(
            "[import-external-draft] refused: the sidecar must declare "
            f'data_classification: "{DATA_CLASSIFICATION_OPEN}". Uploading a corpus to an external '
            "AI service publishes it; only explicitly open corpora may be imported."
        )
    return data


def _labels(row: dict[str, Any], question: str, reference: str, span_text: str) -> ItemLabels:
    """Question-type + difficulty for a row: honor valid inline labels, else classify locally."""
    qtype = str(row.get("question_type") or "").strip()
    if qtype not in QUESTION_TYPES:
        qtype = classify_question_type(question, reference) or DEFAULT_QUESTION_TYPE
    difficulty = str(row.get("difficulty") or "").strip()
    if difficulty not in DIFFICULTIES:
        difficulty = classify_difficulty(len(span_text), rare=False)
    return ItemLabels(question_type=qtype, difficulty=difficulty)


def _row_to_item(
    row: dict[str, Any], corpus_texts: dict[str, str], index: int, report: ImportReport
) -> tuple[GoldItem, ItemLabels] | None:
    """Re-ground one Artifact B row into a GoldItem + labels, or None (dropped + reported)."""
    item_id = str(row.get("id") or f"ext-{index:04d}")
    question = str(row.get("question") or "").strip()
    quote = str(row.get("quote") or row.get("reference_answer") or "").strip()
    doc_id = str(row.get("source_doc_id") or "").strip()
    if not question or not quote or not doc_id:
        report.drop(item_id, "missing question, quote, or source_doc_id")
        return None
    text = corpus_texts.get(doc_id)
    if text is None:
        report.drop(item_id, f"source_doc_id {doc_id} is not in the corpus")
        return None
    grounded = ground_span(text, quote)  # exact, then normalized-but-exact fallback
    if grounded is None:
        report.drop(item_id, f"quote is not a verbatim substring of {doc_id}")
        return None
    start, exact = grounded
    if exact != quote:
        report.repair(item_id, "quote re-grounded to exact corpus text")
    reference = str(row.get("reference_answer") or "").strip() or exact
    labels = _labels(row, question, reference, exact)
    item = GoldItem(
        id=item_id,
        lang=str(row.get("lang") or "uk"),
        question=question,
        reference_answer=reference,
        source_doc_id=doc_id,
        source_spans=[
            SourceSpan(doc_id=doc_id, char_start=start, char_end=start + len(exact), text=exact)
        ],
        provenance=PROVENANCE_EXTERNAL,
        verified=False,
        split="final",
    )
    return item, labels


def _label_distribution(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


@dataclass
class ImportResult:
    out_dir: Path
    items: list[GoldItem]
    item_labels: dict[str, ItemLabels]
    report: ImportReport
    validation: ValidationReport
