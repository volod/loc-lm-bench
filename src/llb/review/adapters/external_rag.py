"""External-RAG answer JSONL adapter."""

from pathlib import Path
from typing import Any

from llb.scoring.external_rag.records import ensure_human_fields, load_jsonl, write_jsonl
from llb.scoring.external_rag_common import (
    HUMAN_CORRECTED_ANSWER_FIELD,
    HUMAN_DECISION_ACCEPT,
    HUMAN_DECISION_FIELD,
    HUMAN_DECISION_PARTIAL,
    HUMAN_DECISION_REJECT,
    HUMAN_NOTES_FIELD,
    HUMAN_SCORE_FIELD,
)
from llb.scoring.external_rag_session.records import _clear_row, _set_decision
from llb.review.core import ReviewAction, ReviewAdapter, ReviewRecord
from llb.review.presentation import fields_section, json_section

_ACTIONS = (
    ReviewAction("a", "Accept", HUMAN_DECISION_ACCEPT, "positive"),
    ReviewAction("p", "Partial", HUMAN_DECISION_PARTIAL, "warning"),
    ReviewAction("r", "Reject", HUMAN_DECISION_REJECT, "negative"),
    ReviewAction("c", "Clear", "clear", "neutral"),
)


class ExternalRagAdapter(ReviewAdapter):
    """Use the established JSONL field initializer and canonical writer."""

    kind = "external-rag"

    def __init__(self, answers: Path | str) -> None:
        self.path = Path(answers)
        self.records = load_jsonl(self.path)
        if ensure_human_fields(self.records):
            write_jsonl(self.path, self.records)

    @property
    def actions(self) -> tuple[ReviewAction, ...]:
        return _ACTIONS

    def __len__(self) -> int:
        return len(self.records)

    def record(self, index: int) -> ReviewRecord:
        row = self.records[index]
        item_id = _value(row.get("id")) or str(index + 1)
        sources = row.get("llm_sources") or row.get("sources") or row.get("retrieved_sources")
        return ReviewRecord(
            key=item_id,
            title=f"external RAG: {item_id}",
            sections=(
                fields_section(
                    "Record content",
                    row,
                    (
                        "question",
                        "reference_answer",
                        "llm_answer",
                        "predicted_answer",
                        HUMAN_CORRECTED_ANSWER_FIELD,
                    ),
                    "data",
                ),
                json_section(
                    "Evidence",
                    {"source_spans": row.get("source_spans"), "returned_sources": sources},
                    "evidence",
                ),
                fields_section(
                    "Metadata",
                    row,
                    (
                        "split",
                        "source_doc_id",
                        "verified",
                        HUMAN_SCORE_FIELD,
                        HUMAN_DECISION_FIELD,
                        HUMAN_NOTES_FIELD,
                    ),
                    "metadata",
                ),
            ),
            stratum=_value(row.get("split")) or "all",
            verdict=_value(row.get(HUMAN_DECISION_FIELD)),
        )

    def apply(self, index: int, action: str) -> None:
        row = self.records[index]
        if action == "clear":
            _clear_row(row)
        elif action in (
            HUMAN_DECISION_ACCEPT,
            HUMAN_DECISION_PARTIAL,
            HUMAN_DECISION_REJECT,
        ):
            _set_decision(row, action)
        else:
            raise ValueError(f"unsupported {self.kind} action: {action}")
        write_jsonl(self.path, self.records)


def _value(value: Any) -> str:
    return "" if value is None else str(value).strip()
