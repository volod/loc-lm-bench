"""Shared constants, result dataclasses, and value coercions for external-RAG scoring (leaf).

The scoring core (`external_rag.py`), the summary aggregation (`external_rag_summary.py`), and the
Markdown report renderer (`external_rag_report.py`) all build on this module. It depends on nothing
else in the family, so it carries no import cycle. `external_rag.py` re-exports the public names
here so `llb.scoring.external_rag.<name>` keeps working.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

ANSWER_FIELD_CANDIDATES = ("llm_answer", "predicted_answer", "model_answer", "answer")
ERROR_FIELD_CANDIDATES = ("llm_error", "error")
MODEL_FIELD_CANDIDATES = ("llm_model", "model", "service")
PROVIDER_FIELD_CANDIDATES = ("llm_provider", "provider")
ROUTE_FIELD_CANDIDATES = ("llm_route", "route")
SOURCES_FIELD_CANDIDATES = ("llm_sources", "sources", "retrieved_sources")
DEFAULT_SOURCE_LIMIT = 3
ANSWER_PREVIEW_CHARS = 180
WORST_CASE_REPORT_LIMIT = 12
SOURCE_REPORT_LIMIT = 8
HUMAN_SCORE_FIELD = "human_score_0_1"
HUMAN_DECISION_FIELD = "human_decision"
HUMAN_NOTES_FIELD = "human_notes"
HUMAN_CORRECTED_ANSWER_FIELD = "human_corrected_answer"
HUMAN_STATUS_FIELD = "human_status"
HUMAN_FIELDS = (
    HUMAN_SCORE_FIELD,
    HUMAN_DECISION_FIELD,
    HUMAN_NOTES_FIELD,
    HUMAN_CORRECTED_ANSWER_FIELD,
    HUMAN_STATUS_FIELD,
)
HUMAN_STATUS_SCORED = "scored"
HUMAN_DECISION_ACCEPT = "accept"
HUMAN_DECISION_PARTIAL = "partial"
HUMAN_DECISION_REJECT = "reject"
HUMAN_DECISIONS = (HUMAN_DECISION_ACCEPT, HUMAN_DECISION_PARTIAL, HUMAN_DECISION_REJECT)

STATUS_ERROR = "error"
STATUS_EMPTY = "empty"
STATUS_ABSTAINED = "abstained"
STATUS_REFUSAL = "refusal"
STATUS_OK = "ok"

ABSTENTION_MARKERS = (
    "context does not contain",
    "not found",
    "no relevant",
    "unknown",
    "немає відповіді",
    "не знайдено",
    "не містить",
    "контекст не містить",
    "у базі знань немає",
    "невідомо",
)

_SOURCE_FOOTER_RE = re.compile(
    r"(?im)^\s*(джерело|джерела|dzherelo|dzherela|source|sources)\s*:\s*.*\Z",
    re.DOTALL,
)


@dataclass(frozen=True)
class ExternalRagPaths:
    """Artifacts written by `score_external_rag_file`."""

    csv: Path
    report: Path


@dataclass(frozen=True)
class ExternalRagResult:
    """In-memory result returned after scoring an answered export."""

    rows: list[dict[str, object]]
    summary: dict[str, object]
    paths: ExternalRagPaths


def _string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int | float | bool):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _round(value: float) -> float:
    return round(float(value), 4)


def _as_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value:
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _ellipsize(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)] + "..."
