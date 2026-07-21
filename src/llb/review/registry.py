"""Conservative path-based adapter detection for ``llb review``."""

import csv
import json
from pathlib import Path

from llb.review.adapters import (
    ConflictResolutionAdapter,
    DraftCompareAdapter,
    ExternalRagAdapter,
    GoldsetVerifyAdapter,
    JudgeCalibrationAdapter,
    KnowledgeCutoffAdapter,
    PromptSystemAdapter,
)
from llb.review.core import ReviewAdapter


def open_review(path: Path | str) -> ReviewAdapter:
    """Open the one adapter whose existing ledger signature matches ``path``."""
    value = Path(path)
    if value.is_dir():
        if (value / "comparison.json").is_file():
            return DraftCompareAdapter(value)
        if (value / "translation_review.csv").is_file():
            return KnowledgeCutoffAdapter(value)
        if (value / "candidates.json").is_file():
            return PromptSystemAdapter(value)
        raise ValueError(f"cannot detect a review ledger in directory: {value}")
    if not value.is_file():
        raise ValueError(f"review path not found: {value}")
    if value.name == "comparison.json":
        return DraftCompareAdapter(value)
    if value.name == "candidates.json" or _is_candidate_json(value):
        return PromptSystemAdapter(value)
    if value.suffix.lower() == ".csv":
        fields = _csv_fields(value)
        if "human_rating" in fields and "model_answer" in fields:
            return JudgeCalibrationAdapter(value)
        if "decision" in fields and "item_id" in fields:
            if "review_profile" in fields and _translation_profile(value):
                return KnowledgeCutoffAdapter(value)
            return GoldsetVerifyAdapter(value)
    if value.suffix.lower() == ".jsonl":
        if _is_conflict_resolution(value):
            return ConflictResolutionAdapter(value)
        if _is_external_rag(value):
            return ExternalRagAdapter(value)
    raise ValueError(f"unrecognized review ledger: {value}")


def _csv_fields(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def _translation_profile(path: Path) -> bool:
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle), {})
    return row.get("review_profile") == "knowledge-cutoff-translation"


def _is_candidate_json(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(payload, list)
        and payload
        and isinstance(payload[0], dict)
        and "prompt_system_id" in payload[0]
    )


def _is_external_rag(path: Path) -> bool:
    try:
        line = next(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        record = json.loads(line)
    except (OSError, StopIteration, json.JSONDecodeError):
        return False
    answer_fields = ("llm_answer", "predicted_answer", "model_answer", "answer")
    return (
        isinstance(record, dict)
        and "question" in record
        and any(field in record for field in answer_fields)
    )


def _is_conflict_resolution(path: Path) -> bool:
    try:
        line = next(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        record = json.loads(line)
    except (OSError, StopIteration, json.JSONDecodeError):
        return False
    return isinstance(record, dict) and record.get("review_type") == "corpus_conflict_resolution"
