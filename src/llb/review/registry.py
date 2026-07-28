"""Conservative path-based adapter detection for ``llb review``."""

import csv
import json
from collections.abc import Callable
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


def _open_review_directory(path: Path) -> ReviewAdapter:
    signatures: tuple[tuple[str, Callable[[Path], ReviewAdapter]], ...] = (
        ("comparison.json", DraftCompareAdapter),
        ("translation_review.csv", KnowledgeCutoffAdapter),
        ("candidates.json", PromptSystemAdapter),
    )
    for filename, adapter in signatures:
        if (path / filename).is_file():
            return adapter(path)
    raise ValueError(f"cannot detect a review ledger in directory: {path}")


def _open_csv_review(path: Path) -> ReviewAdapter | None:
    if path.suffix.lower() != ".csv":
        return None
    fields = _csv_fields(path)
    if "human_rating" in fields and "model_answer" in fields:
        return JudgeCalibrationAdapter(path)
    if "decision" not in fields or "item_id" not in fields:
        return None
    if "review_profile" in fields and _translation_profile(path):
        return KnowledgeCutoffAdapter(path)
    return GoldsetVerifyAdapter(path)


def _open_jsonl_review(path: Path) -> ReviewAdapter | None:
    if path.suffix.lower() != ".jsonl":
        return None
    if _is_conflict_resolution(path):
        return ConflictResolutionAdapter(path)
    if _is_external_rag(path):
        return ExternalRagAdapter(path)
    return None


def _open_review_file(path: Path) -> ReviewAdapter:
    if path.name == "comparison.json":
        return DraftCompareAdapter(path)
    if path.name == "candidates.json" or _is_candidate_json(path):
        return PromptSystemAdapter(path)
    adapter = _open_csv_review(path) or _open_jsonl_review(path)
    if adapter is not None:
        return adapter
    raise ValueError(f"unrecognized review ledger: {path}")


def open_review(path: Path | str) -> ReviewAdapter:
    """Open the one adapter whose existing ledger signature matches ``path``."""
    value = Path(path)
    if value.is_dir():
        return _open_review_directory(value)
    if not value.is_file():
        raise ValueError(f"review path not found: {value}")
    return _open_review_file(value)


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
