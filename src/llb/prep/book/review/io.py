"""Read and write the review queue and the page quality manifest."""

import json
from pathlib import Path

from pydantic import ValidationError

from llb.core.contracts.data_prep.book_review import QualityManifest, ReviewQueue
from llb.core.fsutil import atomic_write_text
from llb.prep.book.constants import METADATA_DIRNAME
from llb.prep.book.resolve.constants import REVIEW_DIRNAME
from llb.prep.book.review.constants import QUALITY_MANIFEST_FILENAME, QUEUE_FILENAME


def queue_path(run_dir: Path) -> Path:
    """`review/queue.json` inside a book-ocr run."""
    return run_dir / REVIEW_DIRNAME / QUEUE_FILENAME


def manifest_path(run_dir: Path) -> Path:
    """`metadata/quality_manifest.json` inside a book-ocr run."""
    return run_dir / METADATA_DIRNAME / QUALITY_MANIFEST_FILENAME


def run_dir_of(queue_path: Path) -> Path:
    """The run that holds this queue. The queue file lives in its `review` directory."""
    if queue_path.parent.name != REVIEW_DIRNAME:
        raise ValueError(f"{queue_path} is not inside a {REVIEW_DIRNAME} directory")
    return queue_path.parent.parent


def write_queue(path: Path, queue: ReviewQueue) -> None:
    _write(path, queue)


def read_queue(path: Path) -> ReviewQueue:
    return _read(path, ReviewQueue, "review queue")


def write_manifest(path: Path, manifest: QualityManifest) -> None:
    _write(path, manifest)


def read_manifest(path: Path) -> QualityManifest:
    return _read(path, QualityManifest, "quality manifest")


def _write(path: Path, model: ReviewQueue | QualityManifest) -> None:
    payload = model.model_dump(mode="json")
    atomic_write_text(path, json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def _read[T: (ReviewQueue, QualityManifest)](path: Path, model: type[T], label: str) -> T:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise ValueError(f"{path} is not a readable {label}") from exc
