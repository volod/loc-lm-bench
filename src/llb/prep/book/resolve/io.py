"""Read and publish the overlay, the resolved page, and the two edit inputs.

Model edits live in the `correct` stage directory, so their digest is that stage's output digest.
Reviewer decisions live at `review/decisions.json`, outside every stage directory. A page's cache
key digests only the decisions that name that page, so a new decision re-runs that page alone.
"""

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from llb.core.contracts.data_prep.book_resolve import (
    CorrectionEdit,
    ModelCorrections,
    PageCorrectionOverlay,
    PageResolvedText,
    ReviewerDecisions,
)
from llb.core.fsutil import atomic_write_text
from llb.prep.book.cache import digest_json
from llb.prep.book.resolve.constants import (
    CORRECTIONS_FILENAME,
    DECISIONS_FILENAME,
    OVERLAY_FILENAME,
    RESOLVED_FILENAME,
    REVIEW_DIRNAME,
)


def decisions_path(run_dir: Path) -> Path:
    """The run-level reviewer file. Absent means the reviewer has decided nothing."""
    return run_dir / REVIEW_DIRNAME / DECISIONS_FILENAME


def reviewer_digest(run_dir: Path, page_id: str) -> str:
    """Digest of this page's reviewer decisions. Other pages' decisions do not enter it."""
    path = decisions_path(run_dir)
    if not path.is_file():
        return digest_json([])
    try:
        document = read_decisions(path)
    except (OSError, ValidationError, ValueError):
        raw = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest_json({"invalid": raw})
    page = [edit.model_dump(mode="json") for edit in document.edits if edit.page_id == page_id]
    return digest_json(page)


def load_reviewer_edits(run_dir: Path, page_id: str) -> list[CorrectionEdit]:
    """This page's reviewer edits. A missing file is an empty list; a broken file is an error."""
    path = decisions_path(run_dir)
    if not path.is_file():
        return []
    document = read_decisions(path)
    return [edit for edit in document.edits if edit.page_id == page_id]


def load_model_edits(directory: Path | None, page_id: str) -> list[CorrectionEdit]:
    """Edits from the `correct` stage, when the run configured one."""
    document = load_model_corrections(directory, page_id)
    if document is None:
        return []
    return list(document.edits)


def load_model_corrections(directory: Path | None, page_id: str) -> ModelCorrections | None:
    """The corrections file, including refusals. None when the stage did not run."""
    if directory is None:
        return None
    path = directory / CORRECTIONS_FILENAME
    if not path.is_file():
        raise ValueError(f"correct stage has no {CORRECTIONS_FILENAME}")
    document = read_corrections(path)
    if document.page_id != page_id:
        raise ValueError(f"{path.name} is for page {document.page_id}, not {page_id}")
    return document


def write_overlay(path: Path, overlay: PageCorrectionOverlay) -> None:
    _write(path, overlay)


def read_overlay(path: Path) -> PageCorrectionOverlay:
    return _read(path, PageCorrectionOverlay, "correction overlay")


def write_resolved(path: Path, page: PageResolvedText) -> None:
    _write(path, page)


def read_resolved(path: Path) -> PageResolvedText:
    return _read(path, PageResolvedText, "resolved page")


def write_decisions(path: Path, decisions: ReviewerDecisions) -> None:
    _write(path, decisions)


def read_decisions(path: Path) -> ReviewerDecisions:
    return _read(path, ReviewerDecisions, "reviewer decisions")


def write_corrections(path: Path, corrections: ModelCorrections) -> None:
    _write(path, corrections)


def read_corrections(path: Path) -> ModelCorrections:
    return _read(path, ModelCorrections, "model corrections")


def publish_page_resolution(stage_directory: Path) -> None:
    """Copy a committed resolve stage up to `overlay.json` and `resolved.json`."""
    _publish(stage_directory, OVERLAY_FILENAME, PageCorrectionOverlay)
    _publish(stage_directory, RESOLVED_FILENAME, PageResolvedText)


def publish_page_corrections(stage_directory: Path) -> None:
    """Copy a committed correct stage up to `pages/<pNNNN>/corrections.json`."""
    _publish(stage_directory, CORRECTIONS_FILENAME, ModelCorrections)


def _publish(
    stage_directory: Path,
    name: str,
    model: type[PageCorrectionOverlay] | type[PageResolvedText] | type[ModelCorrections],
) -> None:
    source = stage_directory / name
    if not source.is_file():
        return
    text = source.read_text(encoding="utf-8")
    target = stage_directory.parent / name
    if target.is_file() and target.read_text(encoding="utf-8") == text:
        return
    model.model_validate_json(text)
    atomic_write_text(target, text)


def _write(
    path: Path,
    model: PageCorrectionOverlay | PageResolvedText | ReviewerDecisions | ModelCorrections,
) -> None:
    payload = model.model_dump(mode="json")
    atomic_write_text(path, json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def _read[T: (PageCorrectionOverlay, PageResolvedText, ReviewerDecisions, ModelCorrections)](
    path: Path, model: type[T], label: str
) -> T:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise ValueError(f"{path} is not a readable {label}") from exc
