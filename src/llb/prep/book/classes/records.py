"""Read and write the class record and the pilot selection."""

import json
from pathlib import Path

from llb.core.contracts.data_prep.book_classes import PageClassRecord, PilotSelection
from llb.core.fsutil import atomic_write_text
from llb.prep.book.classes.constants import CLASSES_FILENAME, CLASSES_STAGE
from llb.prep.book.constants import PAGES_DIRNAME


def classes_path(directory: Path) -> Path:
    return directory / CLASSES_FILENAME


def write_classes(path: Path, record: PageClassRecord) -> None:
    """Write one page's class record."""
    _write(path, record.model_dump(mode="json"))


def read_classes(path: Path) -> PageClassRecord:
    """The class record, refusing a file that is not the current contract."""
    return PageClassRecord.model_validate_json(path.read_text(encoding="utf-8"))


def publish_page_classes(stage_directory: Path) -> None:
    """Copy a committed `classes` stage's file up to `pages/<pNNNN>/classes.json`."""
    source = stage_directory / CLASSES_FILENAME
    if not source.is_file():
        return
    target = stage_directory.parent / CLASSES_FILENAME
    text = source.read_text(encoding="utf-8")
    if target.is_file() and target.read_text(encoding="utf-8") == text:
        return
    PageClassRecord.model_validate_json(text)
    atomic_write_text(target, text)


def write_selection(path: Path, selection: PilotSelection) -> None:
    """Write the pilot selection."""
    _write(path, selection.model_dump(mode="json"))


def read_selection(path: Path) -> PilotSelection:
    """The pilot selection, refusing a file that is not the current contract."""
    return PilotSelection.model_validate_json(path.read_text(encoding="utf-8"))


def load_page_classes(run_dir: Path) -> list[PageClassRecord]:
    """Published class records, or the stage copies when nothing has been published yet."""
    pages = run_dir / PAGES_DIRNAME
    published = sorted(pages.glob(f"p*/{CLASSES_FILENAME}"))
    paths = published or sorted(pages.glob(f"p*/{CLASSES_STAGE}/{CLASSES_FILENAME}"))
    return [read_classes(path) for path in paths]


def _write(path: Path, payload: object) -> None:
    text = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, text)
