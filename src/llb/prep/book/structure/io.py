"""Read and write `metadata/document_structure.json`."""

import json
from pathlib import Path

from pydantic import ValidationError

from llb.core.contracts.data_prep.book_structure import DocumentStructure
from llb.core.fsutil import atomic_write_text
from llb.prep.book.constants import METADATA_DIRNAME
from llb.prep.book.structure.constants import DOCUMENT_STRUCTURE_FILENAME


def structure_path(run_dir: Path) -> Path:
    """`metadata/document_structure.json` inside a book-ocr run."""
    return run_dir / METADATA_DIRNAME / DOCUMENT_STRUCTURE_FILENAME


def write_structure(path: Path, structure: DocumentStructure) -> None:
    """Write the structure atomically. The model is the validation."""
    payload = structure.model_dump(mode="json")
    atomic_write_text(path, json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def read_structure(path: Path) -> DocumentStructure:
    """Read a document structure, refusing a file that is not the current contract."""
    try:
        return DocumentStructure.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise ValueError(f"{path} is not a readable document structure") from exc
