"""Reference accepts, the digest a commit can hold, and the comparison manifest.

Recovered page text stays out of git. The comparison refuses to choose a pipeline
unless every reference page carries an accept.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from llb.core.contracts.data_prep.corpus import DataPrepRow
from llb.prep.book.metrics.reference import RefPage


class PageAccept(DataPrepRow):
    """One reviewer accept. A false accept does not validate."""

    pdf_page: int = Field(ge=1)
    reviewer: str = Field(min_length=1)
    accepted: Literal[True]


class AcceptRecord(DataPrepRow):
    """The accepts file beside a reference transcription."""

    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pages: list[PageAccept] = Field(min_length=1)


def reference_digest(pages: Sequence[RefPage]) -> str:
    """SHA-256 of the pages in PDF order. The same pages digest the same way."""
    ordered = sorted(pages, key=lambda page: page.pdf_page)
    payload = json.dumps(
        [page.model_dump(mode="json") for page in ordered],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def load_accepts(path: Path) -> AcceptRecord:
    """Read ``accepts.json``. A file that is not the record is refused."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return AcceptRecord.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(
            f"the comparison cannot start: {path} is not an accepts record: {exc}"
        ) from exc


def require_accepts(
    pages: list[RefPage], record: AcceptRecord, source_sha256: str | None = None
) -> None:
    """Every reference page is accepted, and the checksum matches when one is given."""
    if source_sha256 is not None and record.source_sha256 != source_sha256:
        raise ValueError("accepts source checksum does not match the scan")
    accepted = {row.pdf_page for row in record.pages}
    missing = [page.pdf_page for page in pages if page.pdf_page not in accepted]
    if missing:
        listed = ", ".join(str(page) for page in missing)
        raise ValueError(f"the comparison cannot start: unaccepted pilot pages: {listed}")
