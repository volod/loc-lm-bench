"""Serialize one projected document and its citation and metadata sidecars."""

import json
from dataclasses import dataclass
from pathlib import Path

from llb.core.contracts.data_prep.corpus import PdfCitations, PdfPageCitationRecord
from llb.core.fsutil import atomic_write_text
from llb.prep.book.project.constants import LOGICAL_KEY_FIELD, PARSER, SOURCE_SYSTEM
from llb.prep.corpus.sidecars import REFERENCE_SIDECAR_SUFFIX
from llb.prep.book.project.spans import PageSpan
from llb.prep.corpus.governance import SOURCE_METADATA_SUFFIX


@dataclass(frozen=True, slots=True)
class ProjectedDocument:
    """One content file's text, the pages it cites, and its logical key."""

    key: str
    text: str
    pages: tuple[PageSpan, ...]
    stem: str
    node_id: str


def write_document(
    out_dir: Path,
    item: ProjectedDocument,
    *,
    doc_id: str,
    revision: str | None,
    licence: str,
    language: str,
    source: str,
    digest: str,
    run_id: str,
    captured: str,
    references: str,
) -> None:
    """Write the document, its citations, its references, and its metadata."""
    _write(out_dir / doc_id, item.text)
    citation = out_dir / Path(doc_id).with_suffix(".citations.json")
    _write(citation, _citations(source, doc_id, digest, item))
    _write(out_dir / f"{doc_id}{REFERENCE_SIDECAR_SUFFIX}", references)
    sidecar = out_dir / f"{doc_id}{SOURCE_METADATA_SUFFIX}"
    _write(sidecar, _metadata(item.key, revision, licence, language, digest, run_id, captured))


def _citations(source: str, doc_id: str, digest: str, item: ProjectedDocument) -> str:
    record = PdfCitations(
        schema_id="llb.pdf-citations",
        schema_version="1.0.0",
        kind="pdf-citations",
        source=source,
        doc_id=doc_id,
        parser=PARSER,
        diagnostics={"source_sha256": digest},
        pages=[_page(page) for page in item.pages],
    )
    payload = record.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _page(page: PageSpan) -> PdfPageCitationRecord:
    return PdfPageCitationRecord(
        page=page.page,
        char_start=page.char_start,
        char_end=page.char_end,
        text_start=page.char_start,
        text_end=page.char_end,
        n_chars=page.char_end - page.char_start,
        parser=PARSER,
        blocks=list(page.blocks),
    )


def _metadata(
    key: str,
    revision: str | None,
    licence: str,
    language: str,
    digest: str,
    run_id: str,
    captured: str,
) -> str:
    row: dict[str, str] = {
        "acquisition_run_id": run_id,
        "capture_id": digest,
        "capture_time": captured,
        LOGICAL_KEY_FIELD: key,
        "language": language,
        "licence": licence,
        "payload_digest": digest,
        "source_system": SOURCE_SYSTEM,
        "source_uri": f"urn:sha256:{digest}",
    }
    if revision:
        row["revision_of"] = revision
    return json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    atomic_write_text(path, content)
