"""Read a corpus whose documents carry physical PDF page provenance."""

from dataclasses import dataclass
from pathlib import Path

from llb.core.contracts.data_prep.corpus import PdfCitations, PdfPageCitationRecord
from llb.prep.book.project.constants import PARSER as PROJECTION_PARSER
from llb.rag.chunking.corpus import iter_docs
from llb.rag.page_metadata import citation_sidecar_path


@dataclass(frozen=True)
class CitedDocument:
    doc_id: str
    text: str
    citations: PdfCitations


def load_cited_documents(
    root: Path, *, projected: bool, max_page: int | None = None
) -> dict[str, CitedDocument]:
    """Read canonical document offsets and verify the PDF citation partition."""
    documents = {}
    for doc_id, text in iter_docs(root):
        citations = PdfCitations.model_validate_json(
            citation_sidecar_path(root, doc_id).read_bytes()
        )
        if citations.doc_id != doc_id or (citations.parser == PROJECTION_PARSER) != projected:
            raise ValueError(
                f"{doc_id}: citation document or parser does not match the corpus lane"
            )
        _check_pages(doc_id, len(text), citations.pages, projected, max_page)
        documents[doc_id] = CitedDocument(doc_id, text, citations)
    if not documents:
        raise ValueError(f"{root}: no corpus documents")
    return documents


def _check_pages(
    doc_id: str,
    text_length: int,
    pages: list[PdfPageCitationRecord],
    projected: bool,
    max_page: int | None,
) -> None:
    cursor = 0
    for page in pages:
        if not _valid_page_offsets(page, cursor, text_length, max_page):
            raise ValueError(f"{doc_id}: invalid citation for PDF page {page.page}")
        if projected and (
            page.text_start != cursor or page.text_end - page.text_start != page.n_chars
        ):
            raise ValueError(f"{doc_id}: projected page citations must partition the text")
        cursor = page.text_end
    if not pages or (projected and cursor != text_length):
        raise ValueError(f"{doc_id}: missing PDF page text provenance")


def _valid_page_offsets(
    page: PdfPageCitationRecord, cursor: int, text_length: int, max_page: int | None
) -> bool:
    return (
        page.page >= 1
        and (max_page is None or page.page <= max_page)
        and cursor <= page.text_start <= page.text_end <= text_length
        and page.char_start <= page.text_start <= page.text_end <= page.char_end
    )
