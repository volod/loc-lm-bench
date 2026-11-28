"""Validate paired corpora and translate accepted evidence by physical PDF page."""

import re
from dataclasses import dataclass
from pathlib import Path

from llb.core.contracts.data_prep.corpus import (
    PdfCitations,
    PdfCorpusManifest,
    PdfPageCitationRecord,
)
from llb.core.contracts.rag import SourceSpanRecord
from llb.executor.cases import spans_as_dicts
from llb.goldset.schema import GoldItem, load_goldset
from llb.goldset.validate import validate_items
from llb.prep.book.project.constants import PARSER as PROJECTION_PARSER
from llb.prep.pdf.model import PDF_CORPUS_MANIFEST
from llb.rag.chunking.corpus import iter_docs
from llb.rag.page_metadata import citation_sidecar_path
from llb.rag.retrieval import spans_overlap

SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CitedDocument:
    doc_id: str
    text: str
    citations: PdfCitations


@dataclass(frozen=True)
class BookPair:
    projection: dict[str, CitedDocument]
    baseline: CitedDocument
    source_sha256: str


@dataclass(frozen=True)
class BookQuestion:
    item_id: str
    question: str
    pages: frozenset[int]
    projection_spans: list[SourceSpanRecord]
    baseline_spans: list[SourceSpanRecord]
    present_parts: int
    evidence_parts: int

    @property
    def baseline_text_present(self) -> bool:
        return self.present_parts == self.evidence_parts


def _load_documents(
    root: Path, *, projected: bool, max_page: int | None
) -> dict[str, CitedDocument]:
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


def load_book_pair(projection_root: Path, baseline_root: Path) -> BookPair:
    """Accept a projection and a one-PDF conversion whose scan digests agree."""
    manifest = PdfCorpusManifest.model_validate_json(
        (baseline_root / PDF_CORPUS_MANIFEST).read_bytes()
    )
    if manifest.kind != "pdf-corpus" or len(manifest.items) != 1:
        raise ValueError("baseline must be the pdf-to-markdown conversion of one PDF")
    record = manifest.items[0]
    digest = record.source_sha256
    if record.status != "ok" or digest is None or not SHA256.fullmatch(digest):
        raise ValueError(
            "baseline conversion needs a successful document and a source PDF checksum"
        )
    baseline_docs = _load_documents(baseline_root, projected=False, max_page=record.page_count)
    if record.doc_id is None or set(baseline_docs) != {record.doc_id}:
        raise ValueError("baseline document set differs from the PDF conversion manifest")
    baseline = baseline_docs[record.doc_id]
    if (baseline.citations.source, baseline.citations.parser) != (record.source, record.parser):
        raise ValueError("baseline citations differ from the PDF conversion manifest")
    if len({page.page for page in baseline.citations.pages}) != len(baseline.citations.pages):
        raise ValueError("baseline cites a physical PDF page more than once")
    projection = _load_documents(projection_root, projected=True, max_page=record.page_count)
    for doc in projection.values():
        if doc.citations.diagnostics.get("source_sha256") != digest:
            raise ValueError(f"{doc.doc_id}: projection and baseline cite different source PDFs")
    return BookPair(projection, baseline, digest)


def load_book_questions(goldset: Path, projection_root: Path, pair: BookPair) -> list[BookQuestion]:
    """Refuse any stale or unverified row before mapping its spans to the baseline."""
    items = load_goldset(goldset)
    if not items:
        raise ValueError("gold set has no items")
    for item in items:
        if not item.verified:
            raise ValueError(f"{item.id}: gold item is unverified")
        if any(span.doc_id not in pair.projection for span in item.source_spans):
            raise ValueError(f"{item.id}: gold item refers to a missing projected document")
    errors = validate_items(items, projection_root)["errors"]
    if errors:
        raise ValueError(f"gold validation failed: {errors[0]}")
    baseline_pages = {page.page: page for page in pair.baseline.citations.pages}
    return [_map_question(item, pair, baseline_pages) for item in items]


def _map_question(
    item: GoldItem, pair: BookPair, baseline_pages: dict[int, PdfPageCitationRecord]
) -> BookQuestion:
    spans = spans_as_dicts(item)
    pages: set[int] = set()
    mapped: list[SourceSpanRecord] = []
    present_parts = evidence_parts = 0
    for span in spans:
        doc = pair.projection[span["doc_id"]]
        parts = _span_parts(doc, span)
        if not parts:
            raise ValueError(f"{item.id}: gold span contains no cited evidence text")
        for page, text in parts:
            pages.add(page)
            matches = _baseline_matches(pair.baseline, baseline_pages.get(page), text)
            evidence_parts += 1
            present_parts += bool(matches)
            mapped.extend(matches)
    return BookQuestion(
        item.id, item.question, frozenset(pages), spans, mapped, present_parts, evidence_parts
    )


def _span_parts(doc: CitedDocument, span: SourceSpanRecord) -> list[tuple[int, str]]:
    parts = []
    for page in doc.citations.pages:
        start = max(span["char_start"], page.text_start)
        end = min(span["char_end"], page.text_end)
        if start < end and (text := doc.text[start:end].strip()):
            parts.append((page.page, text))
    return parts


def _baseline_matches(
    doc: CitedDocument, page: PdfPageCitationRecord | None, text: str
) -> list[SourceSpanRecord]:
    if page is None:
        return []
    pattern = r"\s+".join(re.escape(token) for token in text.split())
    # A number or word embedded in a longer token is a different reading.
    left = r"(?<!\w)" if re.match(r"\w", text[0]) else ""
    right = r"(?!\w)" if re.match(r"\w", text[-1]) else ""
    if text[0].isdecimal():
        left += r"(?<![+\-\u2212])(?<!\d[.,])"
    if text[-1].isdecimal():
        right += r"(?![.,]\d)"
    body = doc.text[page.text_start : page.text_end]
    return [
        SourceSpanRecord(
            doc_id=doc.doc_id,
            char_start=page.text_start + match.start(1),
            char_end=page.text_start + match.end(1),
            text=match[1],
        )
        for match in re.finditer(f"(?={left}({pattern}){right})", body)
    ]


def pages_at_span(doc: CitedDocument, start: int, end: int) -> set[int]:
    """Physical pages whose body intersects the retrieved occurrence."""
    return {
        page.page
        for page in doc.citations.pages
        if page.text_start < page.text_end
        and spans_overlap(start, end, page.text_start, page.text_end)
    }
