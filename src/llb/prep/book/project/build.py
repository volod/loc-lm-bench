"""Write a scanned book's ledger as a corpus ``ingest-corpus`` already reads.

One content file becomes one document. The document id follows the text, so a
second projection of the same ledger writes the same bytes. A changed text is
a new document whose ``revision_of`` names the previous file.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from llb.prep.book.project.identity import (
    document_id,
    load_priors,
    logical_key,
    part_indexes,
    revision_of,
    text_sha256,
)
from llb.prep.book.project.references import reference_sidecars
from llb.prep.book.project.text import render_content
from llb.prep.book.project.write import ProjectedDocument, write_document
from llb.prep.book.render.catalog import build_catalog
from llb.prep.book.render.constants import SECTION_CAP_FLOOR, SECTION_CHAR_CAP
from llb.prep.book.render.documents import layout_files
from llb.prep.book.render.grammar import load_grammar
from llb.prep.book.render.load import load_book
from llb.prep.book.render.names import (
    anchor_map,
    book_slug,
    equation_numbers,
    heading_levels,
)
from llb.prep.book.render.segment import prepare, section_spans
from llb.prep.book.render.types import BookLoad, RenderContext, VaultFile
from llb.prep.book.source import refuse_source_overlap
from llb.prep.corpus.governance import detect_language
from llb.prep.corpus.redistribution import REDISTRIBUTION_CLASSES

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """What one projection wrote."""

    out_dir: Path
    documents: tuple[str, ...]
    revisions: int


def project_corpus(
    run_dir: Path,
    out_dir: Path,
    licence: str,
    *,
    language: str | None = None,
    section_cap: int = SECTION_CHAR_CAP,
    slug: str | None = None,
    reference_grammar: Path | None = None,
) -> ProjectionResult:
    """Project ``run_dir`` into ``out_dir``. ``licence`` has no default."""
    chosen = require_licence(licence)
    if section_cap < SECTION_CAP_FLOOR:
        raise ValueError(f"section cap must be at least {SECTION_CAP_FLOOR}")
    loaded = load_book(run_dir)
    _refuse(out_dir, run_dir, loaded.pdf)
    name = _slug(slug, loaded.pdf.name)
    documents, files = _documents(loaded, section_cap, name)
    if not documents:
        raise ValueError(f"{run_dir} has no content to project")
    grammar = load_grammar(reference_grammar)
    catalog = build_catalog(files, loaded.structure, grammar)
    prepared = [(item, document_id(item.key, text_sha256(item.text))) for item in documents]
    sidecars = reference_sidecars(prepared, loaded.structure, catalog, grammar)
    spoken = _language(language, [item.text for item in documents])
    priors = load_priors(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    revisions = 0
    source = Path(loaded.config.source.original_path).name
    digest = loaded.config.source.sha256
    for (item, doc_id), sidecar in zip(prepared, sidecars, strict=True):
        content_sha = text_sha256(item.text)
        link = revision_of(priors, item.key, content_sha)
        if link and all(prior.doc_id != doc_id for prior in priors):
            revisions += 1
        write_document(
            out_dir,
            item,
            doc_id=doc_id,
            revision=link,
            licence=chosen,
            language=spoken,
            source=source,
            digest=digest,
            run_id=loaded.config.run_id,
            captured=loaded.config.created_at,
            references=sidecar,
        )
        written.append(doc_id)
    _LOG.info(
        "[book-ocr] project %s document(s), %s revision(s) -> %s",
        len(written),
        revisions,
        out_dir,
    )
    return ProjectionResult(out_dir, tuple(written), revisions)


def require_licence(licence: str | None) -> str:
    """The operator-declared class. Missing and unknown values refuse."""
    allowed = ", ".join(REDISTRIBUTION_CLASSES)
    if licence is None or not licence.strip():
        raise ValueError(f"redistribution class is required; pass --licence ({allowed})")
    chosen = licence.strip()
    if chosen not in REDISTRIBUTION_CLASSES:
        raise ValueError(f"unknown redistribution class {chosen!r}; choose one of {allowed}")
    return chosen


def _documents(
    loaded: BookLoad, section_cap: int, slug: str
) -> tuple[list[ProjectedDocument], list[VaultFile]]:
    ctx = _context(loaded, slug)
    blocks = prepare(loaded.blocks, loaded.structure)
    files = layout_files(section_spans(blocks, loaded.structure), section_cap, slug, ctx)
    indexes = part_indexes([file.node.node_id for file in files])
    digest = loaded.config.source.sha256
    projected: list[ProjectedDocument] = []
    for file, part in zip(files, indexes, strict=True):
        rendered = render_content(file, ctx)
        if rendered is None:
            continue
        key = logical_key(digest, file.node.node_id, part)
        projected.append(
            ProjectedDocument(key, rendered.text, rendered.pages, file.stem, file.node.node_id)
        )
    return projected, files


def _context(loaded: BookLoad, slug: str) -> RenderContext:
    return RenderContext(
        book=slug,
        fallback=False,
        anchors=anchor_map(loaded.structure),
        equations=equation_numbers(loaded.structure),
        assets={},
        levels=heading_levels(loaded.structure),
        content_href=lambda page: f"#page={page}",
        toc_href=lambda page: f"#page={page}",
    )


def _slug(requested: str | None, pdf_name: str) -> str:
    slug = requested or book_slug(pdf_name)
    if slug != book_slug(slug):
        raise ValueError("slug must be lowercase ASCII words and hyphens")
    return slug


def _language(requested: str | None, texts: list[str]) -> str:
    if requested is not None and requested.strip():
        return requested.strip()
    return detect_language("\n".join(texts))


def _refuse(dest: Path, run_dir: Path, pdf: Path) -> None:
    dest_real = dest.resolve()
    run_real = run_dir.resolve()
    if dest_real == run_real or dest_real in run_real.parents:
        raise ValueError(f"refusing to replace the run directory {run_dir}")
    refuse_source_overlap(pdf, dest)
