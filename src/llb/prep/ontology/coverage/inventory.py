"""Stage 1 -- inventory + normalize supported documents, preserving exact offsets.

The on-disk text is treated as canonical and is NOT mutated: every later span indexes the same
bytes the validator reads back, so spans stay exact. "Normalize" here means deriving structure
(a content hash + section segmentation) over the untouched text, never rewriting it.

Sections are markdown headings (`# ...`) when present, else paragraph blocks split on blank
lines. They give stage 4 a coverage axis and stage 5 a bounded context window.
"""

import hashlib
import logging
import re
from pathlib import Path

from llb.prep.ontology.constants import SUPPORTED_SUFFIXES
from llb.prep.ontology.models import DocRecord, Section
from llb.rag.chunking.corpus import iter_docs

_LOG = logging.getLogger(__name__)

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_BLANK_LINE = re.compile(r"\n[ \t]*\n")


def sha256_text(text: str) -> str:
    """Stable content hash recorded in provenance for document-version traceability."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def segment_sections(text: str) -> list[Section]:
    """Split `text` into titled sections by markdown heading, else by paragraph block.

    Offsets are exact into `text`; the whole document is covered with no gaps so every
    extracted span lands in exactly one section.
    """
    if not text:
        return []
    headings = list(_HEADING.finditer(text))
    if headings:
        sections: list[Section] = []
        # preamble before the first heading, if any non-whitespace content
        if headings[0].start() > 0 and text[: headings[0].start()].strip():
            sections.append(Section(title="(preamble)", char_start=0, char_end=headings[0].start()))
        for i, match in enumerate(headings):
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            sections.append(
                Section(title=match.group(2).strip(), char_start=match.start(), char_end=end)
            )
        return sections
    # no headings: paragraph blocks
    sections = []
    start = 0
    for gap in _BLANK_LINE.finditer(text):
        if text[start : gap.start()].strip():
            sections.append(_paragraph(text, start, gap.start()))
        start = gap.end()
    if text[start:].strip():
        sections.append(_paragraph(text, start, len(text)))
    return sections or [Section(title="(document)", char_start=0, char_end=len(text))]


def _paragraph(text: str, start: int, end: int) -> Section:
    title = " ".join(text[start:end].split())[:60]
    return Section(title=title or "(paragraph)", char_start=start, char_end=end)


def section_at(sections: list[Section], char_start: int) -> str:
    """Title of the section containing `char_start` (or "(document)" if none matches)."""
    for sec in sections:
        if sec.char_start <= char_start < sec.char_end:
            return sec.title
    return "(document)"


def inventory_corpus(corpus_root: Path | str) -> list[DocRecord]:
    """Inventory every supported doc under `corpus_root` (recursive), offsets preserved.

    Reuses the canonical `iter_docs` walk (corpus-relative ids, the same `.txt`/`.md` set the
    chunker/frontier use) and enriches each doc with the hash + section metadata the later
    stages need.
    """
    root = Path(corpus_root)
    if not root.exists():
        raise ValueError(f"corpus root does not exist: {root}")
    docs = [
        DocRecord(
            doc_id=doc_id,
            text=text,
            sha256=sha256_text(text),
            n_chars=len(text),
            sections=segment_sections(text),
        )
        for doc_id, text in iter_docs(root)
    ]
    if not docs:
        raise ValueError(f"no {'/'.join(SUPPORTED_SUFFIXES)} documents under {root}")
    _LOG.info("[ontology] stage 1: inventoried %d documents under %s", len(docs), root)
    return docs
