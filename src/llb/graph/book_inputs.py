"""Validate the exact source positions a book-reference graph may use."""

import re
from dataclasses import dataclass
from pathlib import Path

from llb.core.contracts.data_prep.book_references import BookReferences, ReferenceTarget
from llb.goldset.markers import MARKER
from llb.graph.model import GraphMention
from llb.prep.corpus.sidecars import REFERENCE_SIDECAR_SUFFIX
from llb.rag.cited_corpus import CitedDocument, load_cited_documents

type BlockKey = tuple[str, str, int, int]

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PrintedBlock:
    key: BlockKey
    page: int
    mention: GraphMention


@dataclass(frozen=True)
class BookDocument:
    source: CitedDocument
    references: BookReferences

    def mention(self, start: int, end: int) -> GraphMention:
        body = self.source.text
        if not (0 <= start < end <= len(body)) or not MARKER.sub("", body[start:end]).strip():
            raise ValueError(f"{self.source.doc_id}: no printed text at offsets {start}:{end}")
        path = self.references.hierarchy
        return GraphMention(
            doc_id=self.source.doc_id,
            char_start=start,
            char_end=end,
            text=body[start:end],
            section_title=path[-1].title if path else "",
        )

    def printed_blocks(self) -> list[PrintedBlock]:
        blocks = []
        for page in self.source.citations.pages:
            for raw in page.blocks:
                identifier, start, end = (
                    raw.get("block_id"),
                    raw.get("char_start"),
                    raw.get("char_end"),
                )
                generated = raw.get("generated")
                if (
                    not isinstance(identifier, str)
                    or type(start) is not int
                    or type(end) is not int
                    or not isinstance(generated, bool)
                    or not page.text_start <= start <= end <= page.text_end
                ):
                    raise ValueError(
                        f"{self.source.doc_id}: invalid projected block on PDF page {page.page}"
                    )
                if generated or not MARKER.sub("", self.source.text[start:end]).strip():
                    continue
                if not identifier:
                    raise ValueError(f"{self.source.doc_id}: printed block has no id")
                blocks.append(
                    PrintedBlock(
                        (self.source.doc_id, identifier, start, end),
                        page.page,
                        self.mention(start, end),
                    )
                )
        return blocks


@dataclass(frozen=True)
class BookSources:
    documents: dict[str, BookDocument]
    blocks: dict[BlockKey, PrintedBlock]
    source_sha256: str

    def target(self, target: ReferenceTarget) -> PrintedBlock:
        key = target.doc_id, target.block_id, target.char_start, target.char_end
        try:
            return self.blocks[key]
        except KeyError:
            raise ValueError(
                f"{target.doc_id}: reference target is not a printed block: {key[1:]}"
            ) from None


def load_book_sources(root: Path) -> BookSources:
    """Read a single projected scan with its citation and reference sidecars intact."""
    documents = {}
    blocks = {}
    digests = set()
    for doc_id, source in load_cited_documents(root, projected=True).items():
        digest = source.citations.diagnostics.get("source_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"{doc_id}: missing source PDF checksum")
        digests.add(digest)
        references = BookReferences.model_validate_json(
            (root / f"{doc_id}{REFERENCE_SIDECAR_SUFFIX}").read_bytes()
        )
        if references.doc_id != doc_id:
            raise ValueError(f"{doc_id}: reference sidecar names another document")
        document = BookDocument(source, references)
        documents[doc_id] = document
        for block in document.printed_blocks():
            if block.key in blocks:
                raise ValueError(f"{doc_id}: duplicate printed block range {block.key[1:]}")
            blocks[block.key] = block
    if len(digests) != 1:
        raise ValueError("book reference graph requires a projection of one source PDF")
    return BookSources(documents, blocks, digests.pop())
