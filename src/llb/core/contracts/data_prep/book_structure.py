"""The inferred structure of one scanned book.

The block ledger is the source. This record is what a later rendering reads when it needs a
printed page, a part, or a figure number. A node exists only where a page supplied the evidence.
`fallback` marks a page or a node the evidence did not settle.
"""

from typing import Final, Literal

from pydantic import Field, model_validator

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.data_prep.corpus import DataPrepRow

PageStatus = Literal["observed", "gap", "unnumbered", "front-matter"]
PAGE_STATUSES: tuple[PageStatus, ...] = ("observed", "gap", "unnumbered", "front-matter")
NumberScheme = Literal["arabic", "roman", "none"]
NUMBER_SCHEMES: tuple[NumberScheme, ...] = ("arabic", "roman", "none")
NodeLevel = Literal["book", "part", "chapter", "section", "subsection", "page-range"]
NODE_LEVELS: tuple[NodeLevel, ...] = (
    "book",
    "part",
    "chapter",
    "section",
    "subsection",
    "page-range",
)
NodeEvidence = Literal[
    "root",
    "numbered-heading",
    "typography",
    "contents",
    "running-header",
    "fallback",
]
NODE_EVIDENCE: tuple[NodeEvidence, ...] = (
    "root",
    "numbered-heading",
    "typography",
    "contents",
    "running-header",
    "fallback",
)
ConflictKind = Literal[
    "contents-without-heading",
    "heading-without-contents",
    "title-mismatch",
    "page-mismatch",
    "header-mismatch",
]
CONFLICT_KINDS: tuple[ConflictKind, ...] = (
    "contents-without-heading",
    "heading-without-contents",
    "title-mismatch",
    "page-mismatch",
    "header-mismatch",
)
ItemKind = Literal["figure", "table", "equation"]
ITEM_KINDS: tuple[ItemKind, ...] = ("figure", "table", "equation")
ItemEvidence = Literal["caption", "page-footer", "display-number"]
ITEM_EVIDENCE: tuple[ItemEvidence, ...] = ("caption", "page-footer", "display-number")

STRUCTURE_SCHEMA_ID: Final = "llb.book-ocr-document-structure"
STRUCTURE_KIND: Final = "book-ocr-document-structure"
STRUCTURE_SCHEMA_VERSION: Final = "1.0.0"
_SHA = r"^[0-9a-f]{64}$"


class PrintedPage(DataPrepRow):
    """One PDF page's printed number, and whether that number was read or fitted."""

    pdf_page: int = Field(ge=1)
    page_id: str = Field(min_length=1)
    printed: str | None = None
    scheme: NumberScheme
    status: PageStatus
    block_id: str | None = None
    fallback: bool


class RunningHeader(DataPrepRow):
    """One running header text across a contiguous run of PDF pages."""

    text: str = Field(min_length=1)
    pdf_start: int = Field(ge=1)
    pdf_end: int = Field(ge=1)
    block_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _range_is_forward(self) -> "RunningHeader":
        if self.pdf_end < self.pdf_start:
            raise ValueError("running header range is reversed")
        return self


class NumberedItem(DataPrepRow):
    """One figure, table, or equation number read on one page."""

    kind: ItemKind
    number: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)
    printed: str | None = None
    block_id: str = Field(min_length=1)
    evidence: ItemEvidence


class StructureConflict(DataPrepRow):
    """A contents entry, a heading, or a running header that did not agree."""

    kind: ConflictKind
    number: str = ""
    contents_title: str = ""
    heading_title: str = ""
    printed: str = ""
    pdf_page: int | None = None
    detail: str = Field(min_length=1)


class StructureNode(DataPrepRow):
    """One node of the hierarchy. Children are the next level down, in reading order."""

    node_id: str = Field(min_length=1)
    level: NodeLevel
    number: str = ""
    title: str = ""
    pdf_start: int = Field(ge=1)
    pdf_end: int = Field(ge=1)
    printed_start: str | None = None
    printed_end: str | None = None
    block_id: str | None = None
    evidence: list[NodeEvidence] = Field(min_length=1)
    fallback: bool
    children: list["StructureNode"] = Field(default_factory=list)

    @model_validator(mode="after")
    def _range_is_forward(self) -> "StructureNode":
        if self.pdf_end < self.pdf_start:
            raise ValueError("structure node page range is reversed")
        return self


class DocumentStructure(ArtifactContract):
    """`metadata/document_structure.json`: printed pages, hierarchy, and numbering."""

    schema_id: Literal["llb.book-ocr-document-structure"]
    schema_version: Literal["1.0.0"]
    kind: Literal["book-ocr-document-structure"]
    source_sha256: str = Field(pattern=_SHA)
    pages: list[PrintedPage] = Field(min_length=1)
    headers: list[RunningHeader] = Field(default_factory=list)
    hierarchy: StructureNode
    inventory: list[NumberedItem] = Field(default_factory=list)
    conflicts: list[StructureConflict] = Field(default_factory=list)
    contents_pages: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _pages_are_unique_and_ordered(self) -> "DocumentStructure":
        numbers = [page.pdf_page for page in self.pages]
        if numbers != sorted(set(numbers)):
            raise ValueError("printed-page mapping must list each PDF page once, in order")
        return self


StructureNode.model_rebuild()
