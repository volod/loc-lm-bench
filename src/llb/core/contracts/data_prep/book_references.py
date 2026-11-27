"""Printed references of one projected book document.

A citation names a span in this document and, when the book names one target, the span of
that target's block in the document that holds it. An index line names its term and the PDF
pages its printed pages map to. Nothing here copies the sentence the number was read in.
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.data_prep.corpus import DataPrepRow

REFERENCES_SCHEMA_ID: Literal["llb.book-references"] = "llb.book-references"
REFERENCES_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
REFERENCES_KIND: Literal["book-references"] = "book-references"


class ReferenceTarget(DataPrepRow):
    """The block a resolved citation names, as offsets in that block's document."""

    doc_id: str = Field(min_length=1)
    block_id: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _span_is_forward(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError("reference target span is empty")
        return self


class PrintedCitation(DataPrepRow):
    """One number the grammar found. Unresolved citations have no target."""

    kind: str = Field(min_length=1)
    key: str = Field(min_length=1)
    outcome: Literal["resolved", "unresolved"]
    reason: Literal["", "ambiguous", "gap", "absent"] = ""
    cue_start: int = Field(ge=0)
    cue_end: int = Field(ge=0)
    target: ReferenceTarget | None = None

    @model_validator(mode="after")
    def _target_matches_outcome(self) -> Self:
        if self.cue_end <= self.cue_start:
            raise ValueError("citation cue span is empty")
        if self.outcome == "resolved":
            if self.target is None or self.reason:
                raise ValueError("a resolved citation needs a target and no reason")
        elif self.target is not None or not self.reason:
            raise ValueError("an unresolved citation has a reason and no target")
        return self


class IndexPage(DataPrepRow):
    """One printed page an index line names, and the PDF page it maps to."""

    printed: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)


class IndexEntry(DataPrepRow):
    """One index line. A sub-entry names its parent's term when the indent shows one."""

    term_start: int = Field(ge=0)
    term_end: int = Field(ge=0)
    parent_start: int | None = None
    parent_end: int | None = None
    pages: list[IndexPage] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _spans_are_forward(self) -> Self:
        if self.term_end <= self.term_start:
            raise ValueError("index term span is empty")
        if (self.parent_start is None) != (self.parent_end is None):
            raise ValueError("index parent span needs both ends")
        if self.parent_start is not None and self.parent_end is not None:
            if self.parent_end <= self.parent_start:
                raise ValueError("index parent span is empty")
        return self


class HierarchyPlace(DataPrepRow):
    """One node on the path from the book root to this document."""

    node_id: str = Field(min_length=1)
    level: str = Field(min_length=1)
    number: str = ""
    title: str = ""


class BookReferences(ArtifactContract):
    """`<document>.md.references.json`: the printed references of one projected document."""

    schema_id: Literal["llb.book-references"] = REFERENCES_SCHEMA_ID
    schema_version: Literal["1.0.0"] = REFERENCES_SCHEMA_VERSION
    kind: Literal["book-references"] = REFERENCES_KIND
    doc_id: str = Field(min_length=1)
    grammar: str = Field(min_length=1)
    hierarchy: list[HierarchyPlace] = Field(default_factory=list)
    citations: list[PrintedCitation] = Field(default_factory=list)
    index: list[IndexEntry] = Field(default_factory=list)
