"""Shapes shared by more than one retrieval or graph contract."""

from pydantic import BaseModel, ConfigDict, Field

from llb.core.contracts.artifacts import DIGEST_PATTERN


class RetrievalRow(BaseModel):
    """Strict nested row shared by the retrieval, graph, and prompt-system contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class OpaqueIndexMember(RetrievalRow):
    """One file a store cannot be queried without whose bytes belong to another library.

    A FAISS index, a per-backend vector directory, a BM25 posting list, and a DuckDB database are
    all in this class. The store owns WHICH members exist and what they weigh; the format itself
    stays the owner's, so the record names the owner and its format version instead of describing
    a layout this project does not control.
    """

    member_id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    path: str = Field(min_length=1)  # relative to the store directory
    owner: str = Field(min_length=1)
    format: str = Field(min_length=1)
    format_version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    digest: str = Field(pattern=DIGEST_PATTERN)
    n_bytes: int = Field(ge=0)


class SourceSpan(RetrievalRow):
    """An offset-bearing span: the coordinates every retrieval metric is scored on."""

    doc_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str
