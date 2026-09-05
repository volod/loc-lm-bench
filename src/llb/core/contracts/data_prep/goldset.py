"""Gold-set and chain record contracts.

`goldset.jsonl` and `chains.jsonl` are the rows every later reading is scored against, and they
are the oldest artifacts in the project: rows written before the optional fields existed omit
them, and rows written today state every one. Both forms are registered, so an old bundle reaches
the same canonical item as a current one instead of two readers disagreeing about what a missing
`lang` or `verified` meant.
"""

from typing import Literal

from pydantic import Field, model_validator

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.data_prep.corpus import DataPrepRow

GoldProvenance = Literal[
    "sample-generated",
    "public-reused",
    "human-authored",
    "frontier-drafted",
    "ontology-drafted",
    "human-verified",
]
GoldSplit = Literal["calibration", "tuning", "final"]

DEFAULT_LANG = "uk"
DEFAULT_VERIFIED = False


class SourceSpanRecord(DataPrepRow):
    """A labelled span: char offsets into a source document plus the exact text."""

    doc_id: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _check_offsets(self) -> "SourceSpanRecord":
        if self.char_end <= self.char_start:
            raise ValueError(f"char_end ({self.char_end}) must be > char_start ({self.char_start})")
        if len(self.text) != self.char_end - self.char_start:
            raise ValueError("span text length does not match char offsets")
        return self


class GoldItemRecordV1(ArtifactContract):
    """The pre-contract gold row: `lang` and `verified` may be absent and mean their defaults."""

    schema_id: Literal["llb.gold-item"]
    schema_version: Literal["1.0.0"]
    id: str = Field(min_length=1)
    lang: str | None = None
    question: str = Field(min_length=1)
    reference_answer: str
    source_doc_id: str = Field(min_length=1)
    source_spans: list[SourceSpanRecord] = Field(min_length=1)
    provenance: GoldProvenance
    verified: bool | None = None
    split: GoldSplit


class GoldItemRecord(ArtifactContract):
    """The canonical gold row: every field stated, nothing left to a reader's default."""

    schema_id: Literal["llb.gold-item"]
    schema_version: Literal["2.0.0"]
    id: str = Field(min_length=1)
    lang: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str
    source_doc_id: str = Field(min_length=1)
    source_spans: list[SourceSpanRecord] = Field(min_length=1)
    provenance: GoldProvenance
    verified: bool
    split: GoldSplit


class ChainStepRecord(DataPrepRow):
    """One ordered step of a chain item, with its own question, answer, and spans."""

    order: int = Field(ge=1)
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    source_doc_id: str = Field(min_length=1)
    source_spans: list[SourceSpanRecord] = Field(min_length=1)
    dependency_note: str = ""


class GoldChainRecord(ArtifactContract):
    """`chains.jsonl`: one ordered chain-of-questions item."""

    schema_id: Literal["llb.gold-chain"]
    schema_version: Literal["1.0.0"]
    chain_id: str = Field(min_length=1)
    lang: str = Field(default=DEFAULT_LANG, min_length=1)
    steps: list[ChainStepRecord] = Field(min_length=2, max_length=4)
    provenance: GoldProvenance = "ontology-drafted"
    verified: bool = DEFAULT_VERIFIED
    split: GoldSplit = "final"


class NeedleItemRecord(ArtifactContract):
    """One `needle_items.jsonl` row: a gold row plus the review labels it is filtered on.

    The labels live here rather than on the gold row because they are drafting metadata, not part
    of what an answer is scored against; `retrieval_rank` is present only where the drafting run
    was given an index, and `None` there means "searched and not found", never "not searched".
    """

    schema_id: Literal["llb.needle-item"]
    schema_version: Literal["1.0.0"]
    id: str = Field(min_length=1)
    lang: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str
    source_doc_id: str = Field(min_length=1)
    source_spans: list[SourceSpanRecord] = Field(min_length=1)
    provenance: GoldProvenance
    verified: bool
    split: GoldSplit
    question_type: str | None = None
    difficulty: str | None = None
    retrieval_rank: int | None = None
    retrieval_k: int | None = None
