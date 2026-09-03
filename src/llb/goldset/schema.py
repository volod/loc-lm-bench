"""Canonical gold-item schema for loc-lm-bench RAG evaluation.

A gold item is a question with a reference answer and SOURCE-SPAN labels (doc id +
character offsets into the source text). Spans are anchored to character offsets, not
chunk ids, so they survive chunk_size tuning. Only `verified=True` items score models.
Verification can be a local review decision or acceptance of a pinned, upstream post-edited
development fixture; `provenance` and the fixture metadata preserve that distinction.

Pydantic enforces the schema (types + allowed provenance/split values) at construction.
"""

import json
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, Field, model_validator

GOLD_ITEM_SCHEMA_ID: Final[Literal["llb.gold-item"]] = "llb.gold-item"
GOLD_ITEM_SCHEMA_VERSION: Final[Literal["2.0.0"]] = "2.0.0"

Provenance = Literal[
    "sample-generated",
    "public-reused",
    "human-authored",
    "frontier-drafted",
    "ontology-drafted",
    "human-verified",
]
Split = Literal["calibration", "tuning", "final"]


class SourceSpan(BaseModel):
    """A labeled span: char offsets into a source doc, plus the exact text."""

    doc_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _check_offsets(self) -> "SourceSpan":
        if self.char_end <= self.char_start:
            raise ValueError(f"char_end ({self.char_end}) must be > char_start ({self.char_start})")
        if len(self.text) != self.char_end - self.char_start:
            raise ValueError("span text length does not match char offsets")
        return self


class GoldItem(BaseModel):
    """One RAG gold item with source-span labels.

    The row carries its own contract identity so a gold set read outside this build says what it
    is. A row written before the registry existed carries neither field and loads through the
    registered migration, which states the two values it left to a reader's defaults.
    """

    schema_id: Literal["llb.gold-item"] = GOLD_ITEM_SCHEMA_ID
    schema_version: Literal["2.0.0"] = GOLD_ITEM_SCHEMA_VERSION
    id: str
    lang: str = "uk"
    question: str
    reference_answer: str
    source_doc_id: str
    source_spans: list[SourceSpan]
    provenance: Provenance
    verified: bool = False
    split: Split

    @model_validator(mode="after")
    def _require_spans(self) -> "GoldItem":
        if not self.source_spans:
            raise ValueError(f"{self.id}: at least one source span is required")
        return self


def _gold_item_at_current(record: dict[str, object], source: str) -> GoldItem:
    """One row at the current contract, migrating a pre-contract row on the way in."""
    from llb.artifacts.default_registry import DEFAULT_REGISTRY

    current = DEFAULT_REGISTRY.read_as(GOLD_ITEM_SCHEMA_ID, record, source=source)
    return GoldItem.model_validate(current.model_dump())


def load_goldset(path: Path | str) -> list[GoldItem]:
    """Load + schema-validate a JSONL gold set, at whatever contract version it was written.

    Raises ValueError with line context.
    """
    path = Path(path)
    items: list[GoldItem] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(_gold_item_at_current(json.loads(line), f"{path}:{line_no}"))
            except Exception as exc:  # add file:line context, then re-raise
                raise ValueError(f"{path}:{line_no}: invalid gold item: {exc}") from exc
    return items


def dump_goldset(items: list[GoldItem], path: Path | str) -> None:
    """Write items as UTF-8 JSONL (one object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for item in items:
            out.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")
