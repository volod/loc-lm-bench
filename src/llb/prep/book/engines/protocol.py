"""The adapter seam: what an engine is asked, and the named refusal when it is not there.

Importing this module imports no recognition library. An engine that is not installed raises
`EngineUnavailable` -- that name is the refusal, and it carries no traceback.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import Field

from llb.core.contracts.data_prep.book_blocks import FormulaReading, TableReading
from llb.core.contracts.data_prep.corpus import DataPrepRow
from llb.core.fsutil import atomic_write_text
from llb.prep.book.constants import READING_FILENAME


class EngineUnavailable(Exception):
    """The named engine is not installed where this adapter runs it."""

    def __init__(self, engine: str, reason: str) -> None:
        self.engine = engine
        self.reason = reason
        super().__init__(f"engine {engine} unavailable: {reason}")


class EngineRejected(Exception):
    """The engine is installed, but this page's output is not usable."""

    def __init__(self, engine: str, reason: str) -> None:
        self.engine = engine
        self.reason = reason
        super().__init__(f"engine {engine} rejected: {reason}")


@dataclass(frozen=True)
class EngineRequest:
    """One page, one image variant, for one engine call."""

    image: Path
    pdf: Path
    work_dir: Path
    variant: str
    model: str | None
    width: int
    height: int
    language: str | None = None


@dataclass(frozen=True)
class Proposed:
    """One region an engine claims, in the rendered page's pixel space before clamping."""

    kind: str
    order: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    confidence: float | None
    formula: FormulaReading | None = None
    table: TableReading | None = None


@dataclass(frozen=True)
class EngineRead:
    """Everything one engine returned for one stage of one page."""

    engine: str
    model: str | None
    variant: str
    stage: str
    blocks: tuple[Proposed, ...]


class EngineAdapter(Protocol):
    """Layout, OCR, formula, and table engines share this. `open`/`close` bracket a batch."""

    engine_id: str
    stages: tuple[str, ...]

    def available(self) -> None:
        """Raise `EngineUnavailable` when this engine cannot run."""

    def open(self) -> None:
        """Start whatever the engine keeps across a batch (a model server). No-op if nothing."""

    def close(self) -> None:
        """Stop what `open` started."""

    def read(self, request: EngineRequest, stage: str) -> EngineRead:
        """Read one page. Lazy: heavy imports happen here or in the engine's own process."""


class ProposedRecord(DataPrepRow):
    """One proposed block as an engine stage writes it to `reading.json`."""

    kind: str = Field(min_length=1)
    order: int = Field(ge=0)
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    confidence: float | None = None
    formula: FormulaReading | None = None
    table: TableReading | None = None


class EngineReadingFile(DataPrepRow):
    """`reading.json`: one engine stage's normalized proposals, before the page ledger merges them."""

    engine: str = Field(min_length=1)
    model: str | None = None
    variant: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    blocks: list[ProposedRecord] = Field(default_factory=list)


def write_reading(directory: Path, read: EngineRead) -> None:
    """Write `reading.json` for a committed engine stage."""
    file = EngineReadingFile(
        engine=read.engine,
        model=read.model,
        variant=read.variant,
        stage=read.stage,
        blocks=[
            ProposedRecord(
                kind=block.kind,
                order=block.order,
                x0=block.x0,
                y0=block.y0,
                x1=block.x1,
                y1=block.y1,
                text=block.text,
                confidence=block.confidence,
                formula=block.formula,
                table=block.table,
            )
            for block in read.blocks
        ],
    )
    payload = file.model_dump(mode="json")
    atomic_write_text(
        directory / READING_FILENAME,
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
    )


def read_reading(path: Path) -> EngineRead:
    """Read `reading.json` back into the merge input."""
    file = EngineReadingFile.model_validate_json(path.read_text(encoding="utf-8"))
    return EngineRead(
        engine=file.engine,
        model=file.model,
        variant=file.variant,
        stage=file.stage,
        blocks=tuple(_proposed(block) for block in file.blocks),
    )


def _proposed(block: ProposedRecord) -> Proposed:
    return Proposed(
        block.kind,
        block.order,
        block.x0,
        block.y0,
        block.x1,
        block.y1,
        block.text,
        block.confidence,
        block.formula,
        block.table,
    )


def proposals_of(reads: Sequence[EngineRead]) -> tuple[Proposed, ...]:
    """Flatten, for tests that want every proposed box."""
    return tuple(block for read in reads for block in read.blocks)
