"""Contracts for incremental store merge plans."""

from dataclasses import dataclass

from llb.core.contracts.rag import ChunkRecord
from llb.rag.duplicates.models import DuplicateStats
from llb.rag.refresh.lexical_merge import MergeEntry


@dataclass
class MergedUnits:
    """Merged build-order content and the stored-vector reuse source per row."""

    indexed: list[ChunkRecord]
    parents: list[ChunkRecord] | None
    row_sources: list[int | None]
    duplicates: DuplicateStats | None = None
    text_reused: int = 0

    @property
    def new_units(self) -> list[ChunkRecord]:
        return [unit for unit, source in zip(self.indexed, self.row_sources) if source is None]

    def lexical_entries(self) -> list[MergeEntry]:
        return [
            source if source is not None else str(unit["text"])
            for unit, source in zip(self.indexed, self.row_sources)
        ]
