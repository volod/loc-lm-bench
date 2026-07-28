"""Builder for interleaving fresh and reused units during an incremental refresh."""

from dataclasses import dataclass, field

from llb.core.contracts.rag import ChunkRecord
from llb.rag.refresh.merge_models import MergedUnits


def group_by_doc(records: list[ChunkRecord]) -> dict[str, list[int]]:
    """Build-order ordinals per doc_id, order preserved."""
    out: dict[str, list[int]] = {}
    for ordinal, record in enumerate(records):
        out.setdefault(str(record["doc_id"]), []).append(ordinal)
    return out


def annotation_only_sources(
    fresh: list[ChunkRecord], old_chunks: list[ChunkRecord], old_ordinals: list[int]
) -> list[int | None]:
    """Reuse old rows when re-chunking changed only record annotations."""
    if len(fresh) != len(old_ordinals):
        return [None] * len(fresh)
    spans_unchanged = all(
        unit["char_start"] == old_chunks[ordinal]["char_start"]
        and unit["char_end"] == old_chunks[ordinal]["char_end"]
        and unit["text"] == old_chunks[ordinal]["text"]
        for unit, ordinal in zip(fresh, old_ordinals)
    )
    return list(old_ordinals) if spans_unchanged else [None] * len(fresh)


@dataclass
class MergeAssemblyBuilder:
    """Accumulate indexed units, optional parents, and vector-row reuse sources."""

    old_chunks: list[ChunkRecord]
    old_parents: list[ChunkRecord] | None
    include_parents: bool
    old_ordinals: dict[str, list[int]] = field(init=False)
    old_parent_ordinals: dict[str, list[int]] = field(init=False)
    indexed: list[ChunkRecord] = field(default_factory=list)
    parents: list[ChunkRecord] | None = field(init=False)
    row_sources: list[int | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.old_ordinals = group_by_doc(self.old_chunks)
        self.old_parent_ordinals = (
            group_by_doc(self.old_parents) if self.old_parents is not None else {}
        )
        self.parents = [] if self.include_parents else None

    def add_fresh(
        self,
        doc_id: str,
        fresh: list[ChunkRecord],
        fresh_parents: list[ChunkRecord],
        *,
        annotation_only: bool,
    ) -> None:
        self.indexed.extend(fresh)
        sources = (
            annotation_only_sources(fresh, self.old_chunks, self.old_ordinals.get(doc_id, []))
            if annotation_only
            else [None] * len(fresh)
        )
        self.row_sources.extend(sources)
        if self.parents is not None:
            self.parents.extend(fresh_parents)

    def add_reused(self, doc_id: str) -> None:
        for ordinal in self.old_ordinals.get(doc_id, []):
            self.indexed.append(self.old_chunks[ordinal])
            self.row_sources.append(ordinal)
        if self.parents is not None and self.old_parents is not None:
            self.parents.extend(
                self.old_parents[ordinal] for ordinal in self.old_parent_ordinals.get(doc_id, [])
            )

    def build(self) -> MergedUnits:
        return MergedUnits(
            indexed=self.indexed,
            parents=self.parents,
            row_sources=self.row_sources,
        )
