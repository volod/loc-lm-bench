"""Link the notes of one vault and record every miss.

The counts name kinds and numbers. They do not quote the sentence the number
was read in.
"""

import json
from dataclasses import dataclass, field

from llb.core.contracts.data_prep.book_structure import DocumentStructure
from llb.prep.book.render.catalog import Catalog, build_catalog
from llb.prep.book.render.grammar import ReferenceGrammar
from llb.prep.book.render.link import Event, link_text
from llb.prep.book.render.types import VaultFile

_BUCKETS = {"resolved": "resolved", "ambiguous": "ambiguous", "unresolved": "unresolved"}


@dataclass
class ReferenceCounts:
    """How many citations of each kind resolved, and which keys were ambiguous."""

    grammar: str
    resolved: dict[str, int] = field(default_factory=dict)
    ambiguous: dict[str, int] = field(default_factory=dict)
    unresolved: dict[str, int] = field(default_factory=dict)
    ambiguous_keys: set[str] = field(default_factory=set)
    gaps: int = 0

    def add(self, event: Event) -> None:
        outcome = event.outcome
        if outcome == "gap":
            self.gaps += 1
            outcome = "unresolved"
        bucket = _BUCKETS.get(outcome)
        if bucket is None:
            return
        current = getattr(self, bucket)
        current[event.kind] = current.get(event.kind, 0) + 1
        if event.outcome == "ambiguous" and event.key:
            self.ambiguous_keys.add(event.key)


def apply_references(
    files: list[VaultFile], structure: DocumentStructure, grammar: ReferenceGrammar
) -> tuple[ReferenceCounts, Catalog]:
    """Rewrite each note in place. Returns the counts and the catalog the lists use."""
    catalog = build_catalog(files, structure, grammar)
    counts = ReferenceCounts(grammar.name)
    for file in files:
        file.body, events = link_text(
            file.body,
            here=file.stem,
            catalog=catalog,
            grammar=grammar,
            index=file.stem in catalog.index_stems,
        )
        for event in events:
            counts.add(event)
    return counts, catalog


def reference_json(counts: ReferenceCounts) -> bytes:
    """``metadata/reference_links.json``. Counts and ambiguous keys only."""
    payload = {
        "ambiguous": _compact(counts.ambiguous),
        "ambiguous_keys": sorted(counts.ambiguous_keys),
        "gaps": counts.gaps,
        "grammar": counts.grammar,
        "resolved": _compact(counts.resolved),
        "unresolved": _compact(counts.unresolved),
    }
    text = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    return text.encode("ascii")


def reference_line(counts: ReferenceCounts) -> str:
    """One log line. It names totals, not the cited numbers."""
    return (
        "references "
        f"resolved={_total(counts.resolved)} "
        f"ambiguous={_total(counts.ambiguous)} "
        f"unresolved={_total(counts.unresolved)}"
    )


def _compact(counts: dict[str, int]) -> dict[str, int]:
    return {key: counts[key] for key in sorted(counts) if counts[key]}


def _total(counts: dict[str, int]) -> int:
    return sum(counts.values())
