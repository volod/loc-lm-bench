"""Distinct-unit census and grouping over a finding list.

A finding COUNT is the first number the report and `summary.json` print, and on a corpus whose
conflicts concentrate that count is a multiple of the evidence behind it: rows that all quote the
same chunk are ONE decision an operator makes once, not six independent results. The claim-tier
precision block already clusters its bound on that fact, but only inside its own section, so every
other count still reads as N independent results.

This module is the shared answer. `finding_census` counts the distinct units a row count rests on,
and `group_findings` collapses rows that reuse a unit into the decision they actually represent.
Both are read-only views over the findings a run produced: nothing here suppresses, merges, or
reorders a row in `findings.jsonl`, which a resolution lane consumes in full.

The unit is the CHUNK each side rests on, falling back to its document for the hash and lexical
tiers, which compare whole documents and carry no chunk. Grouping is transitive over that unit --
a chunk that contradicts six others is one group, and so are three byte-identical copies of one
document -- because a shared unit is exactly what makes two rows the same piece of evidence.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from llb.conflicts.constants import REL_CONTRADICTS, REL_SUPERSEDED_BY
from llb.core.contracts.common import JsonObject

if TYPE_CHECKING:  # `models` renders its own census, so the dependency only runs one way
    from llb.conflicts.models import ClaimRef, Finding

# Findings whose relation means "someone must decide", listed first in the report.
ACTIONABLE = (REL_CONTRADICTS, REL_SUPERSEDED_BY)


def finding_sort_key(finding: "Finding") -> tuple[int, float, str]:
    """Actionable relations first, then by descending score, then stably by claim identity."""
    priority = 0 if finding.relation in ACTIONABLE else 1
    return (priority, -finding.score, str(finding.key()))


def unit_key(ref: "ClaimRef") -> str:
    """The unit a side rests on: its chunk, or its document when the tier has no chunk."""
    return ref.chunk_id or ref.doc_id


@dataclass(frozen=True)
class FindingGroup:
    """Findings joined by a shared unit -- one decision, however many rows carry it."""

    index: int
    findings: tuple["Finding", ...]

    @property
    def shared_units(self) -> tuple[str, ...]:
        """The units more than one row in the group rests on (empty for a single-row group)."""
        counts: dict[str, int] = {}
        for finding in self.findings:
            for key in (unit_key(finding.a), unit_key(finding.b)):
                counts[key] = counts.get(key, 0) + 1
        return tuple(key for key, count in sorted(counts.items()) if count > 1)

    @property
    def documents(self) -> tuple[str, ...]:
        return tuple(sorted({doc for f in self.findings for doc in (f.a.doc_id, f.b.doc_id)}))

    def relation_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.relation] = counts.get(finding.relation, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def label(self) -> str:
        return f"G{self.index}"


def _union_find(findings: list["Finding"]) -> dict[str, str]:
    """Merge every unit two sides of one finding share; returns the parent map."""
    parent: dict[str, str] = {}

    def root(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for finding in findings:
        left, right = root(unit_key(finding.a)), root(unit_key(finding.b))
        if left != right:
            parent[left] = right
    return parent


def group_findings(findings: list["Finding"]) -> list[FindingGroup]:
    """Findings grouped by shared unit, groups and members both in report order."""
    if not findings:
        return []
    parent = _union_find(findings)

    def root(key: str) -> str:
        while parent[key] != key:
            key = parent[key]
        return key

    buckets: dict[str, list["Finding"]] = {}
    for finding in findings:
        buckets.setdefault(root(unit_key(finding.a)), []).append(finding)
    ordered = sorted(
        (sorted(bucket, key=finding_sort_key) for bucket in buckets.values()),
        key=lambda bucket: finding_sort_key(bucket[0]),
    )
    return [FindingGroup(index=i, findings=tuple(b)) for i, b in enumerate(ordered, start=1)]


def finding_census(findings: list["Finding"]) -> JsonObject:
    """The distinct units behind a row count: what the count is evidence of, not how big it is."""
    groups = group_findings(findings)
    return {
        "findings": len(findings),
        "documents": len({doc for f in findings for doc in (f.a.doc_id, f.b.doc_id)}),
        "document_pairs": len({f.doc_pair() for f in findings}),
        "chunk_units": len({key for f in findings for key in (unit_key(f.a), unit_key(f.b))}),
        "groups": len(groups),
        "largest_group": max((len(group.findings) for group in groups), default=0),
    }


def _counted(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def census_units(census: JsonObject) -> str:
    """The units alone, to print after a count that is already on the line."""
    return (
        f"on {_counted(int(census['documents']), 'document')} / "
        f"{_counted(int(census['document_pairs']), 'document pair')} / "
        f"{_counted(int(census['chunk_units']), 'chunk unit')}, in "
        f"{_counted(int(census['groups']), 'group')} (largest {census['largest_group']})"
    )


def census_phrase(census: JsonObject) -> str:
    """The count and the units it rests on as one sentence."""
    return f"{_counted(int(census['findings']), 'finding')} {census_units(census)}"


def relation_census(findings: list["Finding"]) -> dict[str, JsonObject]:
    """The same census per relation, so no single relation's count reads as N independent rows."""
    by_relation: dict[str, list["Finding"]] = {}
    for finding in findings:
        by_relation.setdefault(finding.relation, []).append(finding)
    return {relation: finding_census(rows) for relation, rows in sorted(by_relation.items())}
