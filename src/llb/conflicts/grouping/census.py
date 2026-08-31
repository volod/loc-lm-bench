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

from llb.conflicts.constants import decide_count, is_actionable
from llb.core.contracts.common import JsonObject

if TYPE_CHECKING:  # `models` renders its own census, so the dependency only runs one way
    from llb.conflicts.models import ClaimRef, Finding


def finding_sort_key(finding: "Finding") -> tuple[int, float, str]:
    """Actionable relations first, then by descending score, then stably by claim identity.

    "Actionable" is `constants.is_actionable`, the same predicate the claim-tier precision block
    counts on, so a row the audit counts as work to do can never sort below a row it counts as
    none.
    """
    priority = 0 if is_actionable(finding.relation) else 1
    return (priority, -finding.score, str(finding.key()))


def unit_key(ref: "ClaimRef") -> str:
    """The unit a side rests on: its chunk, or its document when the tier has no chunk."""
    return ref.chunk_id or ref.doc_id


@dataclass(frozen=True)
class PairUnits:
    """The units and documents one finding joins -- all the census and grouping ever read.

    Taking this instead of a `Finding` is what lets the audit (which holds objects) and the
    resolution lane (which holds `findings.jsonl` rows) group on ONE implementation, so a group id
    means the same thing in both artifacts.
    """

    left_unit: str
    right_unit: str
    left_doc: str
    right_doc: str

    @property
    def units(self) -> tuple[str, str]:
        return (self.left_unit, self.right_unit)

    @property
    def doc_pair(self) -> tuple[str, str]:
        first, second = sorted([self.left_doc, self.right_doc])
        return (first, second)


def pair_units(finding: "Finding") -> PairUnits:
    return PairUnits(
        left_unit=unit_key(finding.a),
        right_unit=unit_key(finding.b),
        left_doc=finding.a.doc_id,
        right_doc=finding.b.doc_id,
    )


def row_pair_units(row: JsonObject) -> PairUnits:
    """The same view over one `findings.jsonl` row."""
    sides = []
    for name in ("a", "b"):
        side = row.get(name)
        if not isinstance(side, dict) or not isinstance(side.get("doc_id"), str):
            raise ValueError(f"finding row side {name!r} is missing a string doc_id")
        doc_id = str(side["doc_id"])
        chunk_id = side.get("chunk_id")
        sides.append((str(chunk_id) if isinstance(chunk_id, str) and chunk_id else doc_id, doc_id))
    (left_unit, left_doc), (right_unit, right_doc) = sides
    return PairUnits(
        left_unit=left_unit, right_unit=right_unit, left_doc=left_doc, right_doc=right_doc
    )


def group_indices(pairs: list[PairUnits]) -> list[list[int]]:
    """Member indices per decision group, groups ordered by their first member."""
    parent = _union_find(pairs)

    def root(key: str) -> str:
        while parent[key] != key:
            key = parent[key]
        return key

    buckets: dict[str, list[int]] = {}
    for index, pair in enumerate(pairs):
        buckets.setdefault(root(pair.left_unit), []).append(index)
    return sorted(buckets.values(), key=lambda members: members[0])


def shared_unit_indices(pairs: list[PairUnits], within: list[int] | None = None) -> list[list[int]]:
    """Member indices per distinct piece of shared evidence, ordered by first member.

    Unlike :func:`group_indices`, this is a cover rather than a transitive partition: a row can
    join the group for each shared unit it carries. Units that join exactly the same rows collapse
    to one piece of evidence. Rows sharing no unit remain singleton pieces.

    ``within`` restricts the cover to one transitive decision group. Its resulting group count is
    that decision's chain length: how many distinct pieces of shared evidence the flat row group
    runs through.
    """
    indices = list(range(len(pairs))) if within is None else list(within)
    members_by_unit: dict[str, list[int]] = {}
    for index in indices:
        # `dict.fromkeys` de-duplicates a self-pair, whose two sides are the same unit.
        for unit in dict.fromkeys(pairs[index].units):
            members_by_unit.setdefault(unit, []).append(index)
    distinct: dict[tuple[int, ...], list[int]] = {}
    for _unit, members in sorted(members_by_unit.items()):
        if len(members) > 1:
            distinct.setdefault(tuple(members), members)
    groups = list(distinct.values())
    covered = {index for members in groups for index in members}
    groups += [[index] for index in indices if index not in covered]
    return sorted(groups, key=lambda members: (members[0], members))


def census_of(pairs: list[PairUnits]) -> JsonObject:
    """The distinct units behind a row count: what the count is evidence of, not how big it is."""
    groups = group_indices(pairs)
    return {
        "findings": len(pairs),
        "documents": len({doc for pair in pairs for doc in (pair.left_doc, pair.right_doc)}),
        "document_pairs": len({pair.doc_pair for pair in pairs}),
        "chunk_units": len({unit for pair in pairs for unit in pair.units}),
        "groups": len(groups),
        "largest_group": max((len(members) for members in groups), default=0),
    }


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
    def decide_rows(self) -> int:
        """The group's TO DECIDE count: rows whose relation is work, not facts that coexist.

        Named for its counterpart rather than after the predicate: the resolution plan carries a TO
        REVIEW count over the same group, the two differ in both directions, and the audit can only
        compute this one (see `constants.decide_count`). An audit has no policy, so it has no
        review count to print.
        """
        return decide_count(self.relation_counts())

    @property
    def chain_length(self) -> int:
        """Distinct pieces of shared evidence this transitive group runs through."""
        pairs = [pair_units(finding) for finding in self.findings]
        return len(shared_unit_indices(pairs))

    @property
    def top_score(self) -> float:
        return max((finding.score for finding in self.findings), default=0.0)

    @property
    def label(self) -> str:
        return f"G{self.index}"


def _union_find(pairs: list[PairUnits]) -> dict[str, str]:
    """Merge the two units every pair joins; returns the parent map."""
    parent: dict[str, str] = {}

    def root(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for pair in pairs:
        left, right = root(pair.left_unit), root(pair.right_unit)
        if left != right:
            parent[left] = right
    return parent


def group_findings(findings: list["Finding"]) -> list[FindingGroup]:
    """Findings grouped by shared unit, groups and members both in report order.

    Sorting first is what makes the group ids here equal the ids the resolution lane derives:
    `findings.jsonl` is written in this same order, so grouping its rows in file order reproduces
    these groups exactly.
    """
    ordered = sorted(findings, key=finding_sort_key)
    members = group_indices([pair_units(finding) for finding in ordered])
    return [
        FindingGroup(index=index, findings=tuple(ordered[i] for i in group))
        for index, group in enumerate(members, start=1)
    ]


def finding_census(findings: list["Finding"]) -> JsonObject:
    """The distinct units behind a row count, over findings the audit holds as objects."""
    return census_of([pair_units(finding) for finding in findings])


def rows_census(rows: list[JsonObject]) -> JsonObject:
    """The same census over `findings.jsonl` rows."""
    return census_of([row_pair_units(row) for row in rows])


def counted(count: int, noun: str) -> str:
    """`3 rows` / `1 row` -- one pluralizer, so no count in this lane reads as `1 rows`."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def census_units(census: JsonObject) -> str:
    """The units alone, to print after a count that is already on the line."""
    return (
        f"on {counted(int(census['documents']), 'document')} / "
        f"{counted(int(census['document_pairs']), 'document pair')} / "
        f"{counted(int(census['chunk_units']), 'chunk unit')}, in "
        f"{counted(int(census['groups']), 'group')} (largest {census['largest_group']})"
    )


def census_phrase(census: JsonObject) -> str:
    """The count and the units it rests on as one sentence."""
    return f"{counted(int(census['findings']), 'finding')} {census_units(census)}"


def relation_census(findings: list["Finding"]) -> dict[str, JsonObject]:
    """The same census per relation, so no single relation's count reads as N independent rows."""
    by_relation: dict[str, list["Finding"]] = {}
    for finding in findings:
        by_relation.setdefault(finding.relation, []).append(finding)
    return {relation: finding_census(rows) for relation, rows in sorted(by_relation.items())}
