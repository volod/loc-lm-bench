"""Decision groups as machine-readable output, for the audit sidecar and the resolution plan.

The report collapses rows that share a chunk into one decision, but a consumer reading
`findings.jsonl` sees only rows -- so the resolution planner would plan one overlay edit per ROW,
which on a concentrated corpus is one edit planned six times. This module emits the same grouping
the report renders, addressed by the SAME `finding_id` the plan uses, so a group id means one thing
across `groups.json`, `plan.json`, and `resolution_review.jsonl`.

Everything here reads `findings.jsonl` ROWS rather than `Finding` objects: the audit passes the
rows it is about to write and the resolution lane passes the rows it just read, so neither side can
derive a grouping the other does not have. `findings.jsonl` itself is untouched -- one line per row.
"""

from llb.conflicts.grouping.census import PairUnits, group_indices, row_pair_units, rows_census
from llb.conflicts.constants import decide_count
from llb.conflicts.tiers.hashing import finding_id
from llb.core.contracts.common import JsonObject

GROUPS_SCHEMA_VERSION = 1
# What a group's members share, spelled out once for every consumer of the sidecar.
UNIT_DESCRIPTION = (
    "the chunk each side of a finding rests on, or its document for the hash and lexical tiers, "
    "which compare whole documents; rows are grouped by the transitive closure over that unit"
)


def _shared_units(pairs: list[PairUnits]) -> list[str]:
    """The units more than one member row rests on (empty for a single-row group)."""
    counts: dict[str, int] = {}
    for pair in pairs:
        for unit in pair.units:
            counts[unit] = counts.get(unit, 0) + 1
    return [unit for unit, count in sorted(counts.items()) if count > 1]


def _relation_counts(rows: list[JsonObject]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        relation = str(row.get("relation", ""))
        counts[relation] = counts.get(relation, 0) + 1
    return dict(sorted(counts.items()))


def _editions(documents: list[str], editions: dict[str, str]) -> list[str]:
    """The edition groups a decision group's documents were proposed into, deduplicated.

    A decision group is rows that share a chunk or a document; an edition group is documents one
    linkage fit put in one identity. Naming the second on the first is what lets a reviewer see
    that four rows about four files are four editions of ONE document -- which is a different
    amount of work from four documents that happen to overlap.
    """
    named = {editions[doc] for doc in documents if doc in editions}
    return sorted(named)


def _summary(
    index: int, rows: list[JsonObject], pairs: list[PairUnits], editions: dict[str, str]
) -> JsonObject:
    """One group's public record: what it is, what it rests on, and which rows it speaks for."""
    documents = sorted({doc for pair in pairs for doc in pair.doc_pair})
    grouped = _editions(documents, editions)
    return {
        "group_id": f"G{index}",
        "rows": len(rows),
        "finding_ids": [finding_id(row) for row in rows],
        "relations": _relation_counts(rows),
        "shared_units": _shared_units(pairs),
        "documents": documents,
        # Present only when the edition-linkage lane ran, so a bundle written without it is
        # byte-identical to one written before the lane existed.
        **({"edition_groups": grouped} if grouped else {}),
        # Lists, not tuples: a consumer comparing this against the JSON on disk must not see a
        # difference the serializer invented.
        "document_pairs": [list(pair) for pair in sorted({pair.doc_pair for pair in pairs})],
        "top_score": max((float(row.get("score", 0.0)) for row in rows), default=0.0),
    }


def group_summaries(
    rows: list[JsonObject], editions: dict[str, str] | None = None
) -> list[JsonObject]:
    """One summary per decision group, in the order the rows are listed."""
    pairs = [row_pair_units(row) for row in rows]
    named = editions or {}
    return [
        _summary(index, [rows[at] for at in members], [pairs[at] for at in members], named)
        for index, members in enumerate(group_indices(pairs), start=1)
    ]


def groups_document(
    rows: list[JsonObject], *, findings_sha256: str, editions: dict[str, str] | None = None
) -> JsonObject:
    """The `groups.json` sidecar: the audit's grouping, pinned to the rows it was derived from."""
    return {
        "schema_version": GROUPS_SCHEMA_VERSION,
        "unit": UNIT_DESCRIPTION,
        "source_findings_sha256": findings_sha256,
        "census": rows_census(rows),
        "groups": group_summaries(rows, editions),
    }


def group_decisions(summaries: list[JsonObject], items: list[JsonObject]) -> list[JsonObject]:
    """One decision per group: its members' single action, or an honest record that they differ.

    The decision NEVER invents authority a member row does not already carry -- the overlay is
    still built from the per-row items. `action` is the action every member agreed on and is null
    when they did not, so a mixed group reads as mixed instead of as one confident edit.

    Both counts of the group's work ride here, because this is the only artifact that holds both.
    `decide_rows` is the audit's relation-based count, restated from the summary so a plan reader
    never has to re-derive the number the report ranked on; `review_rows` is this policy's own
    count of rows still needing a human. See `constants` for why neither can serve both roles.
    """
    by_id = {str(item.get("finding_id")): item for item in items}
    decisions: list[JsonObject] = []
    for summary in summaries:
        members = [by_id[fid] for fid in summary["finding_ids"] if fid in by_id]
        actions = _value_counts(members, "action")
        review_rows = sum(1 for item in members if item.get("status") == "review_required")
        decisions.append(
            {
                "group_id": summary["group_id"],
                "rows": summary["rows"],
                "finding_ids": list(summary["finding_ids"]),
                "relations": dict(summary["relations"]),
                "documents": list(summary["documents"]),
                "shared_units": list(summary["shared_units"]),
                "actions": actions,
                "action": next(iter(actions)) if len(actions) == 1 else None,
                "decide_rows": decide_count(summary["relations"]),
                "review_rows": review_rows,
                "status": "review_required" if review_rows else "accepted",
            }
        )
    return decisions


def _value_counts(items: list[JsonObject], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(field, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
