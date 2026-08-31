"""Render the findings of an audit as decisions first, rows second.

A flat list of N rows invites an operator to fund N reviews. Rows that share a chunk are one piece
of evidence seen from several sides -- the same stale paragraph against six neighbours is one edit
-- so the section leads with the GROUPS, states the distinct units the row count rests on, and only
then prints every row. Nothing is dropped: `findings.jsonl` still carries one line per row, and the
row table below the groups is the same list the report always printed, in the same order.

The two tables are ordered on different questions, deliberately. Rows are read in file order, which
is the order a resolution lane consumes them in. Groups are read to decide what to fund first, so
they are ranked by `stake_key` -- work, then size, then score. The ids never move: a group id is
derived from `findings.jsonl` in file order and is the join key `groups.json` and `plan.json` use,
so ranking is a way of reading the table, not a renaming of what is in it.
"""

from llb.conflicts.grouping.census import (
    FindingGroup,
    census_phrase,
    finding_census,
    group_findings,
)
from llb.conflicts.constants import DECIDE_LABEL
from llb.conflicts.grouping.granularity import finding_granularity
from llb.conflicts.grouping.ranking import DecisionStake, stake_key as decision_stake_key
from llb.conflicts.models import AuditResult, Finding
from llb.conflicts.report.granularity import granularity_section
from llb.conflicts.report.projection import projected_columns, two_counts_paragraphs
from llb.core.contracts.common import JsonObject

_EXCERPT = 160
# Shared units beyond this many are summarised rather than listed, to keep the cell readable.
_MAX_LISTED_UNITS = 2


def _excerpt(text: str) -> str:
    """One-line excerpt safe to drop into a Markdown table cell."""
    flat = " ".join(text.split())
    if len(flat) > _EXCERPT:
        flat = flat[: _EXCERPT - 1].rstrip() + "…"
    return flat.replace("|", "\\|")


def side(finding: Finding, which: str) -> str:
    ref = finding.a if which == "a" else finding.b
    mark = "" if ref.offsets_exact else "~"
    return f"`{ref.doc_id}`{mark} [{ref.char_start}:{ref.char_end}]<br>{_excerpt(ref.text)}"


def _shared_cell(group: FindingGroup) -> str:
    """The unit(s) that make this group one decision instead of several rows."""
    units = group.shared_units
    if not units:
        return "-"
    listed = ", ".join(f"`{unit}`" for unit in units[:_MAX_LISTED_UNITS])
    extra = len(units) - _MAX_LISTED_UNITS
    return f"{listed} (+{extra} more)" if extra > 0 else listed


def _relations_cell(group: FindingGroup) -> str:
    return ", ".join(
        f"`{relation}` x{count}" if count > 1 else f"`{relation}`"
        for relation, count in group.relation_counts().items()
    )


def stake_key(group: FindingGroup) -> tuple[int, int, float, int]:
    """Rank a decision by what it costs an operator, not by which row happened to score highest.

    The ranking count is TO DECIDE (`decide_rows`), then how much of the list the decision settles,
    then the top score, then the group's own id so the order is total. It is not TO REVIEW, the
    count an operator funds, for a reason the report cannot engineer away: the review count is a
    property of a resolution POLICY, and an audit runs before one is chosen. `constants` states the
    pair; the resolution plan ranks on the review count once a policy exists.

    A score is the model's confidence in one pair; it says nothing about how much is at stake in
    the group that holds it -- and on a corpus where scores saturate, ranking on it is ranking on
    the identity tiebreak underneath.
    """
    return decision_stake_key(
        DecisionStake(
            group_index=group.index,
            decide_rows=group.decide_rows,
            rows=len(group.findings),
            top_score=group.top_score,
        )
    )


def _groups_table(groups: list[FindingGroup], projection: JsonObject) -> list[str]:
    """The decision table, widened by one column per projected policy and one per delta.

    With no policy named there are no extra columns at all -- not a zero, which would read as a
    measured "nothing to review" rather than as "nobody asked".
    """
    columns = projected_columns(projection)
    projected_header = "".join(f" {column.header} |" for column in columns)
    projected_divider = " --- |" * len(columns)
    lines = [
        f"| group | rows | {DECIDE_LABEL} |{projected_header} relations | shared unit "
        "| documents | top score |",
        f"| --- | --- | --- |{projected_divider} --- | --- | --- | --- |",
    ]
    for group in sorted(groups, key=stake_key):
        documents = ", ".join(f"`{doc}`" for doc in group.documents)
        projected = "".join(f" {column.cell(group.label)} |" for column in columns)
        lines.append(
            f"| {group.label} | {len(group.findings)} | {group.decide_rows} |{projected}"
            f" {_relations_cell(group)} | {_shared_cell(group)} | {documents} "
            f"| {group.top_score:.3f} |"
        )
    lines.append("")
    return lines


def _rows_table(groups: list[FindingGroup]) -> list[str]:
    lines = [
        "| group | relation | tier | score | newer | A | B |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for group in groups:
        for finding in group.findings:
            newer = finding.staleness.newer_side or "-"
            lines.append(
                f"| {group.label} | `{finding.relation}` | `{finding.tier}` | {finding.score:.3f} "
                f"| {newer} | {side(finding, 'a')} | {side(finding, 'b')} |"
            )
    lines.append("")
    return lines


def findings_section(result: AuditResult) -> list[str]:
    """Groups first (what an operator decides), then every row (what a resolution lane consumes).

    The optional TO REVIEW projection arrives as plain data on the result (`policy_projection`),
    computed by `policy_projection.py` above this layer. This module renders it and never derives
    it, which is what keeps the report free of the resolution vocabulary.
    """
    if not result.findings:
        return []
    groups = group_findings(result.findings)
    census = finding_census(result.findings)
    projection = result.policy_projection
    return (
        [
            "## Findings",
            "",
            f"**{census_phrase(census)}.** Rows that share a chunk -- or a document, where the "
            "tier compares whole documents -- are ONE decision, not several, so they are collapsed "
            "into a group below. The row count is not a count of independent conflicts; the group "
            "count is what an operator triages.",
            "",
            "### Decision groups",
            "",
            f"Ranked by **{DECIDE_LABEL}** -- rows whose relation is work rather than two facts "
            "coexisting -- then by how many rows the decision settles, then by the top score. "
            "Group ids are assigned in file order, so `G3` leading this table is a ranking, not a "
            "renumbering -- the row table below and `findings.jsonl` keep the file's own order.",
            "",
        ]
        + two_counts_paragraphs(projection)
        + _groups_table(groups, projection)
        + granularity_section(finding_granularity(result.findings))
        + [
            "### Rows",
            "",
            f"Every row `findings.jsonl` carries, grouped and with the **{DECIDE_LABEL}** rows "
            "first -- every relation but `complementary`, the same set the claim-tier precision "
            "block counts. Offsets are exact character positions in the source document; `~` marks "
            "a claim whose quote could not be located, where the span falls back to the enclosing "
            "chunk.",
            "",
        ]
        + _rows_table(groups)
    )
