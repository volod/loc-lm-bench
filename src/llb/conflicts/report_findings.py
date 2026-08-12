"""Render the findings of an audit as decisions first, rows second.

A flat list of N rows invites an operator to fund N reviews. Rows that share a chunk are one piece
of evidence seen from several sides -- the same stale paragraph against six neighbours is one edit
-- so the section leads with the GROUPS, states the distinct units the row count rests on, and only
then prints every row. Nothing is dropped: `findings.jsonl` still carries one line per row, and the
row table below the groups is the same list the report always printed, in the same order.
"""

from llb.conflicts.census import FindingGroup, census_phrase, finding_census, group_findings
from llb.conflicts.models import AuditResult, Finding

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


def _groups_table(groups: list[FindingGroup]) -> list[str]:
    lines = [
        "| group | rows | relations | shared unit | documents | top score |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for group in groups:
        documents = ", ".join(f"`{doc}`" for doc in group.documents)
        top = max(finding.score for finding in group.findings)
        lines.append(
            f"| {group.label} | {len(group.findings)} | {_relations_cell(group)} "
            f"| {_shared_cell(group)} | {documents} | {top:.3f} |"
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
    """Groups first (what an operator decides), then every row (what a resolution lane consumes)."""
    if not result.findings:
        return []
    groups = group_findings(result.findings)
    census = finding_census(result.findings)
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
        ]
        + _groups_table(groups)
        + [
            "### Rows",
            "",
            "Every row `findings.jsonl` carries, grouped and with actionable relations first. "
            "Offsets are exact character positions in the source document; `~` marks a claim whose "
            "quote could not be located, where the span falls back to the enclosing chunk.",
            "",
        ]
        + _rows_table(groups)
    )
