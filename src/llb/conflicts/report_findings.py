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

from llb.conflicts.census import FindingGroup, census_phrase, finding_census, group_findings
from llb.conflicts.constants import DECIDE_LABEL, REVIEW_LABEL
from llb.conflicts.models import AuditResult, Finding
from llb.core.contracts.common import JsonObject

_EXCERPT = 160
# Shared units beyond this many are summarised rather than listed, to keep the cell readable.
_MAX_LISTED_UNITS = 2
# The opt-in projection's column, named so the word an operator funds is never printed bare: this
# module renders a projection it is HANDED and never computes one, which is what keeps the
# resolution vocabulary out of the report layer (see `policy_projection.py`).
_PROJECTED_LABEL = f"{REVIEW_LABEL} (projected)"


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
    return (-group.decide_rows, -len(group.findings), -group.top_score, group.index)


def _projected_cell(projection: JsonObject, group: FindingGroup) -> str:
    """One group's projected review count, read out of data this module did not compute.

    Empty when no policy was named, which is how the table keeps exactly the columns it had before
    the flag existed instead of printing a zero that would read as a measured "nothing to review".
    """
    if not projection:
        return ""
    groups = projection.get("groups")
    value = groups.get(group.label) if isinstance(groups, dict) else None
    return f" {int(value) if isinstance(value, int) else 0} |"


def _groups_table(groups: list[FindingGroup], projection: JsonObject) -> list[str]:
    projected_header = f" {_PROJECTED_LABEL} |" if projection else ""
    projected_divider = " --- |" if projection else ""
    lines = [
        f"| group | rows | {DECIDE_LABEL} |{projected_header} relations | shared unit "
        "| documents | top score |",
        f"| --- | --- | --- |{projected_divider} --- | --- | --- | --- |",
    ]
    for group in sorted(groups, key=stake_key):
        documents = ", ".join(f"`{doc}`" for doc in group.documents)
        lines.append(
            f"| {group.label} | {len(group.findings)} | {group.decide_rows} |"
            f"{_projected_cell(projection, group)}"
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


def _two_counts_paragraphs(projection: JsonObject) -> list[str]:
    """Why the audit prints one count, and -- when asked -- what the projected second one is.

    The claim never weakens: the audit still cannot MEASURE the review count, because measuring it
    means running the policy over the corpus an operator chose. What `--project-policy` adds is one
    named policy replayed over these same rows, which is a projection and says so wherever it
    appears.
    """
    measured = "an audit cannot measure it" if projection else "an audit cannot print it"
    paragraphs = [
        f"`{DECIDE_LABEL}` is NOT the count of human reviews this corpus costs. That second "
        f"count, **{REVIEW_LABEL}**, is a property of the resolution POLICY -- the rows "
        "`resolve-corpus-conflicts` leaves at `review_required` -- so it does not exist until "
        f"a policy is chosen and {measured}. The two diverge in both directions: "
        "an accepted `drop_duplicate` is to decide and not to review, while every semantic-tier "
        "duplicate is to review under every policy because that tier has no deletion authority. "
        f"`plan.json` prints both per group (`decide_rows`, `review_rows`) and ranks the review "
        "ledger on the second, which is the one an operator funds.",
        "",
    ]
    if not projection:
        return paragraphs
    policy = str(projection.get("policy", ""))
    return paragraphs + [
        f"The **{_PROJECTED_LABEL}** column is that second count PROJECTED under policy "
        f"`{policy}`, because `--project-policy {policy}` was passed: each group's rows replayed "
        f"through `resolve-corpus-conflicts --policy {policy}` and counted where that policy "
        "leaves `review_required`. It is a projection under a named policy, never a measurement --"
        " a different policy gives a different column, a reviewer's decisions can still move it, "
        "and the ranking below is unchanged (it still ranks on "
        f"`{DECIDE_LABEL}`, the only count that exists without a policy). The measured count is "
        "`review_rows` in the `plan.json` that command writes, and it equals this column group for "
        "group under the same policy.",
        "",
    ]


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
        + _two_counts_paragraphs(projection)
        + _groups_table(groups, projection)
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
