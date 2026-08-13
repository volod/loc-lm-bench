"""Render the two grouping rules: the audit's own section, and a cross-run recomputation.

Both readers ask the same question -- how many decisions is this row count, really -- so both are
rendered from the one payload `granularity.py` computes. The audit answers it for the run it just
produced; `compare-conflict-granularity` answers it for run artifacts already on disk, which is how
the rule is compared across corpora without re-adjudicating a single pair.
"""

from llb.conflicts.census import counted
from llb.conflicts.granularity import (
    QUOTED_RULE,
    RULE_SHARED_UNIT,
    RULE_TRANSITIVE,
    RULES,
)
from llb.core.contracts.common import JsonObject

_TABLE_HEADER = (
    "| rule | groups | largest | median | rows in >1 group | memberships | partition |",
    "| --- | --- | --- | --- | --- | --- | --- |",
)
# Longest chain named in prose; beyond this the table is the reading, not a sentence per group.
_NAMED_CHAINS = 3


def _sizes_phrase(distribution: JsonObject) -> str:
    """`1 x18, 2 x13, 51 x1` -- the size histogram, which is what a long `sizes` list is read for."""
    counts = distribution.get("size_counts") or {}
    return ", ".join(f"{size} x{count}" for size, count in sorted(counts.items(), key=_by_size))


def _by_size(item: tuple[str, int]) -> int:
    return int(item[0])


def _rule_row(distribution: JsonObject, *, label: str = "") -> str:
    rule = str(distribution["rule"])
    name = label or (f"`{rule}` (quoted)" if rule == QUOTED_RULE else f"`{rule}`")
    partition = "yes" if distribution["partition"] else "no"
    return (
        f"| {name} | {distribution['groups']} | {distribution['largest_group']} "
        f"| {float(distribution['median_group']):.1f} "
        f"| {distribution['rows_in_multiple_groups']} | {distribution['memberships']} "
        f"| {partition} |"
    )


def rule_table(granularity: JsonObject) -> list[str]:
    """Both rules side by side over one run's rows."""
    rules = granularity["rules"]
    return [*_TABLE_HEADER, *(_rule_row(rules[rule]) for rule in RULES), ""]


def _chain_phrase(granularity: JsonObject) -> str:
    """The quoted groups whose chain runs through the most distinct pieces of evidence."""
    split = sorted(
        granularity.get("quoted_group_split") or [],
        key=lambda entry: (-int(entry["shared_unit_groups"]), str(entry["group_id"])),
    )
    chains = [entry for entry in split if int(entry["shared_unit_groups"]) > 1][:_NAMED_CHAINS]
    if not chains:
        return (
            "No quoted group splits: every group rests on a single shared unit, so the two rules "
            "agree and the range is a point."
        )
    named = "; ".join(
        f"{entry['group_id']}'s {counted(int(entry['rows']), 'row')} run through "
        f"{counted(int(entry['shared_unit_groups']), 'distinct shared unit')}"
        for entry in chains
    )
    return (
        f"Where the range comes from: {named}. A group resting on ONE shared unit is genuinely one "
        "decision; a long chain is several, and the quoted count alone cannot tell them apart."
    )


def granularity_section(granularity: JsonObject) -> list[str]:
    """The audit's own reading: which rule it quotes, why, and where the other one lands."""
    rules = granularity["rules"]
    quoted, cover = rules[RULE_TRANSITIVE], rules[RULE_SHARED_UNIT]
    low, high = granularity["decision_range"]
    rows = int(granularity["rows"])
    return [
        "### How many decisions the row count is",
        "",
        f"{counted(rows, 'row')} over-states the decisions and "
        f"{counted(int(quoted['groups']), 'group')} under-states them, so the number an operator "
        f"funds sits in a RANGE: **{low} to {high}**. Both ends are group counts, measured under "
        "the two rules below, and the top end replaces the row count in that role.",
        "",
        *rule_table(granularity),
        f"**The audit quotes `{QUOTED_RULE}`** -- its group ids, `groups.json`, `plan.json`, and "
        "the decision table above are all built on it -- because it is the only PARTITION of the "
        "rows: one review per group accounts for every row exactly once. "
        f"`{RULE_SHARED_UNIT}` is a COVER, one group per unit that more than one row rests on, so "
        f"a row carrying two shared units joins two groups: here "
        f"{cover['rows_in_multiple_groups']} of {rows} rows do, and its "
        f"{counted(int(cover['groups']), 'group')} hold {cover['memberships']} memberships. That "
        "count cannot be funded one review each, which is why it bounds the range instead of "
        "replacing the quoted count.",
        "",
        _chain_phrase(granularity),
        "",
        f"Group sizes under `{RULE_TRANSITIVE}`: {_sizes_phrase(quoted)}. "
        f"Under `{RULE_SHARED_UNIT}`: {_sizes_phrase(cover)}.",
        "",
    ]


def comparison_report(entries: list[JsonObject]) -> str:
    """Both rules across several runs, for the cross-corpus reading of which rule to quote.

    Each entry is `{"label", "source", "granularity"}` -- one audited run's rows, recomputed. No
    entry is re-adjudicated: the rules read `findings.jsonl`, so a run measured months ago on
    another host answers this question without a model call.
    """
    lines = [
        "# Decision-group granularity: transitive closure against shared units",
        "",
        "Two rules over the same rows. `transitive` joins findings through the closure over a "
        "shared unit and is what the audit quotes; `shared_unit` makes one group per unit that "
        "more than one row rests on, so a row can join two groups. The decisions an operator funds "
        "sit between the two counts.",
        "",
        "| run | rows | transitive | shared unit | decision range | rows in >1 shared-unit group |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        granularity = entry["granularity"]
        rules = granularity["rules"]
        low, high = granularity["decision_range"]
        lines.append(
            f"| `{entry['label']}` | {granularity['rows']} "
            f"| {rules[RULE_TRANSITIVE]['groups']} | {rules[RULE_SHARED_UNIT]['groups']} "
            f"| {low} - {high} | {rules[RULE_SHARED_UNIT]['rows_in_multiple_groups']} |"
        )
    lines.append("")
    for entry in entries:
        lines += _run_section(entry)
    return "\n".join(lines)


def _run_section(entry: JsonObject) -> list[str]:
    granularity = entry["granularity"]
    return [
        f"## {entry['label']}",
        "",
        f"- source: `{entry['source']}`",
        "",
        *rule_table(granularity),
        _chain_phrase(granularity),
        "",
        f"Group sizes under `{RULE_TRANSITIVE}`: "
        f"{_sizes_phrase(granularity['rules'][RULE_TRANSITIVE])}. "
        f"Under `{RULE_SHARED_UNIT}`: {_sizes_phrase(granularity['rules'][RULE_SHARED_UNIT])}.",
        "",
    ]
