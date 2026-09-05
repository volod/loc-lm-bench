"""The second decision-grouping rule, and the size distribution that reads the two together.

The audit groups findings TRANSITIVELY over a shared unit (`census.group_indices`), and on a real
corpus that closure runs long: the goods budget-100 rows collapse into 6 groups, but the largest of
them chains 51 rows across three documents through 23 different shared chunks, which is not one
decision either. So the audit quotes a RANGE -- more than the group count, less than the row count
-- without saying where inside it the truth sits, and the row count is a very loose top end.

This module measures the alternative that tightens it. Under the SHARED-UNIT rule a group is one
unit that more than one row rests on, and a row joins every group whose unit it carries, so a row
that shares its left chunk with one neighbour and its right chunk with another appears TWICE. That
is the rule's defining property and also its limit:

- the transitive rule is a PARTITION -- every row lands in exactly one group, so the sizes sum to
  the row count and an operator can fund one review per group;
- the shared-unit rule is a COVER -- its sizes sum to more than the row count whenever any row
  carries two shared units, so its group count cannot be funded row for row.

Neither can therefore replace the other, which is why this module reports both and the audit keeps
quoting the partition. What the cover buys is the top end of the range: the number of distinct
pieces of shared evidence a chain actually runs through, which is far below the row count.
`QUOTED_RULE` names the rule the audit quotes and every renderer reads it from here.
"""

from statistics import median
from typing import TYPE_CHECKING

from llb.conflicts.bundle.fold import smaller_form
from llb.conflicts.grouping.census import (
    PairUnits,
    group_indices,
    row_pair_units,
    shared_unit_indices,
)
from llb.core.contracts.common import JsonObject

if TYPE_CHECKING:  # `models` renders this granularity, so the dependency only runs one way
    from llb.conflicts.models import Finding

GRANULARITY_SCHEMA_VERSION = 2
RECORDED_CHAIN_LIMIT = 3

RULE_TRANSITIVE = "transitive"
RULE_SHARED_UNIT = "shared_unit"
RULES = (RULE_TRANSITIVE, RULE_SHARED_UNIT)

# The rule the audit's group ids, `groups.json`, `plan.json`, and the decision table are built on.
# It is the transitive one because it is the only PARTITION of the rows: a group count an operator
# funds has to account for every row exactly once, and the shared-unit cover does not.
QUOTED_RULE = RULE_TRANSITIVE


def _distribution(groups: list[list[int]], rows: int) -> JsonObject:
    """One rule's group-size distribution, plus whether it accounts for each row exactly once."""
    sizes = sorted((len(members) for members in groups), reverse=True)
    memberships = sum(sizes)
    counts = _size_counts(sizes)
    multiple = _rows_in_multiple_groups(groups)
    return {
        "groups": len(groups),
        # Keep exactly one form. A short, irregular distribution costs less as a list; repeated
        # sizes collapse into the histogram the report reads. The fold cannot make a run larger.
        **smaller_form({"size_counts": counts}, {"sizes": sizes}),
        "largest_group": sizes[0] if sizes else 0,
        "median_group": float(median(sizes)) if sizes else 0.0,
        "singletons": counts.get("1", 0),
        "memberships": memberships,
        "rows_in_multiple_groups": multiple,
        # A partition can be funded one review per group; a cover cannot, because its sizes
        # double-count the rows that carry two shared units.
        "partition": memberships == rows and multiple == 0,
    }


def _size_counts(sizes: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for size in sizes:
        key = str(size)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def distribution_size_counts(distribution: JsonObject) -> dict[str, int]:
    """Read a size distribution from either schema-2 form or schema 1's redundant pair."""
    counts = distribution.get("size_counts")
    if isinstance(counts, dict):
        return {str(size): int(count) for size, count in counts.items()}
    sizes = distribution.get("sizes")
    return _size_counts([int(size) for size in sizes]) if isinstance(sizes, list) else {}


def _rows_in_multiple_groups(groups: list[list[int]]) -> int:
    seen: set[int] = set()
    repeated: set[int] = set()
    for members in groups:
        for index in members:
            (repeated if index in seen else seen).add(index)
    return len(repeated)


def granularity_of(pairs: list[PairUnits]) -> JsonObject:
    """Both grouping rules over one finding list, with the rule the audit quotes named.

    `decision_range` is the pair an operator should read: the transitive count bounds the number of
    decisions from below and the shared-unit count from above. It replaces the row count as the top
    end, which over-states the work by however many rows share a chunk.
    """
    rows = len(pairs)
    quoted_groups = group_indices(pairs)
    transitive = _distribution(quoted_groups, rows)
    shared = _distribution(shared_unit_indices(pairs), rows)
    return {
        "schema_version": GRANULARITY_SCHEMA_VERSION,
        "quoted_rule": QUOTED_RULE,
        "rows": rows,
        "rules": {RULE_TRANSITIVE: transitive, RULE_SHARED_UNIT: shared},
        "decision_range": [int(transitive["groups"]), int(shared["groups"])],
        **_quoted_group_record(pairs, quoted_groups),
    }


def _quoted_group_record(pairs: list[PairUnits], quoted: list[list[int]]) -> JsonObject:
    """The longest chains the report reads, or the full split when that is smaller.

    This is the bridge between the two counts and the reason the range is readable at all: a group
    of 51 rows resting on 23 shared chunks is a long chain, while a group of 6 rows resting on one
    chunk is genuinely one decision, and the quoted group count alone cannot tell them apart. Group
    ids follow `census.group_indices`, so they are the ids `groups.json` and `plan.json` address.

    Schema 1 recorded an entry for every group although the report names at most three. Schema 2
    caps that growth by recording only those three, but keeps the complete form when it is already
    shorter (which can happen only at or below the cap).
    """
    split: list[JsonObject] = [
        {
            "group_id": f"G{index}",
            "rows": len(members),
            "shared_unit_groups": len(shared_unit_indices(pairs, members)),
        }
        for index, members in enumerate(quoted, start=1)
    ]
    chains = sorted(
        (entry for entry in split if int(entry["shared_unit_groups"]) > 1),
        key=_chain_key,
    )[:RECORDED_CHAIN_LIMIT]
    compact = {"quoted_group_chains": chains} if chains else {}
    return smaller_form(compact, {"quoted_group_split": split})


def _chain_key(entry: JsonObject) -> tuple[int, str]:
    return (-int(entry["shared_unit_groups"]), str(entry["group_id"]))


def reported_chains(granularity: JsonObject) -> list[JsonObject]:
    """The chains a report names, read identically from schema 1 or either schema-2 fold."""
    entries = granularity.get("quoted_group_chains")
    if not isinstance(entries, list):
        entries = granularity.get("quoted_group_split")
    if not isinstance(entries, list):
        return []
    chains = [
        entry
        for entry in entries
        if isinstance(entry, dict) and int(entry.get("shared_unit_groups", 0)) > 1
    ]
    return sorted(chains, key=_chain_key)[:RECORDED_CHAIN_LIMIT]


def rows_granularity(rows: list[JsonObject]) -> JsonObject:
    """Both rules over `findings.jsonl` rows -- the form every recomputation over a run uses."""
    return granularity_of([row_pair_units(row) for row in rows])


def finding_granularity(findings: list["Finding"]) -> JsonObject:
    """Both rules over findings the audit holds as objects.

    Sorted into report order first, for the same reason `census.group_findings` sorts: the group
    ids in the recorded chain summaries have to be the ids `findings.jsonl`, `groups.json`, and
    `plan.json` address, and those come from the file's own order.
    """
    from llb.conflicts.grouping.census import finding_sort_key, pair_units

    ordered = sorted(findings, key=finding_sort_key)
    return granularity_of([pair_units(finding) for finding in ordered])
