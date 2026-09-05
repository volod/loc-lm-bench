"""The compact schema for the decision-granularity block in ``summary.json``.

Schema 1 repeated rule prose and stored both a size list and its histogram, then added one chain
entry per quoted group although the report names only three. These tests pin schema 2's two folds,
its sublinear row growth, and the one gate that matters for an artifact migration: both forms must
render the same operator reading.
"""

import json

from llb.conflicts.bundle.fold import json_bytes
from llb.conflicts.grouping.census import PairUnits, group_indices, shared_unit_indices
from llb.conflicts.grouping.granularity import (
    GRANULARITY_SCHEMA_VERSION,
    RECORDED_CHAIN_LIMIT,
    RULES,
    distribution_size_counts,
    granularity_of,
)
from llb.conflicts.report.granularity import granularity_section


def _isolated(rows: int) -> list[PairUnits]:
    """Rows that share no unit: the worst case for an expanded list of group sizes."""
    return [
        PairUnits(
            left_unit=f"left-{index}",
            right_unit=f"right-{index}",
            left_doc=f"left-{index}.md",
            right_doc=f"right-{index}.md",
        )
        for index in range(rows)
    ]


def _chain(rows: int, prefix: str) -> list[PairUnits]:
    units = [f"{prefix}-{index}" for index in range(rows + 1)]
    return [
        PairUnits(
            left_unit=units[index],
            right_unit=units[index + 1],
            left_doc=f"{units[index]}.md",
            right_doc=f"{units[index + 1]}.md",
        )
        for index in range(rows)
    ]


def _several_chains() -> list[PairUnits]:
    return [pair for rows in range(2, 8) for pair in _chain(rows, f"chain-{rows}")]


def _legacy_form(pairs: list[PairUnits]) -> dict:
    """Rebuild schema 1 from current data, including every field that form repeated."""
    legacy = json.loads(json.dumps(granularity_of(pairs)))
    legacy["schema_version"] = 1
    legacy["unit"] = "legacy constant unit prose"
    for rule in RULES:
        distribution = legacy["rules"][rule]
        counts = distribution_size_counts(distribution)
        distribution["rule"] = rule
        distribution["description"] = "legacy constant rule prose"
        distribution["sizes"] = sorted(
            (int(size) for size, count in counts.items() for _ in range(count)), reverse=True
        )
        distribution["size_counts"] = counts
    quoted = group_indices(pairs)
    legacy.pop("quoted_group_chains", None)
    legacy["quoted_group_split"] = [
        {
            "group_id": f"G{index}",
            "rows": len(members),
            "shared_unit_groups": len(shared_unit_indices(pairs, members)),
        }
        for index, members in enumerate(quoted, start=1)
    ]
    return legacy


def test_schema_two_drops_constants_and_keeps_only_the_smaller_size_form():
    compact = granularity_of(_isolated(100))

    assert compact["schema_version"] == GRANULARITY_SCHEMA_VERSION == 2
    assert "unit" not in compact
    for distribution in compact["rules"].values():
        assert "rule" not in distribution and "description" not in distribution
        assert ("sizes" in distribution) != ("size_counts" in distribution)
        assert distribution["size_counts"] == {"1": 100}

    short = granularity_of(_chain(4, "short"))
    assert all("sizes" in distribution for distribution in short["rules"].values())


def test_ten_times_as_many_singleton_rows_costs_less_than_twice_as_many_bytes():
    hundred = json_bytes(granularity_of(_isolated(100)))
    thousand = json_bytes(granularity_of(_isolated(1_000)))

    assert thousand < hundred * 2, f"100 rows: {hundred} bytes; 1,000 rows: {thousand} bytes"


def test_old_and_new_forms_render_the_exact_same_reading():
    pairs = _several_chains()
    compact = granularity_of(pairs)
    legacy = _legacy_form(pairs)

    assert "quoted_group_chains" in compact
    assert len(compact["quoted_group_chains"]) == RECORDED_CHAIN_LIMIT
    assert "quoted_group_split" not in compact
    assert granularity_section(compact) == granularity_section(legacy)
    assert json_bytes(compact) < json_bytes(legacy)
