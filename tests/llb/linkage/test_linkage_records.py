"""The record table contract: derived column types, and the errors that name the bad row."""

import json

import pytest
from llb.linkage.records import (
    ReviewerLabel,
    column_types,
    read_labels,
    read_records,
    validate_records,
)
from llb.linkage.comparison_spec import ComparisonSpec
from llb.linkage.spec import BlockingRule, LinkageSpec

SPEC = LinkageSpec(
    comparisons=(
        ComparisonSpec("name", "jaro_winkler", (0.9,)),
        ComparisonSpec("aliases", "array_intersect", (1.0,)),
        ComparisonSpec("vector", "cosine", (0.9,), dimension=2),
        ComparisonSpec("effective_date", "date_difference", (30.0,)),
    ),
    blocking_rules=(BlockingRule(("name",)),),
    retain_columns=("source_doc",),
)

RECORD = {
    "unique_id": "r1",
    "name": "Іван Франко",
    "aliases": ["Франко"],
    "vector": [0.1, 0.9],
    "effective_date": "2021-03-04",
    "source_doc": "d.md",
}


def test_column_types_come_from_the_comparison_kinds_not_the_data():
    assert column_types(SPEC) == {
        "unique_id": "VARCHAR",
        "name": "VARCHAR",
        "aliases": "VARCHAR[]",
        "vector": "DOUBLE[2]",
        "effective_date": "VARCHAR",
        "source_doc": "VARCHAR",
    }


def test_a_valid_table_passes():
    validate_records([RECORD, {**RECORD, "unique_id": "r2"}], SPEC)


@pytest.mark.parametrize(
    "records, message",
    [
        ([], "empty"),
        ([{k: v for k, v in RECORD.items() if k != "vector"}], "missing column"),
        ([{**RECORD, "unique_id": ""}], "has no 'unique_id' value"),
        ([RECORD, dict(RECORD)], "duplicate unique_id"),
        ([{**RECORD, "vector": [0.1]}], "declares dimension 2"),
    ],
)
def test_a_table_the_spec_cannot_be_applied_to_is_refused(records, message):
    with pytest.raises(ValueError, match=message):
        validate_records(records, SPEC)


def test_records_and_labels_read_from_jsonl(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(RECORD, ensure_ascii=False) + "\n\n", encoding="utf-8")
    assert read_records(path) == [RECORD]

    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        '{"left": "r1", "right": "r2", "match": true}\n'
        '{"left": "r1", "right": "r3", "match": false}\n',
        encoding="utf-8",
    )
    assert read_labels(labels, SPEC) == [
        ReviewerLabel("r1", "r2", True),
        ReviewerLabel("r1", "r3", False),
    ]


def test_a_label_row_naming_no_pair_is_refused(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text('{"match": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="names no record pair"):
        read_labels(path, SPEC)


def test_a_reviewer_decision_becomes_a_binary_clerical_score():
    assert ReviewerLabel("a", "b", True).score == 1.0
    assert ReviewerLabel("a", "b", False).score == 0.0
