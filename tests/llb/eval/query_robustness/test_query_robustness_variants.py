"""Focused tests split from ``test_query_robustness.py``."""

from pathlib import Path

import pytest
from tests.llb.eval._query_robustness_helpers import (
    APOSTROPHE_QUESTION,
)

from llb.board.io import read_case_rows
from llb.eval.query_robustness.uncertainty import (
    READING_DEGRADED,
    READING_INDISTINGUISHABLE,
    directional_comparison,
)
from llb.eval.query_robustness.variants import (
    ALL_VARIANT_CLASSES,
    APOSTROPHE_MIXED_SCRIPT,
    APOSTROPHE_VARIANT,
    KEYBOARD_TYPOS,
    MIXED_SCRIPT,
    VARIANT_CLASSES,
    generate_variant,
    parse_variant_classes,
)
from llb.eval.query_robustness.languages import LANGUAGE_VARIANT_CLASSES
from llb.rag.fusion_evidence.stats import bootstrap_index_sets


def test_signed_delta_stability_marks_a_degradation_near_miss_as_borderline():
    n = 30
    candidate = [0.0 if i < 7 else (1.0 if i == 7 else 0.0) for i in range(n)]
    baseline = [1.0 if i < 7 else 0.0 for i in range(n)]
    comparison = directional_comparison(
        candidate,
        baseline,
        bootstrap_index_sets(n, 2000, 13),
    )
    stability = comparison["stability"]
    assert stability["reading"] == READING_INDISTINGUISHABLE
    assert stability["looser_reading"] == READING_DEGRADED
    assert stability["borderline"] is True
    assert stability["p_positive"] < 0.05


@pytest.mark.parametrize("variant_class", ALL_VARIANT_CLASSES)
def test_variants_are_seeded_deterministic_and_non_identity(variant_class: str):
    kwargs = {"item_id": "q1", "seed": 17, "typo_rate": 0.1}
    if variant_class in LANGUAGE_VARIANT_CLASSES:
        kwargs["language_variants"] = {("q1", variant_class): "Другой вопрос?"}
    first = generate_variant(APOSTROPHE_QUESTION, variant_class, **kwargs)
    assert first == generate_variant(APOSTROPHE_QUESTION, variant_class, **kwargs)
    assert first != APOSTROPHE_QUESTION


def test_variant_rate_validation():
    with pytest.raises(ValueError, match="between 0 and 1"):
        generate_variant("query", KEYBOARD_TYPOS, item_id="q", seed=1, typo_rate=1.1)


def test_split_classes_apply_one_mechanism_each_and_compose_into_the_combined_class():
    kwargs = {"item_id": "q1", "seed": 17, "typo_rate": 0.5}
    question = APOSTROPHE_QUESTION
    apostrophes = generate_variant(question, APOSTROPHE_VARIANT, **kwargs)
    homoglyphs = generate_variant(question, MIXED_SCRIPT, **kwargs)
    combined = generate_variant(question, APOSTROPHE_MIXED_SCRIPT, **kwargs)
    # the apostrophe class re-types the apostrophe and touches nothing else
    assert (
        apostrophes.replace("ʼ", "'").replace("’", "'").replace("‘", "'").replace("`", "'")
        == question
    )
    assert "'" not in apostrophes
    # the homoglyph class substitutes Latin look-alikes and leaves the apostrophe alone
    assert "'" in homoglyphs
    assert any(char in homoglyphs for char in "aeikmnoptuxc")
    # the combined class is exactly the composition of the two halves at the same seed
    assert combined == generate_variant(homoglyphs, APOSTROPHE_VARIANT, **kwargs)
    assert combined != apostrophes and combined != homoglyphs


def test_apostrophe_class_is_a_no_op_on_a_question_without_an_apostrophe():
    kwargs = {"item_id": "q1", "seed": 17, "typo_rate": 0.1}
    assert generate_variant("Який закон?", APOSTROPHE_VARIANT, **kwargs) == "Який закон?"


def test_class_selection_defaults_to_single_mechanism_classes_and_rejects_unknown_names():
    assert APOSTROPHE_MIXED_SCRIPT not in VARIANT_CLASSES
    assert parse_variant_classes(f"{MIXED_SCRIPT}, {APOSTROPHE_VARIANT} ,{MIXED_SCRIPT}") == (
        MIXED_SCRIPT,
        APOSTROPHE_VARIANT,
    )
    assert parse_variant_classes(APOSTROPHE_MIXED_SCRIPT) == (APOSTROPHE_MIXED_SCRIPT,)
    with pytest.raises(ValueError, match="unknown query robustness variant class"):
        parse_variant_classes("apostrophe")
    with pytest.raises(ValueError, match="at least one"):
        parse_variant_classes(" , ")


def test_clean_baseline_reads_canonical_case_rows_not_aggregate_rows(tmp_path: Path):
    scores = tmp_path / "scores.jsonl"
    scores.write_text('{"item_id":"q1","objective_score":1,"retrieval_hit":1}\n')
    assert read_case_rows(scores)[0]["item_id"] == "q1"
    scores.write_text('{"model":"aggregate"}\n')
    with pytest.raises(ValueError, match="per-case score row"):
        read_case_rows(scores)
