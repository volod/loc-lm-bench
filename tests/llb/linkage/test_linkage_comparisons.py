"""Every comparison kind in the vocabulary must build, fit, and score on a real table.

The sample record table exercises five of the kinds; `levenshtein` on a code column, `cosine` on an
embedding column, and `set_overlap` on a shingle column are covered here on a tiny synthetic table,
so no kind is declared in the vocabulary without a run behind it.
"""

import pytest
from llb.linkage.comparisons import settings_uid
from llb.linkage.constants import COMPARISON_KINDS
from llb.linkage.comparison_spec import ComparisonSpec
from llb.linkage.spec import BlockingRule, LinkageSpec

VECTOR_WIDTH = 3

THRESHOLDS = {
    "exact": (),
    "levenshtein": (1.0, 3.0),
    "jaro_winkler": (0.96, 0.88),
    "array_intersect": (2.0, 1.0),
    "jaccard": (0.9, 0.8),
    "date_difference": (30.0, 365.0),
    "cosine": (0.95, 0.8),
    "set_overlap": (0.8, 0.5),
}

# `set_overlap` is the one kind with a SECOND ladder over the same column.
CONTAINMENT = {"set_overlap": (0.9, 0.6)}

COLUMNS = {
    "exact": "city",
    "levenshtein": "code",
    "jaro_winkler": "name",
    "array_intersect": "aliases",
    "jaccard": "address",
    "date_difference": "effective_date",
    "cosine": "vector",
    "set_overlap": "shingles",
}

RECORDS = [
    {
        "unique_id": "r1",
        "name": "Іван Франко",
        "code": "01001350",
        "address": "вулиця Університетська, 1",
        "aliases": ["Франко", "ЛНУ"],
        "city": "Львів",
        "effective_date": "2021-03-04",
        "vector": [1.0, 0.0, 0.0],
        "shingles": ["a", "b", "c", "d"],
    },
    {
        "unique_id": "r2",
        "name": "Івана Франка",
        "code": "01001351",
        "address": "вул. Університетська, 1",
        "aliases": ["Франко"],
        "city": "Львів",
        "effective_date": "2021-03-11",
        "vector": [0.98, 0.19, 0.0],
        "shingles": ["a", "b", "c", "e"],
    },
    {
        "unique_id": "r3",
        "name": "Леся Українка",
        "code": "77770001",
        "address": "вулиця Саксаганського, 97",
        "aliases": ["Українка"],
        "city": "Київ",
        "effective_date": "2019-01-02",
        "vector": [0.0, 0.0, 1.0],
        "shingles": ["x", "y", "z"],
    },
]


def _comparison(kind: str) -> ComparisonSpec:
    return ComparisonSpec(
        COLUMNS[kind],
        kind,
        THRESHOLDS[kind],
        dimension=VECTOR_WIDTH,
        containment_thresholds=CONTAINMENT.get(kind, ()),
    )


def _spec_for(kind: str) -> LinkageSpec:
    return LinkageSpec(
        comparisons=(
            _comparison(kind),
            ComparisonSpec("city", "exact") if kind != "exact" else ComparisonSpec("name", "exact"),
        ),
        blocking_rules=(BlockingRule(("city",)),),
        random_match_probability=0.1,
    )


@pytest.mark.heavy_env
@pytest.mark.parametrize("kind", COMPARISON_KINDS)
def test_each_comparison_kind_fits_and_scores(kind):
    pytest.importorskip("splink")
    from llb.linkage.engine import run_linkage

    spec = _spec_for(kind)
    result = run_linkage(RECORDS, spec)
    assert result.n_records == 3
    scored = {(p.left_id, p.right_id) for p in result.pairs}
    assert ("r1", "r2") in scored, f"{kind} produced no comparison for the blocked pair"
    assert {p.comparison for p in result.match_parameters} == set(spec.compared_columns)


@pytest.mark.heavy_env
def test_cosine_separates_a_near_vector_from_an_orthogonal_one():
    pytest.importorskip("splink")
    from llb.linkage.engine import run_linkage

    spec = LinkageSpec(
        comparisons=(
            ComparisonSpec("vector", "cosine", (0.95, 0.8), dimension=VECTOR_WIDTH),
            ComparisonSpec("name", "exact"),
        ),
        blocking_rules=(BlockingRule(("city",)),),
        retain_columns=("city",),
        random_match_probability=0.1,
    )
    one_city = [{**record, "city": "Львів"} for record in RECORDS]
    result = run_linkage(one_city, spec)
    agreement = {(p.left_id, p.right_id): p.agreement["vector"] for p in result.pairs}
    assert agreement[("r1", "r2")] > agreement[("r1", "r3")]


def test_the_model_id_is_derived_from_the_specification_not_random():
    spec = _spec_for("jaro_winkler")
    assert settings_uid(spec) == settings_uid(_spec_for("jaro_winkler"))
    assert settings_uid(spec) != settings_uid(_spec_for("levenshtein"))
