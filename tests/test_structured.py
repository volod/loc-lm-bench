"""M5.4 structured-output -- Pydantic conformance + field accuracy."""

import pytest

from llb.bench import structured as bench_st
from llb.scoring import structured
from llb.scoring.aggregate import TIER_STRUCTURED

SCHEMA = {
    "name": {"type": "string", "required": True},
    "age": {"type": "integer", "required": True},
    "city": {"type": "string", "required": False},
}


def case(case_id="c", expected=None):
    return structured.StructuredCase(case_id, "extract", SCHEMA, expected or {})


def test_parse_output_variants():
    assert structured.parse_output('{"a": 1}') == {"a": 1}
    assert structured.parse_output('```json\n{"a": 1}\n```') == {"a": 1}
    assert structured.parse_output("not json") is None
    assert structured.parse_output("[1,2]") is None  # array is not an object
    assert structured.parse_output("") is None


def test_conformance():
    c = case()
    assert structured.is_conformant(c, {"name": "Олена", "age": 34}) is True
    assert structured.is_conformant(c, {"name": "Олена", "age": 34, "city": "Львів"}) is True
    assert structured.is_conformant(c, {"name": "Олена"}) is False  # missing required age
    assert structured.is_conformant(c, {"name": "Олена", "age": "abc"}) is False  # bad type
    assert structured.is_conformant(c, None) is False


def test_field_accuracy():
    assert (
        structured.field_accuracy({"name": "Олена", "age": 34}, {"name": "олена", "age": 34}) == 1.0
    )
    assert (
        structured.field_accuracy({"name": "Олена", "age": 34}, {"name": "Олена", "age": 99}) == 0.5
    )
    assert structured.field_accuracy({"x": 1}, None) == 0.0
    assert structured.field_accuracy({}, {"a": 1}) == 1.0  # nothing expected -> full


def test_score_structured_non_conformant_zeroes_accuracy():
    cases = [case("a", {"name": "Олена", "age": 34}), case("b", {"name": "Іван", "age": 40})]
    outputs = ['{"name":"Олена","age":34}', '{"name":"Іван"}']  # b missing age -> non-conformant
    score = structured.score_structured(cases, outputs)
    assert score.conformance_rate == 0.5
    assert score.field_accuracy == 0.5  # a=1.0, b=0.0
    assert score.cases[1].score == 0.0


def test_score_structured_length_mismatch():
    with pytest.raises(ValueError, match="aligned"):
        structured.score_structured([case()], [])


def test_run_structured_persists(tmp_path):
    cases = [case("a", {"name": "Олена", "age": 34})]
    run = bench_st.run_structured(
        cases,
        model="m",
        backend="ollama",
        complete=lambda _: '{"name":"Олена","age":34}',
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    assert run.result.tier == TIER_STRUCTURED
    assert run.result.objective_score == 1.0
    assert run.score.conformance_rate == 1.0
    assert run.paths is not None and "structured" in run.paths["manifest"]


def test_load_committed_structured_cases():
    cases = bench_st.load_cases_file("samples/structured_cases_uk.json")
    assert len(cases) == 3 and all(c.schema for c in cases)


# --- nested-object / array-item validation + per-field tolerance (M5.4 residual) ----------

NESTED_SCHEMA = {
    "name": {"type": "string", "required": True},
    "address": {
        "type": "object",
        "required": True,
        "fields": {
            "city": {"type": "string", "required": True},
            "zip": {"type": "integer", "required": True, "tolerance": 100},
        },
    },
}

ARRAY_SCHEMA = {
    "items": {
        "type": "array",
        "required": True,
        "items": {
            "type": "object",
            "fields": {
                "sku": {"type": "string", "required": True},
                "qty": {"type": "integer", "required": True},
            },
        },
    },
}


def test_nested_object_conformance():
    c = structured.StructuredCase("n", "extract", NESTED_SCHEMA, {})
    assert structured.is_conformant(c, {"name": "О", "address": {"city": "Львів", "zip": 79000}})
    assert not structured.is_conformant(c, {"name": "О", "address": {"city": "Львів"}})  # no zip
    assert not structured.is_conformant(c, {"name": "О", "address": {"city": "Львів", "zip": "x"}})
    assert not structured.is_conformant(c, {"name": "О", "address": "Львів"})  # not an object


def test_array_of_objects_conformance():
    c = structured.StructuredCase("a", "extract", ARRAY_SCHEMA, {})
    assert structured.is_conformant(c, {"items": [{"sku": "a", "qty": 1}, {"sku": "b", "qty": 2}]})
    assert not structured.is_conformant(c, {"items": [{"sku": "a"}]})  # item missing qty
    assert not structured.is_conformant(c, {"items": [{"sku": "a", "qty": "two"}]})  # bad item type


def test_field_accuracy_recurses_into_nested_leaves():
    exp = {"name": "Олена", "address": {"city": "Львів", "zip": 79000}}
    # exact (3 leaves; city casefold, zip exact)
    assert structured.field_accuracy(exp, exp, NESTED_SCHEMA) == 1.0
    # one nested leaf wrong -> 2/3
    wrong = {"name": "Олена", "address": {"city": "Київ", "zip": 79000}}
    assert round(structured.field_accuracy(exp, wrong, NESTED_SCHEMA), 4) == round(2 / 3, 4)


def test_field_accuracy_numeric_tolerance():
    exp = {"name": "Олена", "address": {"city": "Львів", "zip": 79000}}
    near = {"name": "Олена", "address": {"city": "Львів", "zip": 79050}}  # within zip tolerance 100
    assert structured.field_accuracy(exp, near, NESTED_SCHEMA) == 1.0
    far = {"name": "Олена", "address": {"city": "Львів", "zip": 79500}}  # outside tolerance
    assert round(structured.field_accuracy(exp, far, NESTED_SCHEMA), 4) == round(2 / 3, 4)
    # without a schema the same near-miss is exact-compared -> mismatch
    assert round(structured.field_accuracy(exp, near), 4) == round(2 / 3, 4)


def test_field_accuracy_array_elementwise():
    exp = {"items": [{"sku": "a", "qty": 1}, {"sku": "b", "qty": 2}]}
    one = {"items": [{"sku": "A", "qty": 1}]}  # 2 of 4 leaves (sku casefold), 2nd element missing
    assert structured.field_accuracy(exp, one, ARRAY_SCHEMA) == 0.5


def test_field_accuracy_int_float_coercion():
    assert structured.field_accuracy({"n": 3}, {"n": 3.0}) == 1.0


def test_score_case_threads_schema_tolerance():
    schema = {"score": {"type": "number", "required": True, "tolerance": 0.5}}
    c = structured.StructuredCase("t", "rate it", schema, {"score": 9.0})
    s = structured.score_case(c, '{"score": 9.3}')  # conformant + within tolerance
    assert s.conformant == 1.0 and s.score == 1.0
    s2 = structured.score_case(c, '{"score": 9.9}')  # conformant but outside tolerance
    assert s2.conformant == 1.0 and s2.score == 0.0
