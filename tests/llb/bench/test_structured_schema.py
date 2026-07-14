"""Tests for structured schema."""

from llb.scoring import structured
from test_structured import ARRAY_SCHEMA, NESTED_SCHEMA


def test_field_accuracy_unordered_array_set_match():
    schema = {"tags": {"type": "array", "unordered": True, "items": {"type": "string"}}}
    exp = {"tags": ["alpha", "beta", "gamma"]}
    shuffled = {"tags": ["gamma", "alpha", "beta"]}  # same set, different order
    assert structured.field_accuracy(exp, shuffled, schema) == 1.0
    partial = {"tags": ["beta", "alpha"]}  # 2 of 3 present
    assert round(structured.field_accuracy(exp, partial, schema), 4) == round(2 / 3, 4)


def test_field_accuracy_unordered_array_of_objects():
    schema = {
        "items": {
            "type": "array",
            "unordered": True,
            "items": {
                "type": "object",
                "fields": {"sku": {"type": "string"}, "qty": {"type": "integer"}},
            },
        }
    }
    exp = {"items": [{"sku": "a", "qty": 1}, {"sku": "b", "qty": 2}]}
    reordered = {"items": [{"sku": "b", "qty": 2}, {"sku": "a", "qty": 1}]}
    assert structured.field_accuracy(exp, reordered, schema) == 1.0


def test_leaf_match_fuzzy_and_relative_tolerance():
    fuzzy = {"s": {"type": "string", "string_match": "fuzzy", "threshold": 0.6}}
    assert (
        structured.field_accuracy(
            {"s": "відновлювана енергетика"}, {"s": "відновлювана енергія"}, fuzzy
        )
        == 1.0
    )
    contains = {"s": {"type": "string", "string_match": "contains"}}
    assert structured.field_accuracy({"s": "енерг"}, {"s": "сонячна енергетика"}, contains) == 1.0
    rel = {"n": {"type": "number", "rel_tolerance": 0.01}}
    assert structured.field_accuracy({"n": 51200}, {"n": 51199}, rel) == 1.0
    assert structured.field_accuracy({"n": 51200}, {"n": 50000}, rel) == 0.0


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
