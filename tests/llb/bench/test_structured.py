"""structured-output -- Pydantic conformance + field accuracy."""

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


def test_run_structured_reports_meter_throughput(tmp_path):
    import json
    from pathlib import Path

    from llb.bench.common_backend import ThroughputMeter

    meter = ThroughputMeter()
    meter.completion_tokens, meter.generation_s, meter.calls = 100, 4.0, 4  # 25 tok/s
    run = bench_st.run_structured(
        [case("a", {"name": "Олена", "age": 34})],
        model="m",
        backend="ollama",
        complete=lambda _: '{"name":"Олена","age":34}',
        data_dir=tmp_path,
        mirror=lambda *_: None,
        meter=meter,
    )
    assert run.result.tokens_per_s == 25.0  # real throughput flows onto the board row
    manifest = json.loads(Path(run.paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["metrics"]["tokens_per_s"] == 25.0


def test_run_structured_invalid_verification_ref_does_not_call_model(tmp_path):
    bad_ref = tmp_path / "verify_sample.csv"
    bad_ref.write_text("item_id,stratum,decision\nok,s,\n", encoding="utf-8")
    calls = 0

    def complete(_: str) -> str:
        nonlocal calls
        calls += 1
        return '{"name":"Олена","age":34}'

    with pytest.raises(ValueError) as excinfo:
        bench_st.run_structured(
            [case("a", {"name": "Олена", "age": 34})],
            model="m",
            backend="ollama",
            complete=complete,
            data_dir=tmp_path,
            data_verified=True,
            verification_ref=str(bad_ref),
            mirror=lambda *_: None,
        )

    assert calls == 0
    assert "verification reference cannot be used with --data-verified" in str(excinfo.value)
    assert "undecided: 1" in str(excinfo.value)


def test_load_committed_structured_cases():
    cases = bench_st.load_cases_file("samples/benchmarks/structured_cases_uk.json")
    assert len(cases) == 6 and all(c.schema for c in cases)
    # nested / array / unordered cases are now committed (exercise the recursive matcher)
    by_id = {c.id: c for c in cases}
    assert by_id["st-004"].schema["address"]["type"] == "object"
    assert by_id["st-005"].schema["items"]["items"]["type"] == "object"
    assert by_id["st-006"].schema["tags"]["unordered"] is True


# --- nested-object / array-item validation + per-field tolerance (category expansion residual) ----------

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
