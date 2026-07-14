"""tooling benchmark tooling / function-calling -- parse, validate, score, runner."""

import types

import pytest

from llb.bench import tooling as bench_tool
from llb.scoring import tooling
from llb.scoring import tool_calls
from llb.scoring.aggregate import TIER_TOOLING

WEATHER = {
    "name": "get_weather",
    "description": "weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "date": {"type": "string"}},
        "required": ["city"],
    },
}
SEARCH = {
    "name": "search_documents",
    "description": "search",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
        "required": ["query"],
    },
}
CATALOG = {"get_weather": WEATHER, "search_documents": SEARCH}


# --- parse layer --------------------------------------------------------------------------


def test_parse_native_tool_calls():
    msg = types.SimpleNamespace(
        tool_calls=[
            types.SimpleNamespace(
                function=types.SimpleNamespace(name="get_weather", arguments='{"city": "Київ"}')
            )
        ]
    )
    call = tool_calls.parse_tool_call(msg)
    assert call is not None
    assert call.name == "get_weather" and call.arguments == {"city": "Київ"} and call.well_formed


def test_parse_text_json_call_and_aliases():
    call = tool_calls.parse_tool_call('{"tool": "search_documents", "args": {"query": "вода"}}')
    assert call is not None and call.name == "search_documents"
    assert call.arguments == {"query": "вода"}


def test_parse_null_and_plain_text_are_no_call():
    assert tool_calls.parse_tool_call('{"name": null}') is None
    assert tool_calls.parse_tool_call("Привіт, як справи?") is None
    assert tool_calls.parse_tool_call("") is None
    assert tool_calls.parse_tool_call(None) is None


def test_parse_malformed_arguments_not_well_formed():
    call = tool_calls.parse_tool_call('{"name": "get_weather", "arguments": "not-json"}')
    assert call is not None and call.name == "get_weather" and call.well_formed is False


# --- argument validation ------------------------------------------------------------------


def test_validate_arguments_ok():
    assert tool_calls.validate_arguments(WEATHER, {"city": "Київ"}) == []
    assert tool_calls.validate_arguments(WEATHER, {"city": "Київ", "date": "завтра"}) == []


def test_validate_arguments_missing_required():
    errs = tool_calls.validate_arguments(WEATHER, {"date": "завтра"})
    assert any("missing required" in e for e in errs)


def test_validate_arguments_unknown_and_type():
    assert any(
        "unknown argument" in e
        for e in tool_calls.validate_arguments(WEATHER, {"city": "K", "x": 1})
    )
    assert any("expected string" in e for e in tool_calls.validate_arguments(WEATHER, {"city": 5}))
    # bool must not satisfy integer
    assert any(
        "expected integer" in e
        for e in tool_calls.validate_arguments(SEARCH, {"query": "q", "top_k": True})
    )


def test_arguments_match_casefold_and_keys():
    assert tool_calls.arguments_match({"city": "Київ"}, {"city": "київ"}) is True
    assert tool_calls.arguments_match({"city": "Київ"}, {"city": "Київ", "date": "x"}) is False
    assert tool_calls.arguments_match({"amount": 10}, {"amount": 10}) is True


# --- per-case + aggregate scoring ---------------------------------------------------------


def case(case_id, instruction, tool, args=None):
    return tooling.ToolingCase(case_id, instruction, tool, args or {})


def test_score_case_perfect_call():
    c = case("c", "погода", "get_weather", {"city": "Київ"})
    call = tool_calls.ToolCall("get_weather", {"city": "київ"}, well_formed=True)
    s = tooling.score_case(c, call, CATALOG)
    assert s.correct == 1.0 and s.tool_selected == 1.0 and s.arguments_exact == 1.0


def test_score_case_wrong_tool_and_hallucination():
    c = case("c", "погода", "get_weather", {"city": "Київ"})
    wrong = tooling.score_case(
        c, tool_calls.ToolCall("search_documents", {"query": "x"}, True), CATALOG
    )
    assert wrong.tool_selected == 0.0 and wrong.correct == 0.0 and wrong.no_hallucinated_tool == 1.0
    halluc = tooling.score_case(c, tool_calls.ToolCall("translate", {}, True), CATALOG)
    assert halluc.no_hallucinated_tool == 0.0


def test_score_case_no_tool_expected():
    c = case("c", "привіт", None)
    assert tooling.score_case(c, None, CATALOG).correct == 1.0
    called = tooling.score_case(c, tool_calls.ToolCall("get_weather", {"city": "K"}, True), CATALOG)
    assert called.correct == 0.0 and called.no_hallucinated_tool == 1.0


def test_score_case_wrong_arguments():
    c = case("c", "погода", "get_weather", {"city": "Київ"})
    extra = tooling.score_case(
        c, tool_calls.ToolCall("get_weather", {"city": "Київ", "date": "завтра"}, True), CATALOG
    )
    assert extra.tool_selected == 1.0 and extra.schema_valid == 1.0 and extra.arguments_exact == 0.0


def test_score_tooling_aggregate():
    cases = [
        case("a", "погода", "get_weather", {"city": "Київ"}),
        case("b", "пошук", "search_documents", {"query": "вода"}),
        case("c", "привіт", None),
    ]
    calls = [
        tool_calls.ToolCall("get_weather", {"city": "Київ"}, True),
        tool_calls.ToolCall("search_documents", {"query": "інше"}, True),  # wrong arg value
        None,
    ]
    score = tooling.score_tooling(cases, calls, CATALOG)
    assert score.tool_selection_accuracy == 1.0  # all selected the right tool / correctly no-call
    assert round(score.call_accuracy, 4) == round(2 / 3, 4)  # b's args wrong
    assert round(score.argument_exactness, 4) == 0.5  # 1 of 2 tool-expecting cases exact


def test_score_tooling_length_mismatch():
    with pytest.raises(ValueError, match="aligned"):
        tooling.score_tooling([case("a", "x", None)], [], CATALOG)


# --- runner -------------------------------------------------------------------------------


def scripted(outputs):
    it = iter(outputs)
    return lambda _prompt: next(it)


def test_run_tooling_perfect_model_persists(tmp_path):
    catalog, cases = bench_tool.load_catalog_file("samples/benchmarks/tooling_cases_uk.json")
    outputs = [
        '{"name":"get_weather","arguments":{"city":"Київ"}}',
        '{"name":"get_weather","arguments":{"city":"Львів","date":"завтра"}}',
        '{"name":"convert_currency","arguments":{"amount":2350,"from_currency":"UAH","to_currency":"USD"}}',
        '{"name":"search_documents","arguments":{"query":"відновлювана енергія"}}',
        '{"name":"calculator","arguments":{"expression":"15 * 24 + 100"}}',
        '{"name":"create_reminder","arguments":{"title":"Дзвінок з клієнтом","datetime":"2026-07-01 09:00"}}',
        '{"name": null}',
        '{"name": null}',
        # tc-009..tc-012 exercise per-argument tolerance (contains / numeric / oneof / fuzzy)
        '{"name":"search_documents","arguments":{"query":"відновлювана енергетика України"}}',
        '{"name":"convert_currency","arguments":{"amount":99.99,"from_currency":"USD","to_currency":"UAH"}}',
        '{"name":"get_weather","arguments":{"city":"Dnipro"}}',
        '{"name":"calculator","arguments":{"expression":"100/4"}}',
    ]
    run = bench_tool.run_tooling(
        catalog,
        cases,
        model="m",
        backend="ollama",
        complete=scripted(outputs),
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    assert run.result.tier == TIER_TOOLING
    assert run.result.objective_score == 1.0
    assert run.score.call_accuracy == 1.0
    assert run.accuracy_ci is not None
    assert run.paths is not None and "tooling" in run.paths["manifest"]


def test_run_tooling_reports_meter_throughput(tmp_path):
    import json
    from pathlib import Path

    from llb.bench.common_backend import ThroughputMeter

    meter = ThroughputMeter()
    meter.completion_tokens, meter.generation_s, meter.calls = 100, 4.0, 4  # 25 tok/s
    run = bench_tool.run_tooling(
        CATALOG,
        [case("c", "погода", "get_weather", {"city": "Київ"})],
        model="m",
        backend="ollama",
        complete=lambda _: '{"name":"get_weather","arguments":{"city":"Київ"}}',
        data_dir=tmp_path,
        mirror=lambda *_: None,
        meter=meter,
    )
    assert run.result.tokens_per_s == 25.0  # real throughput flows onto the board row
    manifest = json.loads(Path(run.paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["metrics"]["tokens_per_s"] == 25.0


def test_run_tooling_text_only_model_scores_low():
    catalog, cases = bench_tool.load_catalog_file("samples/benchmarks/tooling_cases_uk.json")
    # always answers in prose, never calls -> only the no-tool cases score
    run = bench_tool.run_tooling(
        catalog, cases, model="m", backend="ollama", complete=lambda _: "Звичайно!", persist=False
    )
    n_no_tool = sum(1 for c in cases if c.expected_tool is None)
    assert round(run.score.call_accuracy, 4) == round(n_no_tool / len(cases), 4)


def test_load_catalog_file_shape():
    catalog, cases = bench_tool.load_catalog_file("samples/benchmarks/tooling_cases_uk.json")
    assert "get_weather" in catalog and len(cases) == 12
    assert any(c.expected_tool is None for c in cases)  # no-tool cases present
    assert any(c.arg_match for c in cases)  # per-argument tolerance cases present
