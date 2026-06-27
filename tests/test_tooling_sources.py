"""tooling benchmark tooling residuals -- per-argument tolerance, native FC caller, MCP specs, BFCL adapter."""

from llb.bench import mcp_server
from llb.bench import tooling as bench_tooling
from llb.prep import tooling_sources
from llb.scoring import tooling

CATALOG = {
    "get_weather": {
        "name": "get_weather",
        "description": "weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    "search_documents": {
        "name": "search_documents",
        "description": "search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


# --- per-argument tolerance ----------------------------------------------------------------


def test_arguments_match_contains():
    spec = {"query": {"mode": "contains"}}
    assert tooling.arguments_match({"query": "енерг"}, {"query": "відновлювана енергія"}, spec)
    assert not tooling.arguments_match({"query": "енерг"}, {"query": "погода"}, spec)


def test_arguments_match_numeric_tolerance():
    spec = {"amount": {"mode": "numeric", "tol": 0.5}}
    assert tooling.arguments_match({"amount": 100}, {"amount": 99.99}, spec)
    assert not tooling.arguments_match({"amount": 100}, {"amount": 90}, spec)


def test_arguments_match_oneof():
    spec = {"city": {"mode": "oneof", "values": ["Дніпро", "Dnipro"]}}
    assert tooling.arguments_match({"city": "Дніпро"}, {"city": "dnipro"}, spec)
    assert not tooling.arguments_match({"city": "Дніпро"}, {"city": "Київ"}, spec)


def test_arguments_match_fuzzy():
    spec = {"expression": {"mode": "fuzzy", "threshold": 0.6}}
    assert tooling.arguments_match({"expression": "100 / 4"}, {"expression": "100/4"}, spec)


def test_arguments_match_default_exact_unchanged():
    assert tooling.arguments_match({"city": "Київ"}, {"city": "київ"})  # casefold exact
    assert not tooling.arguments_match({"city": "Київ"}, {"city": "Львів"})


def test_committed_cases_exercise_tolerance():
    _catalog, cases = bench_tooling.load_catalog_file("samples/tooling_cases_uk.json")
    by_id = {c.id: c for c in cases}
    assert by_id["tc-009"].arg_match["query"]["mode"] == "contains"
    assert by_id["tc-011"].arg_match["city"]["mode"] == "oneof"


# --- native OpenAI tools= caller -----------------------------------------------------------


class _FakeFn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name, arguments):
        self.function = _FakeFn(name, arguments)


class _FakeMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResp:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


class _FakeClient:
    def __init__(self, message):
        self._message = message
        self.last_kwargs = None

        class _Completions:
            def create(_self, **kwargs):
                self.last_kwargs = kwargs
                return _FakeResp(self._message)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_openai_tools_schema():
    specs = bench_tooling.openai_tools(CATALOG)
    assert specs[0]["type"] == "function"
    assert specs[0]["function"]["name"] == "get_weather"


def test_native_tool_caller_parses_native_response():
    client = _FakeClient(_FakeMessage([_FakeToolCall("get_weather", '{"city": "Київ"}')]))
    caller = bench_tooling.native_tool_caller(client, "m")
    call = caller("Яка погода у Києві?", CATALOG)
    assert call is not None and call.name == "get_weather"
    assert call.arguments == {"city": "Київ"}
    assert client.last_kwargs["tools"][0]["function"]["name"] == "get_weather"


def test_native_tool_caller_no_call():
    caller = bench_tooling.native_tool_caller(_FakeClient(_FakeMessage(None)), "m")
    assert caller("just chatting", CATALOG) is None


def test_run_tooling_with_native_caller():
    client = _FakeClient(_FakeMessage([_FakeToolCall("get_weather", '{"city": "Київ"}')]))
    caller = bench_tooling.native_tool_caller(client, "m")
    cases = [
        tooling.ToolingCase(
            id="c1",
            instruction="погода?",
            expected_tool="get_weather",
            expected_arguments={"city": "Київ"},
        )
    ]
    run = bench_tooling.run_tooling(
        CATALOG,
        cases,
        model="m",
        backend="vllm",
        caller=caller,
        capability=bench_tooling.TOOL_PROTOCOL_NATIVE,
        persist=False,
    )
    assert run.score.call_accuracy == 1.0


# --- MCP transport specs -------------------------------------------------------------------


def test_mcp_tool_specs_from_catalog():
    specs = mcp_server.mcp_tool_specs(CATALOG)
    names = {s["name"] for s in specs}
    assert names == {"get_weather", "search_documents"}
    assert specs[0]["inputSchema"]["type"] == "object"


def test_mcp_specs_match_catalog_source():
    specs = mcp_server.mcp_tool_specs(CATALOG)
    for spec, tool in zip(specs, CATALOG.values()):
        assert spec["inputSchema"] == tool["parameters"]  # same single source of truth


# --- BFCL adapter --------------------------------------------------------------------------

BFCL_ENTRY = {
    "id": "simple_0",
    "question": [[{"role": "user", "content": "What is the weather in Kyiv?"}]],
    "function": [
        {
            "name": "get_weather",
            "description": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ],
}
BFCL_ANSWER = {"id": "simple_0", "ground_truth": [{"get_weather": {"city": ["Kyiv", "Kiev"]}}]}


def test_from_bfcl_builds_bundle_with_oneof_tolerance():
    bundle = tooling_sources.from_bfcl([BFCL_ENTRY], [BFCL_ANSWER])
    assert bundle["tools"][0]["name"] == "get_weather"
    case = bundle["cases"][0]
    assert case["expected_tool"] == "get_weather"
    assert case["expected_arguments"] == {"city": "Kyiv"}
    assert case["arg_match"]["city"]["mode"] == "oneof"
    # round-trips into ToolingCases and scores the acceptable variant
    cases = [tooling.ToolingCase.from_record(c) for c in bundle["cases"]]
    call = tooling.ToolCall(name="get_weather", arguments={"city": "Kiev"}, well_formed=True)
    catalog = {t["name"]: t for t in bundle["tools"]}
    assert tooling.score_case(cases[0], call, catalog).correct == 1.0


def test_from_bfcl_translate_injected():
    bundle = tooling_sources.from_bfcl([BFCL_ENTRY], translate=lambda _t: "Яка погода у Києві?")
    assert bundle["cases"][0]["instruction"] == "Яка погода у Києві?"
    assert bundle["cases"][0]["expected_tool"] is None  # no answers -> control


def test_from_bfcl_question_string_shape():
    entry = {"id": "x", "question": "plain string question", "function": []}
    bundle = tooling_sources.from_bfcl([entry])
    assert bundle["cases"][0]["instruction"] == "plain string question"
