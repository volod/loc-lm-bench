"""tooling benchmark tooling / function-calling scoring -- objective, CALL-ONLY (pure).

Scores a candidate's emitted tool call against an expected (tool name + argument JSON), without
EXECUTING the tool (execution is the agentic benchmark). Two layers, both pure:

  * the PARSE layer (`parse_tool_call`) normalizes a backend response into a `ToolCall` whether
    the backend emits a NATIVE OpenAI `tool_calls` object or a text-only backend emits a JSON
    tool call in `content` (so tool-capable and text-only backends share one scorer; the runner
    records which capability a backend actually has and never cross-ranks them);
  * the SCORE layer (`score_tooling`) reports the four objective metrics from the plan:
    tool-selection accuracy, argument-exactness (schema-valid + value match), no-hallucinated-tool
    rate, and well-formed-call rate. The headline (`call_accuracy`) requires the right tool AND
    exact arguments.

Argument validation is a lightweight structural check (required present, declared types, no
unknown properties) -- no new `jsonschema` dependency (the heavier Pydantic conformance check is
the structured-output category).
"""

import json
from dataclasses import dataclass, field
from typing import Any

from llb.contracts import ToolDef
from llb.prep.frontier import parse_json_block

# JSON-schema primitive type -> accepted Python types.
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}
# Accepted aliases the model/dataset may use for the call envelope.
_NAME_KEYS = ("name", "tool", "function", "tool_name")
_ARG_KEYS = ("arguments", "args", "parameters", "params")


@dataclass(frozen=True)
class ToolCall:
    """A normalized tool call extracted from a backend response."""

    name: str
    arguments: dict[str, Any]
    well_formed: bool  # parsed cleanly into name + a JSON-object argument map
    raw: str = ""


def _load_args(raw: Any) -> tuple[dict[str, Any], bool]:
    """Coerce an arguments payload (a dict or a JSON string) into (dict, well_formed)."""
    if isinstance(raw, dict):
        return raw, True
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}, False
        return (parsed, True) if isinstance(parsed, dict) else ({}, False)
    return {}, False


def _from_dict(obj: dict[str, Any], raw: str = "") -> ToolCall | None:
    name = next((str(obj[k]) for k in _NAME_KEYS if k in obj and obj[k]), "")
    if not name:
        return None
    arg_value: Any = next((obj[k] for k in _ARG_KEYS if k in obj), {})
    arguments, well = _load_args(arg_value)
    return ToolCall(name=name, arguments=arguments, well_formed=well, raw=raw or json.dumps(obj))


def _from_native_tool_call(raw: Any) -> ToolCall | None:
    tool_calls = getattr(raw, "tool_calls", None)
    if not tool_calls:
        return None
    fn = tool_calls[0].function
    name = str(getattr(fn, "name", "") or "")
    raw_args = str(getattr(fn, "arguments", ""))
    arguments, well = _load_args(raw_args or "")
    return ToolCall(
        name=name,
        arguments=arguments,
        well_formed=well and bool(name),
        raw=raw_args,
    )


def _from_text_tool_call(raw: str) -> ToolCall | None:
    text = raw.strip()
    if not text:
        return None
    try:
        obj = parse_json_block(text)
    except json.JSONDecodeError:
        return None  # a plain text answer is not a tool-call attempt
    return _from_dict(obj, raw=text) if isinstance(obj, dict) else None


def parse_tool_call(raw: Any) -> ToolCall | None:
    """Normalize a backend response into a `ToolCall`, or None when no tool call was attempted.

    Handles a native OpenAI message (an object exposing `.tool_calls`), a pre-parsed dict, and a
    text response carrying a JSON tool call (the text-only-backend fallback). A non-JSON text
    answer means the model did NOT attempt a tool call -> None.
    """
    if raw is None:
        return None
    native = _from_native_tool_call(raw)
    if native is not None:
        return native
    if isinstance(raw, dict):
        return _from_dict(raw)
    if isinstance(raw, str):
        return _from_text_tool_call(raw)
    return None


# --- argument schema validation (lightweight; no jsonschema dep) ---------------------------


def _required_argument_errors(required: list[str], arguments: dict[str, Any]) -> list[str]:
    return [f"missing required argument: {name}" for name in required if name not in arguments]


def _type_error(name: str, value: Any, declared: str | None) -> str | None:
    accepted = _TYPE_MAP.get(declared) if declared else None
    if accepted and (not isinstance(value, accepted) or _bad_bool(declared or "", value)):
        return f"argument {name}: expected {declared}, got {type(value).__name__}"
    return None


def _argument_schema_error(name: str, value: Any, properties: dict[str, Any]) -> str | None:
    if name not in properties:
        return f"unknown argument: {name}"
    return _type_error(name, value, properties[name].get("type"))


def validate_arguments(tool: ToolDef, arguments: dict[str, Any]) -> list[str]:
    """Structural check of `arguments` against a tool's `parameters` schema: required present,
    declared types, no unknown properties. Returns a list of human-readable errors ([] == valid)."""
    schema = tool.get("parameters", {}) or {}
    properties: dict[str, Any] = schema.get("properties", {}) or {}
    required: list[str] = list(schema.get("required", []) or [])
    errors = _required_argument_errors(required, arguments)
    for name, value in arguments.items():
        error = _argument_schema_error(name, value, properties)
        if error is not None:
            errors.append(error)
    return errors


def _bad_bool(declared: str, value: Any) -> bool:
    return declared in ("integer", "number") and isinstance(value, bool)


def _normalize_value(value: Any) -> Any:
    return value.casefold().strip() if isinstance(value, str) else value


# Per-argument match modes (a case may relax exact match for free-text / numeric args; BFCL's
# possible-answer sets, which list several acceptable values per argument, map onto `oneof`).
MATCH_EXACT = "exact"
MATCH_CONTAINS = "contains"  # expected (normalized) is a substring of provided (free-text query)
MATCH_FUZZY = "fuzzy"  # difflib ratio >= threshold (default 0.8); stdlib, no fuzzy-match dep
MATCH_NUMERIC = "numeric"  # |provided - expected| <= tol (default 1e-9)
MATCH_ONEOF = "oneof"  # provided matches any value in the spec's `values` list
DEFAULT_FUZZY_THRESHOLD = 0.8


def _match_oneof(provided: Any, spec: dict[str, Any] | None) -> bool:
    values = [_normalize_value(v) for v in (spec or {}).get("values", [])]
    return _normalize_value(provided) in values


def _match_numeric(expected: Any, provided: Any, spec: dict[str, Any] | None) -> bool:
    try:
        return abs(float(provided) - float(expected)) <= float((spec or {}).get("tol", 1e-9))
    except (TypeError, ValueError):
        return False


def _match_contains(expected: Any, provided: Any) -> bool:
    exp = _normalize_value(expected)
    prov = _normalize_value(provided)
    return isinstance(exp, str) and isinstance(prov, str) and exp in prov


def _match_fuzzy(expected: Any, provided: Any, spec: dict[str, Any] | None) -> bool:
    exp = _normalize_value(expected)
    prov = _normalize_value(provided)
    if not (isinstance(exp, str) and isinstance(prov, str)):
        return bool(exp == prov)
    import difflib

    threshold = float((spec or {}).get("threshold", DEFAULT_FUZZY_THRESHOLD))
    return difflib.SequenceMatcher(None, exp, prov).ratio() >= threshold


def _match_value(expected: Any, provided: Any, spec: dict[str, Any] | None) -> bool:
    """Match one argument value under its per-argument tolerance spec (default: exact)."""
    mode = (spec or {}).get("mode", MATCH_EXACT)
    if mode == MATCH_ONEOF:
        return _match_oneof(provided, spec)
    if mode == MATCH_NUMERIC:
        return _match_numeric(expected, provided, spec)
    if mode == MATCH_CONTAINS:
        return _match_contains(expected, provided)
    if mode == MATCH_FUZZY:
        return _match_fuzzy(expected, provided, spec)
    return bool(_normalize_value(expected) == _normalize_value(provided))  # MATCH_EXACT


def arguments_match(
    expected: dict[str, Any],
    provided: dict[str, Any],
    arg_match: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """Argument match over the same key set, each value compared under its per-argument tolerance.

    Without `arg_match` every value is exact (strings casefold/strip-insensitive, the legacy
    behavior). `arg_match[name]` relaxes a single argument to contains / fuzzy / numeric / oneof.
    """
    arg_match = arg_match or {}
    if set(expected) != set(provided):
        return False
    return all(_match_value(expected[k], provided[k], arg_match.get(k)) for k in expected)


# --- per-case + aggregate scoring ----------------------------------------------------------


@dataclass(frozen=True)
class ToolingCase:
    """One function-calling case: a UA instruction -> expected tool + argument JSON."""

    id: str
    instruction: str
    expected_tool: str | None  # None == the model should NOT call any tool
    expected_arguments: dict[str, Any] = field(default_factory=dict)
    arg_match: dict[str, dict[str, Any]] = field(default_factory=dict)  # per-arg tolerance specs

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ToolingCase":
        return cls(
            id=str(record["id"]),
            instruction=str(record["instruction"]),
            expected_tool=record.get("expected_tool"),
            expected_arguments=dict(record.get("expected_arguments", {}) or {}),
            arg_match=dict(record.get("arg_match", {}) or {}),
        )


@dataclass(frozen=True)
class ToolingCaseScore:
    item_id: str
    expected_tool: str | None
    called_tool: str | None
    attempted: bool
    tool_selected: float
    schema_valid: float
    arguments_exact: float
    no_hallucinated_tool: float
    well_formed: float
    correct: float  # headline: right tool AND exact args (or correctly no-call)


@dataclass(frozen=True)
class ToolingScore:
    n_cases: int
    call_accuracy: float  # headline (mean per-case correct)
    tool_selection_accuracy: float
    argument_exactness: float  # over cases expecting a tool
    no_hallucinated_tool_rate: float
    well_formed_rate: float  # over attempted calls
    case_correct: list[float]
    cases: list[ToolingCaseScore]


@dataclass(frozen=True)
class _CallContext:
    attempted: bool
    in_catalog: bool
    called_tool: str | None
    no_hallucinated_tool: float
    well_formed: float


def _call_context(call: ToolCall | None, catalog: dict[str, ToolDef]) -> _CallContext:
    attempted = call is not None
    in_catalog = call is not None and call.name in catalog
    return _CallContext(
        attempted=attempted,
        in_catalog=in_catalog,
        called_tool=call.name if call else None,
        no_hallucinated_tool=1.0 if (not attempted or in_catalog) else 0.0,
        well_formed=(1.0 if (call and call.well_formed) else 0.0) if attempted else 0.0,
    )


def _no_tool_expected_score(case: ToolingCase, ctx: _CallContext) -> ToolingCaseScore:
    correct = 0.0 if ctx.attempted else 1.0
    return ToolingCaseScore(
        item_id=case.id,
        expected_tool=None,
        called_tool=ctx.called_tool,
        attempted=ctx.attempted,
        tool_selected=correct,
        schema_valid=0.0,
        arguments_exact=0.0,
        no_hallucinated_tool=ctx.no_hallucinated_tool,
        well_formed=ctx.well_formed,
        correct=correct,
    )


def _tool_selected(case: ToolingCase, call: ToolCall | None) -> float:
    return 1.0 if (call and call.name == case.expected_tool) else 0.0


def _schema_valid(
    call: ToolCall | None,
    ctx: _CallContext,
    catalog: dict[str, ToolDef],
    tool_selected: float,
) -> float:
    if not (tool_selected and call is not None and ctx.in_catalog):
        return 0.0
    errors = validate_arguments(catalog[call.name], call.arguments)
    return 1.0 if (call.well_formed and not errors) else 0.0


def _arguments_exact(case: ToolingCase, call: ToolCall | None, schema_valid: float) -> float:
    if not (schema_valid and call is not None):
        return 0.0
    return 1.0 if arguments_match(case.expected_arguments, call.arguments, case.arg_match) else 0.0


def _tool_expected_score(
    case: ToolingCase,
    call: ToolCall | None,
    ctx: _CallContext,
    catalog: dict[str, ToolDef],
) -> ToolingCaseScore:
    tool_selected = _tool_selected(case, call)
    schema_valid = _schema_valid(call, ctx, catalog, tool_selected)
    arguments_exact = _arguments_exact(case, call, schema_valid)
    return ToolingCaseScore(
        item_id=case.id,
        expected_tool=case.expected_tool,
        called_tool=ctx.called_tool,
        attempted=ctx.attempted,
        tool_selected=tool_selected,
        schema_valid=schema_valid,
        arguments_exact=arguments_exact,
        no_hallucinated_tool=ctx.no_hallucinated_tool,
        well_formed=ctx.well_formed,
        correct=arguments_exact,  # full correctness requires the right tool AND exact args
    )


def score_case(
    case: ToolingCase, call: ToolCall | None, catalog: dict[str, ToolDef]
) -> ToolingCaseScore:
    """Score one tooling case against the candidate's (parsed) tool call."""
    ctx = _call_context(call, catalog)
    if case.expected_tool is None:
        return _no_tool_expected_score(case, ctx)
    return _tool_expected_score(case, call, ctx, catalog)


def _rate(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _score_cases(
    cases: list[ToolingCase], calls: list[ToolCall | None], catalog: dict[str, ToolDef]
) -> list[ToolingCaseScore]:
    return [score_case(case, call, catalog) for case, call in zip(cases, calls)]


def _expecting_tool(scored: list[ToolingCaseScore]) -> list[ToolingCaseScore]:
    return [score for score in scored if score.expected_tool is not None]


def _attempted_calls(scored: list[ToolingCaseScore]) -> list[ToolingCaseScore]:
    return [score for score in scored if score.attempted]


def score_tooling(
    cases: list[ToolingCase], calls: list[ToolCall | None], catalog: dict[str, ToolDef]
) -> ToolingScore:
    """Aggregate the four objective tooling metrics over a model's calls (aligned by index)."""
    if len(cases) != len(calls):
        raise ValueError("cases and calls must be aligned (same length)")
    scored = _score_cases(cases, calls, catalog)
    case_correct = [score.correct for score in scored]
    expecting = _expecting_tool(scored)
    attempted = _attempted_calls(scored)
    return ToolingScore(
        n_cases=len(cases),
        call_accuracy=_rate(case_correct),
        tool_selection_accuracy=_rate([score.tool_selected for score in scored]),
        argument_exactness=_rate([score.arguments_exact for score in expecting]),
        no_hallucinated_tool_rate=_rate([score.no_hallucinated_tool for score in scored]),
        well_formed_rate=_rate([score.well_formed for score in attempted]),
        case_correct=case_correct,
        cases=scored,
    )
