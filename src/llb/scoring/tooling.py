"""M5.2 tooling / function-calling scoring -- objective, CALL-ONLY (pure).

Scores a candidate's emitted tool call against an expected (tool name + argument JSON), without
EXECUTING the tool (execution is the M5.3 agentic benchmark). Two layers, both pure:

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
the M5.4 structured-output category).
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


def parse_tool_call(raw: Any) -> ToolCall | None:
    """Normalize a backend response into a `ToolCall`, or None when no tool call was attempted.

    Handles a native OpenAI message (an object exposing `.tool_calls`), a pre-parsed dict, and a
    text response carrying a JSON tool call (the text-only-backend fallback). A non-JSON text
    answer means the model did NOT attempt a tool call -> None.
    """
    if raw is None:
        return None
    tool_calls = getattr(raw, "tool_calls", None)
    if tool_calls:
        fn = tool_calls[0].function
        arguments, well = _load_args(getattr(fn, "arguments", "") or "")
        return ToolCall(
            name=str(getattr(fn, "name", "") or ""),
            arguments=arguments,
            well_formed=well and bool(getattr(fn, "name", "")),
            raw=str(getattr(fn, "arguments", "")),
        )
    if isinstance(raw, dict):
        return _from_dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            obj = parse_json_block(text)
        except json.JSONDecodeError:
            return None  # a plain text answer is not a tool-call attempt
        return _from_dict(obj, raw=text) if isinstance(obj, dict) else None
    return None


# --- argument schema validation (lightweight; no jsonschema dep) ---------------------------


def validate_arguments(tool: ToolDef, arguments: dict[str, Any]) -> list[str]:
    """Structural check of `arguments` against a tool's `parameters` schema: required present,
    declared types, no unknown properties. Returns a list of human-readable errors ([] == valid)."""
    schema = tool.get("parameters", {}) or {}
    properties: dict[str, Any] = schema.get("properties", {}) or {}
    required: list[str] = list(schema.get("required", []) or [])
    errors: list[str] = []
    for name in required:
        if name not in arguments:
            errors.append(f"missing required argument: {name}")
    for name, value in arguments.items():
        if name not in properties:
            errors.append(f"unknown argument: {name}")
            continue
        declared = properties[name].get("type")
        accepted = _TYPE_MAP.get(declared) if declared else None
        # bool is a subclass of int -- reject it for integer/number to keep types exact.
        if accepted and (not isinstance(value, accepted) or _bad_bool(declared, value)):
            errors.append(f"argument {name}: expected {declared}, got {type(value).__name__}")
    return errors


def _bad_bool(declared: str, value: Any) -> bool:
    return declared in ("integer", "number") and isinstance(value, bool)


def _normalize_value(value: Any) -> Any:
    return value.casefold().strip() if isinstance(value, str) else value


def arguments_match(expected: dict[str, Any], provided: dict[str, Any]) -> bool:
    """Exact argument match: same key set, each value equal (strings casefold/strip-insensitive)."""
    if set(expected) != set(provided):
        return False
    return all(_normalize_value(provided[k]) == _normalize_value(v) for k, v in expected.items())


# --- per-case + aggregate scoring ----------------------------------------------------------


@dataclass(frozen=True)
class ToolingCase:
    """One function-calling case: a UA instruction -> expected tool + argument JSON."""

    id: str
    instruction: str
    expected_tool: str | None  # None == the model should NOT call any tool
    expected_arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ToolingCase":
        return cls(
            id=str(record["id"]),
            instruction=str(record["instruction"]),
            expected_tool=record.get("expected_tool"),
            expected_arguments=dict(record.get("expected_arguments", {}) or {}),
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


def score_case(
    case: ToolingCase, call: ToolCall | None, catalog: dict[str, ToolDef]
) -> ToolingCaseScore:
    """Score one tooling case against the candidate's (parsed) tool call."""
    attempted = call is not None
    in_catalog = call is not None and call.name in catalog
    no_hallucinated = 1.0 if (not attempted or in_catalog) else 0.0

    if case.expected_tool is None:
        correct = 0.0 if attempted else 1.0
        return ToolingCaseScore(
            item_id=case.id,
            expected_tool=None,
            called_tool=call.name if call else None,
            attempted=attempted,
            tool_selected=correct,
            schema_valid=0.0,
            arguments_exact=0.0,
            no_hallucinated_tool=no_hallucinated,
            well_formed=(1.0 if (call and call.well_formed) else 0.0) if attempted else 0.0,
            correct=correct,
        )

    tool_selected = 1.0 if (call and call.name == case.expected_tool) else 0.0
    schema_valid = 0.0
    arguments_exact = 0.0
    if tool_selected and call is not None and in_catalog:
        errors = validate_arguments(catalog[call.name], call.arguments)
        schema_valid = 1.0 if (call.well_formed and not errors) else 0.0
        if schema_valid and arguments_match(case.expected_arguments, call.arguments):
            arguments_exact = 1.0
    return ToolingCaseScore(
        item_id=case.id,
        expected_tool=case.expected_tool,
        called_tool=call.name if call else None,
        attempted=attempted,
        tool_selected=tool_selected,
        schema_valid=schema_valid,
        arguments_exact=arguments_exact,
        no_hallucinated_tool=no_hallucinated,
        well_formed=(1.0 if (call and call.well_formed) else 0.0) if attempted else 0.0,
        correct=arguments_exact,  # full correctness requires the right tool AND exact args
    )


def _rate(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def score_tooling(
    cases: list[ToolingCase], calls: list[ToolCall | None], catalog: dict[str, ToolDef]
) -> ToolingScore:
    """Aggregate the four objective tooling metrics over a model's calls (aligned by index)."""
    if len(cases) != len(calls):
        raise ValueError("cases and calls must be aligned (same length)")
    scored = [score_case(c, call, catalog) for c, call in zip(cases, calls)]
    case_correct = [s.correct for s in scored]
    expecting = [s for s in scored if s.expected_tool is not None]
    attempted = [s for s in scored if s.attempted]
    return ToolingScore(
        n_cases=len(cases),
        call_accuracy=_rate(case_correct),
        tool_selection_accuracy=_rate([s.tool_selected for s in scored]),
        argument_exactness=_rate([s.arguments_exact for s in expecting]),
        no_hallucinated_tool_rate=_rate([s.no_hallucinated_tool for s in scored]),
        well_formed_rate=_rate([s.well_formed for s in attempted]),
        case_correct=case_correct,
        cases=scored,
    )
