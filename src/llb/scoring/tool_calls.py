"""Focused tool calls implementation."""

import json
from dataclasses import dataclass
from typing import Any
from llb.core.contracts import ToolDef
from llb.prep.frontier import parse_json_block

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}

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
