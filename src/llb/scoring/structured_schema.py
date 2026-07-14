"""Focused structured schema implementation."""

import json
from dataclasses import dataclass, field
from typing import Any, Optional
from pydantic import BaseModel, ValidationError, create_model
from llb.prep.frontier import parse_json_block

_PY_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass(frozen=True)
class StructuredCase:
    """One structured-output case: a UA instruction -> a target schema + expected field values."""

    id: str
    instruction: str
    schema: dict[str, dict[str, Any]]  # {field: {"type": ..., "required": bool}}
    expected: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "StructuredCase":
        return cls(
            id=str(record["id"]),
            instruction=str(record["instruction"]),
            schema=dict(record.get("schema", {}) or {}),
            expected=dict(record.get("expected", {}) or {}),
        )


def _field_type(spec: dict[str, Any], name: str) -> Any:
    """Resolve one field spec to a Python/Pydantic type, recursing for nested object/array.

    `{type: object, fields: {...}}` -> a nested model; `{type: array, items: <spec>}` ->
    `list[item type]` (items may itself be an object). A bare `object`/`array` (no `fields`/`items`)
    falls back to `dict`/`list`, the flat behavior.
    """
    kind = str(spec.get("type", "string"))
    if kind == "object" and isinstance(spec.get("fields"), dict):
        return build_model(name, spec["fields"])
    if kind == "array":
        items = spec.get("items")
        if isinstance(items, dict):
            container: Any = list  # subscript via an Any binding (item type is built at runtime)
            return container[_field_type(items, f"{name}_item")]
        return list
    return _PY_TYPES.get(kind, str)


def build_model(name: str, schema: dict[str, dict[str, Any]]) -> type[BaseModel]:
    """Build a Pydantic model from a field schema ({field: {type, required, fields?, items?}}).

    Nested `object` fields and typed `array` items are built recursively, so conformance validates
    the whole shape, not just the top level.
    """
    fields: dict[str, Any] = {}
    for fname, spec in schema.items():
        ftype = _field_type(spec, f"{name}_{fname}")
        if spec.get("required", True):
            fields[fname] = (ftype, ...)
        else:
            fields[fname] = (Optional[ftype], None)
    return create_model(name, **fields)


def parse_output(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from the model output (tolerating a fence/prose), else None."""
    if not text or not text.strip():
        return None
    try:
        parsed = parse_json_block(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def is_conformant(case: StructuredCase, data: dict[str, Any] | None) -> bool:
    """True when `data` validates against the case's schema (required fields + declared types)."""
    if data is None:
        return False
    model = build_model(f"Schema_{case.id}", case.schema)
    try:
        model.model_validate(data)
    except ValidationError:
        return False
    return True
