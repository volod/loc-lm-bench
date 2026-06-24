"""M5.4 structured-output scoring -- JSON-schema conformance + field accuracy via Pydantic (pure).

Validates a model's JSON output against a target schema with Pydantic (no new `jsonschema`
dependency -- Pydantic is already a core dep) and scores field accuracy against expected values.
Two objective signals per case:

  * CONFORMANCE -- the output parses as JSON and validates against the schema's required fields +
    declared types (a `pydantic` model built from the schema);
  * FIELD ACCURACY -- the fraction of expected fields whose value matches (strings casefold/strip-
    insensitive). A non-conformant output scores 0 field accuracy.

The headline (`field_accuracy`, non-conformant == 0) and the conformance rate are both reported.
Everything is pure and unit-tested without a model.
"""

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


def build_model(name: str, schema: dict[str, dict[str, Any]]) -> type[BaseModel]:
    """Build a Pydantic model from a field schema ({field: {type, required}})."""
    fields: dict[str, Any] = {}
    for fname, spec in schema.items():
        ftype = _PY_TYPES.get(str(spec.get("type", "string")), str)
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


def _norm(value: Any) -> Any:
    return value.strip().casefold() if isinstance(value, str) else value


def field_accuracy(expected: dict[str, Any], data: dict[str, Any] | None) -> float:
    """Fraction of expected fields whose value matches (strings casefold/strip-insensitive)."""
    if not expected:
        return 1.0
    if data is None:
        return 0.0
    matched = sum(1 for k, v in expected.items() if k in data and _norm(data[k]) == _norm(v))
    return matched / len(expected)


@dataclass(frozen=True)
class StructuredCaseScore:
    item_id: str
    conformant: float
    field_accuracy: float
    score: float  # headline: field accuracy, 0 when non-conformant


@dataclass(frozen=True)
class StructuredScore:
    n_cases: int
    field_accuracy: float  # headline (mean per-case score)
    conformance_rate: float
    case_score: list[float]
    cases: list[StructuredCaseScore]


def score_case(case: StructuredCase, output: str) -> StructuredCaseScore:
    data = parse_output(output)
    conformant = is_conformant(case, data)
    accuracy = field_accuracy(case.expected, data) if conformant else 0.0
    return StructuredCaseScore(
        item_id=case.id,
        conformant=1.0 if conformant else 0.0,
        field_accuracy=accuracy,
        score=accuracy,
    )


def score_structured(cases: list[StructuredCase], outputs: list[str]) -> StructuredScore:
    """Aggregate conformance + field accuracy over a model's outputs (aligned by index)."""
    if len(cases) != len(outputs):
        raise ValueError("cases and outputs must be aligned (same length)")
    scored = [score_case(c, o) for c, o in zip(cases, outputs)]
    case_score = [s.score for s in scored]
    n = len(scored)
    return StructuredScore(
        n_cases=n,
        field_accuracy=round(sum(case_score) / n, 6) if n else 0.0,
        conformance_rate=round(sum(s.conformant for s in scored) / n, 6) if n else 0.0,
        case_score=case_score,
        cases=scored,
    )
