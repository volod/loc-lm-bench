"""structured-output scoring -- JSON-schema conformance + field accuracy via Pydantic (pure).

Validates a model's JSON output against a target schema with Pydantic (no new `jsonschema`
dependency -- Pydantic is already a core dep) and scores field accuracy against expected values.
Two objective signals per case:

  * CONFORMANCE -- the output parses as JSON and validates against the schema's required fields +
    declared types (a `pydantic` model built from the schema). Schemas may be NESTED: a field of
    `type: object` carries `fields` (a sub-schema) and a field of `type: array` carries `items`
    (an element spec, scalar or object), so nested objects and array items are validated too.
  * FIELD ACCURACY -- the fraction of expected LEAF values that match, recursing into nested objects
    and array items (so a 2-field address counts as 2 leaves). Strings match casefold/strip-
    insensitive; numbers match exactly unless the field spec sets a `tolerance` (abs numeric
    tolerance). A non-conformant output scores 0 field accuracy.

The headline (`field_accuracy`, non-conformant == 0) and the conformance rate are both reported.
Everything is pure and unit-tested without a model.
"""

from dataclasses import dataclass
from typing import Any


from llb.scoring.structured_schema import StructuredCase, is_conformant, parse_output


def _norm(value: Any) -> Any:
    return value.strip().casefold() if isinstance(value, str) else value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _leaf_match(expected: Any, actual: Any, spec: dict[str, Any] | None) -> bool:
    """Compare two scalar values under the field spec.

    Numbers: exact unless `tolerance` (abs) or `rel_tolerance` (relative). Strings: casefold/strip
    exact unless `string_match` relaxes it to `fuzzy` (difflib ratio >= `threshold`, default 0.85)
    or `contains` (expected normalized is a substring of actual). Otherwise `==`.
    """
    spec = spec or {}
    if actual is None:
        return expected is None
    if _is_number(expected) and _is_number(actual):
        tol = spec.get("tolerance")
        rel = spec.get("rel_tolerance")
        if tol is not None:
            return abs(float(expected) - float(actual)) <= float(tol)
        if rel is not None:
            return abs(float(expected) - float(actual)) <= float(rel) * abs(float(expected))
        return float(expected) == float(actual)
    if isinstance(expected, str) and isinstance(actual, str):
        mode = str(spec.get("string_match", "exact"))
        exp, act = _norm(expected), _norm(actual)
        if mode == "contains":
            return bool(exp in act)
        if mode == "fuzzy":
            import difflib

            return difflib.SequenceMatcher(None, exp, act).ratio() >= float(
                spec.get("threshold", 0.85)
            )
        return bool(exp == act)
    return bool(expected == actual)


def _compare_unordered(
    expected_items: list[Any], actual: Any, item_spec: dict[str, Any] | None
) -> tuple[int, int]:
    """Order-insensitive array compare: greedily assign each expected item to its best-matching
    UNUSED actual item, summing matched/total leaves regardless of position."""
    actual_items = actual if isinstance(actual, list) else []
    used: set[int] = set()
    matched_sum = total_sum = 0
    for exp in expected_items:
        _, item_total = _compare(exp, None, item_spec)
        total_sum += item_total
        best_matched, best_j = 0, -1
        for j, act in enumerate(actual_items):
            if j in used:
                continue
            cm, _ = _compare(exp, act, item_spec)
            if cm > best_matched:
                best_matched, best_j = cm, j
        if best_j >= 0:
            used.add(best_j)
            matched_sum += best_matched
    return matched_sum, total_sum


def _compare_object(
    expected: dict[str, Any], actual: Any, spec: dict[str, Any] | None
) -> tuple[int, int]:
    subspecs = (spec or {}).get("fields") or {}
    matched = total = 0
    for key, exp_val in expected.items():
        act_val = actual.get(key) if isinstance(actual, dict) else None
        cm, ct = _compare(exp_val, act_val, subspecs.get(key))
        matched += cm
        total += ct
    return matched, total


def _compare_ordered_array(
    expected: list[Any], actual: Any, item_spec: dict[str, Any] | None
) -> tuple[int, int]:
    matched = total = 0
    for i, exp_val in enumerate(expected):
        act_val = actual[i] if isinstance(actual, list) and i < len(actual) else None
        cm, ct = _compare(exp_val, act_val, item_spec)
        matched += cm
        total += ct
    return matched, total


def _compare_array(
    expected: list[Any], actual: Any, spec: dict[str, Any] | None
) -> tuple[int, int]:
    item_spec = (spec or {}).get("items") if spec else None
    if (spec or {}).get("unordered"):
        return _compare_unordered(expected, actual, item_spec)
    return _compare_ordered_array(expected, actual, item_spec)


def _compare(expected: Any, actual: Any, spec: dict[str, Any] | None) -> tuple[int, int]:
    """Recursively count (matched_leaves, total_leaves) of `expected` vs `actual`.

    Objects recurse per expected key (using the spec's `fields`); arrays recurse per expected index
    (using the spec's `items`), or order-insensitively when the array spec sets `unordered: true`;
    everything else is one scalar leaf compared via `_leaf_match`.
    """
    if isinstance(expected, dict):
        return _compare_object(expected, actual, spec)
    if isinstance(expected, list):
        return _compare_array(expected, actual, spec)
    return (1 if _leaf_match(expected, actual, spec) else 0), 1


def field_accuracy(
    expected: dict[str, Any],
    data: dict[str, Any] | None,
    schema: dict[str, dict[str, Any]] | None = None,
) -> float:
    """Fraction of expected LEAF values that match, recursing into nested objects + array items.

    `schema` (the case's field schema) supplies per-field `tolerance` and the nested `fields`/`items`
    structure; with no schema this degrades to the flat top-level comparison.
    """
    if not expected:
        return 1.0
    if data is None:
        return 0.0
    matched, total = _compare(expected, data, {"type": "object", "fields": schema or {}})
    return matched / total if total else 1.0


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
    accuracy = field_accuracy(case.expected, data, case.schema) if conformant else 0.0
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
