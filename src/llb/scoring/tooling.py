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

from dataclasses import dataclass, field
from typing import Any

from llb.core.contracts.benchmarks import ToolDef
from llb.scoring.tool_calls import ToolCall, arguments_match, validate_arguments

# JSON-schema primitive type -> accepted Python types.
# Accepted aliases the model/dataset may use for the call envelope.


# --- argument schema validation (lightweight; no jsonschema dep) ---------------------------


# Per-argument match modes (a case may relax exact match for free-text / numeric args; BFCL's
# possible-answer sets, which list several acceptable values per argument, map onto `oneof`).


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
