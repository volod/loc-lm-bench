"""M5.2 tooling / function-calling runner -- objective call-only board under TIER_TOOLING.

Drives a candidate over a catalog of tools + UA instruction cases, parses the emitted tool call
(native OpenAI `tool_calls` OR a text-only backend's JSON call -- one scorer for both), and scores
call correctness OBJECTIVELY without executing the tool (execution is M5.3). The headline is
call-accuracy (right tool + exact arguments); tool-selection / argument-exactness /
no-hallucinated-tool / well-formed rates are recorded alongside, all under `TIER_TOOLING` --
never cross-ranked with the RAG board or text-only candidates.

The candidate is reached through an injectable `complete` using a universal TEXT tool-calling
protocol (the catalog is embedded in the prompt; the model returns a JSON call), so every backend
is exercised uniformly and a FAKE endpoint proves the flow. Native FC responses are also parsed.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.bench.common import (
    LLMComplete,
    Mirror,
    category_result,
    persist_category_run,
    render_board,
)
from llb.contracts import BoardRow, RunMetrics, RunPaths, ToolDef, ToolingCaseRow
from llb.scoring import tooling
from llb.scoring.aggregate import TIER_TOOLING, ModelResult, bootstrap_mean_ci

_LOG = logging.getLogger(__name__)

METHOD = "tooling"
TOOL_PROTOCOL_TEXT = "text"  # catalog-in-prompt JSON protocol (works on every backend)


@dataclass(slots=True)
class ToolingRun:
    result: ModelResult
    score: tooling.ToolingScore
    rows: list[ToolingCaseRow]
    board: list[BoardRow]
    table: str
    accuracy_ci: tuple[float, float] | None
    paths: RunPaths | None


def text_tool_prompt(instruction: str, catalog: dict[str, ToolDef]) -> str:
    """A backend-agnostic tool-calling prompt: the catalog as JSON + the user instruction, asking
    for a single JSON tool call (or a null call when no tool is needed)."""
    tools_json = json.dumps(list(catalog.values()), ensure_ascii=False, indent=2)
    return (
        "Ти асистент із доступом до інструментів. Доступні інструменти (назва, опис, JSON-схема "
        f"аргументів):\n{tools_json}\n\n"
        f"Запит користувача: {instruction}\n\n"
        'Якщо потрібен інструмент, поверни ЛИШЕ JSON {"name": <назва інструмента>, '
        '"arguments": {<аргументи за схемою>}}.\n'
        'Якщо жоден інструмент не потрібен, поверни ЛИШЕ {"name": null}.\n'
    )


def _row(score: tooling.ToolingCaseScore) -> ToolingCaseRow:
    return {
        "item_id": score.item_id,
        "expected_tool": score.expected_tool,
        "called_tool": score.called_tool,
        "attempted": score.attempted,
        "tool_selected": score.tool_selected,
        "schema_valid": score.schema_valid,
        "arguments_exact": score.arguments_exact,
        "no_hallucinated_tool": score.no_hallucinated_tool,
        "well_formed": score.well_formed,
        "correct": score.correct,
    }


def run_tooling(
    catalog: dict[str, ToolDef],
    cases: list[tooling.ToolingCase],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    capability: str = TOOL_PROTOCOL_TEXT,
    data_dir: Path | str | None = None,
    run_name: str = "m5-tooling",
    persist: bool = True,
    mirror: Mirror | None = None,
) -> ToolingRun:
    """Score one model's call correctness over the catalog + cases under TIER_TOOLING."""
    if not cases:
        raise SystemExit("no tooling cases provided")
    calls = [
        tooling.parse_tool_call(complete(text_tool_prompt(c.instruction, catalog))) for c in cases
    ]
    score = tooling.score_tooling(cases, calls, catalog)
    rows = [_row(s) for s in score.cases]

    n_responding = sum(1 for s in score.cases if s.attempted or s.expected_tool is None)
    reliability = n_responding / len(cases) if cases else 0.0
    result = category_result(
        model=model,
        backend=backend,
        tier=TIER_TOOLING,
        case_objectives=score.case_correct,
        reliability=reliability,
    )
    accuracy_ci = bootstrap_mean_ci(score.case_correct)
    board, table = render_board([result])

    paths: RunPaths | None = None
    if persist and data_dir is not None:
        metrics: RunMetrics = {
            "objective_score": result.objective_score,  # call accuracy
            "reliability": reliability,
            "tokens_per_s": 0.0,
        }
        config = {
            "model": model,
            "backend": backend,
            "tier": TIER_TOOLING,
            "category": "tooling",
            "tool_protocol": capability,  # per-backend tool-call capability (never cross-ranked)
            "n_cases": score.n_cases,
            "call_accuracy": score.call_accuracy,
            "tool_selection_accuracy": score.tool_selection_accuracy,
            "argument_exactness": score.argument_exactness,
            "no_hallucinated_tool_rate": score.no_hallucinated_tool_rate,
            "well_formed_rate": score.well_formed_rate,
            "call_accuracy_ci": list(accuracy_ci) if accuracy_ci else None,
        }
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=run_name,
            config=config,
            metrics=metrics,
            case_rows=rows,
            mirror=mirror,
        )
        _LOG.info(
            "[tooling] %s call-accuracy=%.3f tool-selection=%.3f args-exact=%.3f -> %s",
            model,
            score.call_accuracy,
            score.tool_selection_accuracy,
            score.argument_exactness,
            paths["manifest"],
        )
    return ToolingRun(
        result=result,
        score=score,
        rows=rows,
        board=board,
        table=table,
        accuracy_ci=accuracy_ci,
        paths=paths,
    )


def load_catalog_file(path: Path | str) -> tuple[dict[str, ToolDef], list[tooling.ToolingCase]]:
    """Load a tooling bundle: a JSON object {"tools": [ToolDef...], "cases": [case...]}."""
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "tools" not in raw or "cases" not in raw:
        raise ValueError(f"{path}: expected an object with 'tools' and 'cases'")
    catalog: dict[str, ToolDef] = {str(tool["name"]): tool for tool in raw["tools"]}
    cases = [tooling.ToolingCase.from_record(c) for c in raw["cases"]]
    return catalog, cases
