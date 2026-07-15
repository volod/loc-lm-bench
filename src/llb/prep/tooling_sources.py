"""tooling benchmark Berkeley Function-Calling Leaderboard (BFCL) adapter -- UA-adapted tooling cases.

Maps the public BFCL cases into the project's tooling bundle shape (a `{tools, cases}` object the
`bench-tooling` runner loads), so the function-calling board can score against the established BFCL
catalog, not only the committed hand-authored UA seed. BFCL splits the data across two files:

  * the function-doc file -- one entry per case `{id, question, function: [schema, ...]}`;
  * the possible-answer file -- `{id, ground_truth: [{func_name: {arg: [acceptable, ...]}}]}`,
    where each argument lists SEVERAL acceptable values. That list maps directly onto the scorer's
    per-argument `oneof` tolerance (tooling benchmark), so a free-text / formatting variant still counts.

UA adaptation is INJECTABLE: pass `translate` to render the BFCL instruction in Ukrainian; the tool
SCHEMAS (names + JSON parameters) are kept verbatim, as tool identifiers are language-neutral. The
raw entries are passed IN (loaded from a local BFCL checkout under its own license), so nothing is
vendored or fetched at import; everything here is pure and unit-tested from a small fixture.
"""

import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from llb.core.contracts.benchmarks import ToolDef

_LOG = logging.getLogger(__name__)

Translate = Callable[[str], str]


def _question_text(question: Any) -> str:
    """Extract the user instruction from BFCL's `question` (nested turns, a flat list, or a str)."""
    if isinstance(question, str):
        return question.strip()
    turns = question
    if isinstance(turns, list) and turns and isinstance(turns[0], list):
        turns = turns[0]  # first turn of a multi-turn case
    if isinstance(turns, list):
        users = [
            m.get("content", "") for m in turns if isinstance(m, dict) and m.get("role") == "user"
        ]
        if users:
            return str(users[-1]).strip()
        if turns and isinstance(turns[0], dict):
            return str(turns[0].get("content", "")).strip()
    return ""


def _tooldefs(functions: Any) -> list[ToolDef]:
    out: list[ToolDef] = []
    for fn in functions or []:
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        out.append(
            {
                "name": str(fn["name"]),
                "description": str(fn.get("description", "")),
                "parameters": dict(fn.get("parameters", {}) or {}),
            }
        )
    return out


def _expected_from_ground_truth(
    ground_truth: Any,
) -> tuple[str | None, dict[str, Any], dict[str, dict[str, Any]]]:
    """One BFCL ground-truth call -> (tool, expected_arguments, arg_match) with `oneof` tolerance."""
    calls = ground_truth
    if isinstance(calls, list):
        calls = calls[0] if calls else {}
    if not isinstance(calls, dict) or not calls:
        return None, {}, {}
    func_name, arg_specs = next(iter(calls.items()))
    expected: dict[str, Any] = {}
    arg_match: dict[str, dict[str, Any]] = {}
    for arg, acceptable in (arg_specs or {}).items():
        values = acceptable if isinstance(acceptable, list) else [acceptable]
        values = [v for v in values if v != ""] or list(
            values
        )  # keep "" only if it is the sole value
        if not values:
            continue
        expected[arg] = values[0]
        if len(values) > 1:
            arg_match[arg] = {"mode": "oneof", "values": values}
    return str(func_name), expected, arg_match


def _answer_by_id(answers: Iterable[dict[str, Any]] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ans in answers or []:
        if isinstance(ans, dict) and "id" in ans:
            out[str(ans["id"])] = ans.get("ground_truth", ans.get("possible_answer"))
    return out


def _add_entry_tools(catalog: dict[str, ToolDef], entry: dict[str, Any]) -> None:
    for tool in _tooldefs(entry.get("function")):
        catalog.setdefault(tool["name"], tool)


def _case_id(entry: dict[str, Any], index: int) -> str:
    return str(entry.get("id", f"bfcl-{index:04d}"))


def _translated_instruction(entry: dict[str, Any], translate: Translate | None) -> str:
    instruction = _question_text(entry.get("question"))
    if translate is not None and instruction:
        return translate(instruction)
    return instruction


def _case_expectations(answer: Any) -> dict[str, Any]:
    tool_name, expected, arg_match = _expected_from_ground_truth(answer)
    expectations: dict[str, Any] = {"expected_tool": tool_name}
    if expected:
        expectations["expected_arguments"] = expected
    if arg_match:
        expectations["arg_match"] = arg_match
    return expectations


def _bfcl_case_record(
    entry: dict[str, Any],
    index: int,
    answers: dict[str, Any],
    translate: Translate | None,
) -> dict[str, Any]:
    case_id = _case_id(entry, index)
    record: dict[str, Any] = {
        "id": case_id,
        "instruction": _translated_instruction(entry, translate),
    }
    if case_id in answers:
        return {**record, **_case_expectations(answers[case_id])}
    return {**record, "expected_tool": None}


def from_bfcl(
    entries: Iterable[dict[str, Any]],
    answers: Iterable[dict[str, Any]] | None = None,
    *,
    translate: Translate | None = None,
) -> dict[str, Any]:
    """Adapt BFCL function-doc entries (+ optional possible-answers) into a `{tools, cases}` bundle."""
    answers_by_case = _answer_by_id(answers)
    catalog: dict[str, ToolDef] = {}
    cases: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        _add_entry_tools(catalog, entry)
        cases.append(_bfcl_case_record(entry, i, answers_by_case, translate))

    _LOG.info("[tooling-sources] adapted %d BFCL cases over %d tools", len(cases), len(catalog))
    return {"tools": list(catalog.values()), "cases": cases}


def load_jsonl_or_json(path: Path | str) -> list[dict[str, Any]]:
    """Load a BFCL file as JSON Lines (the BFCL default) or a JSON array."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        raw = json.loads(text)
        return [dict(r) for r in raw if isinstance(r, dict)] if isinstance(raw, list) else [raw]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
