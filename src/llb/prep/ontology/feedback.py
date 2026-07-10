"""Rejection-feedback loop for the drafting pipeline (draft-feedback-rejection-reasons).

The verify gate exports WHY items were rejected (`rejection_reasons.json`, written by
`llb.goldset.verify.write_rejection_reasons` from the closed reject-code set). This module
closes the loop: `prepare-goldset-draft --rejection-feedback <file>` maps each dominant reject
code to a deterministic draft-prompt hint, so a re-draft after a failed acceptance does not get
the same prompts that produced the rejected items. The applied hints and the feedback file's
digest land in the bundle's `provenance.json`, so a bundle always names the feedback it was
drafted under.

Deterministic by construction: the code -> hint mapping is a fixed table over the closed code
set, hints are ordered by rejection count (ties by code), and the one example carried into a
hint is the first rejected item's note. No learned prompt optimizer, no re-drafting loop.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

# One prompt hint per closed reject code (`llb.goldset.verify.REJECT_CODES`). The hints speak
# the drafter's language (Ukrainian, like the draft prompt) and each names the failure the
# reviewers actually rejected for.
REJECT_CODE_HINTS: dict[str, str] = {
    "ungrounded": (
        "Рецензенти відхиляли питання, чиї відповіді не є точними цитатами з контексту. "
        "Відповідь МУСИТЬ бути дослівним фрагментом наведеного контексту."
    ),
    "circular": (
        "Рецензенти відхиляли циркулярні питання. Питання НЕ МОЖЕ містити або переказувати "
        "власну відповідь; його має бути неможливо розв'язати без контексту."
    ),
    "wrong_reference": (
        "Рецензенти відхиляли неправильні еталонні відповіді. Перевір, що відповідь точно "
        "відповідає тому, що стверджує контекст, а не схожому твердженню."
    ),
    "label_mismatch": (
        "Рецензенти відхиляли розмітку, що не збігається з документом. Кожна мітка мусить "
        "дослівно відповідати тексту документа."
    ),
    "bad_question": (
        "Рецензенти відхиляли тривіальні або неоднозначні питання (номери сторінок, зміст, "
        "формальні дрібниці). Питай про змістовні факти, які має знати користувач корпусу."
    ),
    "other": (
        "Частину чернеток рецензенти відхилили з інших причин. Формулюй питання чітко, "
        "однозначно і лише про те, що прямо підтверджує контекст."
    ),
}


def load_rejection_feedback(path: Path | str) -> dict[str, Any]:
    """Read a `rejection_reasons.json` summary; raises on a file that is not one."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("by_code"), dict):
        raise ValueError(f"{path}: not a rejection_reasons.json summary (missing by_code)")
    return payload


def feedback_hints(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic hint list from a rejection summary, dominant codes first.

    Each entry carries the code, its rejection count, the mapped prompt hint, and the first
    rejected item's note as the example (when one was recorded). An empty summary is a no-op.
    """
    by_code = summary.get("by_code")
    if not isinstance(by_code, dict):
        return []
    hints: list[dict[str, Any]] = []
    for code, cell in by_code.items():
        if not isinstance(cell, dict):
            continue
        hint = REJECT_CODE_HINTS.get(str(code))
        if hint is None:
            _LOG.warning("[ontology] unknown reject code in feedback: %s (skipped)", code)
            continue
        count = int(cell.get("count") or 0)
        if count < 1:
            continue
        example = ""
        items = cell.get("items")
        if isinstance(items, list):
            example = next(
                (
                    str(item.get("note") or "").strip()
                    for item in items
                    if isinstance(item, dict) and str(item.get("note") or "").strip()
                ),
                "",
            )
        hints.append({"code": str(code), "count": count, "hint": hint, "example": example})
    hints.sort(key=lambda h: (-h["count"], h["code"]))
    return hints


def feedback_hint_text(hints: list[dict[str, Any]]) -> str:
    """The prompt block appended to the draft hint line ("" when there is nothing to apply)."""
    if not hints:
        return ""
    lines = ["Врахуй відгук рецензентів попередньої чернетки:"]
    for entry in hints:
        line = f"- {entry['hint']}"
        if entry["example"]:
            line += f" (приклад відхилення: {entry['example']})"
        lines.append(line)
    return "\n".join(lines)


def feedback_digest(path: Path | str) -> str:
    """sha256 of the feedback file, recorded in provenance so the source is pinned."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def applied_feedback_block(path: Path | str, hints: list[dict[str, Any]]) -> dict[str, Any]:
    """The `provenance.json` record of what feedback was applied to this draft run."""
    return {
        "source": str(path),
        "sha256": feedback_digest(path),
        "hints": [
            {"code": entry["code"], "count": entry["count"], "example": entry["example"]}
            for entry in hints
        ],
    }
