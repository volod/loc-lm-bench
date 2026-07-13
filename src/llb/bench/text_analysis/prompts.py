"""Candidate prompts, response parsing, and long-document question selection."""

import json
from collections.abc import Sequence

from llb.prep.frontier import parse_json_block
from llb.prompts import render_text, render_text_map
from llb.scoring import text_analysis as ta

JUDGED_EXTRACT_KINDS = (ta.NARRATIVE, ta.INSIGHT)
JUDGE_INTENT = render_text_map("bench.text_analysis.judge_intents")
DEFAULT_LONG_DOC_QUESTION = render_text("bench.text_analysis.long_doc_default_question")
KIND_UA = render_text_map("bench.text_analysis.kind_labels")


def analysis_prompt(doc_id: str, text: str, kinds: Sequence[str]) -> str:
    bullets = "\n".join(f"- {kind}: {KIND_UA.get(kind, kind)}" for kind in kinds)
    return render_text(
        "bench.text_analysis.analysis",
        {"doc_id": doc_id, "text": text, "bullets": bullets, "keys": ", ".join(kinds)},
    )


def parse_predictions(raw: str, kinds: Sequence[str]) -> dict[str, list[str]]:
    payload = parse_json_block(raw)
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object keyed by sub-task")
    predictions: dict[str, list[str]] = {}
    for kind in kinds:
        value = payload.get(kind, [])
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            predictions[kind] = []
            continue
        predictions[kind] = [str(item).strip() for item in value if str(item).strip()]
    return predictions


def long_doc_question(labels: list[ta.PlantedLabel]) -> str | None:
    for label in labels:
        if label.kind == ta.LONG_DOC:
            return str(label.attrs.get("question") or "").strip() or DEFAULT_LONG_DOC_QUESTION
    return None


def parse_or_empty(raw: str, kinds: Sequence[str]) -> dict[str, list[str]]:
    try:
        return parse_predictions(raw, kinds)
    except (ValueError, json.JSONDecodeError):
        return {}
