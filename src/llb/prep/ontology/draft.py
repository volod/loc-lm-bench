"""Stage 5 -- draft Ukrainian question / reference-answer / exact-span triples per seed.

Each seed carries an exact evidence span and a focus (SRO fact or entity). The drafter is
handed a bounded context window around that evidence and asked for ONE grounded QA pair whose
answer is a verbatim quote. The drafts are raw here; stage 6 re-grounds, dedups, and rejects
circular/unsupported items (the answer is never trusted just because the model returned it).
"""

import json
import logging
from typing import Any

from llb.prep.frontier import LLMComplete, parse_json_block
from llb.prep.ontology.constants import DRAFT_CONTEXT_RADIUS
from llb.prep.ontology.models import DocRecord, DraftSeed
from llb.prompts import render_text

_LOG = logging.getLogger(__name__)


def context_window(text: str, char_start: int, char_end: int, radius: int) -> str:
    """A bounded slice of `text` around [char_start, char_end), clamped to the document."""
    start = max(0, char_start - radius)
    end = min(len(text), char_end + radius)
    return text[start:end]


def _focus_line(seed: DraftSeed) -> str:
    if seed.fact is not None:
        f = seed.fact
        return f"Сфокусуйся на факті: {f.subject} | {f.relation} | {f.object}."
    if seed.entity is not None:
        return f"Сфокусуйся на сутності: {seed.entity.name} (тип {seed.entity.type})."
    return "Сфокусуйся на наведеному фрагменті."


def draft_prompt(seed: DraftSeed, context: str, ontology_hint: str = "") -> str:
    """One UA QA pair grounded in `context`, answer = exact substring, difficulty-aware.

    `ontology_hint` (optional, verified-data hardening) carries the corpus's high-confidence induced types as an
    explicit constraint, nudging the drafter toward the reliable types of THIS corpus."""
    return render_text(
        "prep.ontology.draft",
        {
            "focus_line": _focus_line(seed),
            "ontology_hint_line": f"{ontology_hint}\n" if ontology_hint else "",
            "difficulty": seed.difficulty,
            "context": context,
        },
    )


def draft_for_seed(
    complete: LLMComplete, doc_text: str, seed: DraftSeed, ontology_hint: str = ""
) -> dict[str, Any] | None:
    """Draft one raw QA dict for `seed` (tagged with its doc_id), or None on failure."""
    context = context_window(
        doc_text, seed.evidence.char_start, seed.evidence.char_end, DRAFT_CONTEXT_RADIUS
    )
    try:
        payload = parse_json_block(complete(draft_prompt(seed, context, ontology_hint)))
    except json.JSONDecodeError:
        _LOG.warning("[ontology] unparseable draft for %s seed; skipping", seed.doc_id)
        return None
    except Exception as exc:  # endpoint/transport error -> skip this seed, keep going
        _LOG.warning("[ontology] draft call failed for %s: %s", seed.doc_id, exc)
        return None
    if not isinstance(payload, dict):
        _LOG.warning("[ontology] draft for %s is not a JSON object; skipping", seed.doc_id)
        return None
    payload["doc_id"] = seed.doc_id
    return payload


def draft_items(
    complete: LLMComplete,
    docs: list[DocRecord],
    seeds: list[DraftSeed],
    ontology_hint: str = "",
) -> list[dict[str, Any]]:
    """Draft a raw QA dict per seed; failures are skipped (not fatal). `ontology_hint` (verified-data hardening)
    carries the induced high-confidence types into every draft prompt as an explicit constraint."""
    by_id = {doc.doc_id: doc for doc in docs}
    drafts: list[dict[str, Any]] = []
    for seed in seeds:
        doc = by_id.get(seed.doc_id)
        if doc is None:
            continue
        draft = draft_for_seed(complete, doc.text, seed, ontology_hint)
        if draft is not None:
            drafts.append(draft)
    _LOG.info("[ontology] stage 5: %d raw drafts from %d seeds", len(drafts), len(seeds))
    return drafts
