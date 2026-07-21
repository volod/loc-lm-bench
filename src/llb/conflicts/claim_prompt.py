"""Prompt and response parsing for the claim-adjudication tier.

The model is asked one narrow question about one pair of spans: what is the relation between the
single most important claim in A and its counterpart in B? It is never shown dates, versions, or
document names, so it cannot rationalize a verdict from provenance -- `superseded_by` is derived
afterwards from the governance fields, not requested here.

The model must quote both claims verbatim. Quoting is what lets `ground_span` map each claim back
to exact source offsets; a claim that cannot be located falls back to the enclosing chunk span and
is marked `offsets_exact: false` rather than silently pointing at text that is not there.
"""

from typing import Any

from llb.conflicts.constants import (
    MODEL_RELATIONS,
    REL_COMPLEMENTARY,
    REL_CONTRADICTS,
    REL_DUPLICATE,
    REL_SUBSUMED_BY,
    REL_SUBSUMES,
)
from llb.prep.frontier_parsing import parse_json_block

_RELATION_GUIDE = f"""\
- "{REL_DUPLICATE}": both passages assert the same fact with the same specificity.
- "{REL_SUBSUMES}": A asserts everything B asserts and is strictly more specific or complete.
- "{REL_SUBSUMED_BY}": B asserts everything A asserts and is strictly more specific or complete.
- "{REL_CONTRADICTS}": both assert something about the same subject and they cannot both be true.
- "{REL_COMPLEMENTARY}": same topic, but the passages assert different, compatible facts."""


def adjudication_prompt(text_a: str, text_b: str) -> str:
    """One pair of passages -> a strict-JSON relation verdict."""
    return f"""\
You compare two passages from a document collection and decide how their central claims relate.

Passage A:
\"\"\"
{text_a.strip()}
\"\"\"

Passage B:
\"\"\"
{text_b.strip()}
\"\"\"

Choose exactly one relation between the central claim of A and the central claim of B:
{_RELATION_GUIDE}

Rules:
- Judge only what the passages state. Do not use outside knowledge.
- "claim_a" must be copied VERBATIM from passage A, "claim_b" VERBATIM from passage B. Quote the
  single sentence or clause that carries the claim, not the whole passage.
- If the passages are about different subjects, answer "{REL_COMPLEMENTARY}".

Reply with JSON only, no prose and no code fence:
{{"relation": "<one of the five>", "confidence": <0.0-1.0>,
 "claim_a": "<verbatim from A>", "claim_b": "<verbatim from B>",
 "rationale": "<one short sentence>"}}"""


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return value.strip() if isinstance(value, str) else ""


def _confidence(payload: dict[str, Any]) -> float:
    try:
        value = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


class AdjudicationError(ValueError):
    """The completion was not a usable relation verdict."""


def parse_adjudication(completion: str) -> dict[str, Any]:
    """Parse a verdict; raises `AdjudicationError` when the relation is missing or unknown."""
    try:
        payload = parse_json_block(completion)
    except Exception as exc:  # noqa: BLE001 -- any malformed completion is one failure mode
        raise AdjudicationError(f"completion was not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdjudicationError(f"expected a JSON object, got {type(payload).__name__}")
    relation = _text(payload, "relation").lower()
    if relation not in MODEL_RELATIONS:
        raise AdjudicationError(f"unknown relation {relation!r}; expected one of {MODEL_RELATIONS}")
    return {
        "relation": relation,
        "confidence": _confidence(payload),
        "claim_a": _text(payload, "claim_a"),
        "claim_b": _text(payload, "claim_b"),
        "rationale": _text(payload, "rationale"),
    }
