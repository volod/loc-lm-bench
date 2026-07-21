"""Query decomposition step and robust local-model output parser."""

import json
import logging
import re
from typing import Any

from llb.rag.query_prep.base import STEP_DECOMPOSE, QueryEdit, QueryGenerator

_LOG = logging.getLogger(__name__)

MAX_SUBQUERIES = 5
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")


def parse_subqueries(text: str, limit: int = MAX_SUBQUERIES) -> tuple[str, ...]:
    """Parse JSON or one-query-per-line output; deduplicate while preserving order."""
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    candidates = _json_subqueries(cleaned)
    if candidates is None:
        candidates = [_LIST_PREFIX_RE.sub("", line).strip() for line in cleaned.splitlines()]
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        query = str(candidate).strip().strip('"')
        marker = query.casefold()
        if not query or marker in seen:
            continue
        seen.add(marker)
        out.append(query)
        if len(out) >= limit:
            break
    return tuple(out)


def _json_subqueries(text: str) -> list[Any] | None:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(payload, dict):
        payload = payload.get("subqueries")
    return payload if isinstance(payload, list) else None


def apply_decompose(
    query: str, generator: QueryGenerator
) -> tuple[tuple[str, ...], list[QueryEdit], str | None]:
    """Generate bounded subqueries; blank/unparseable output is a safe no-op."""
    generated = (generator(query) or "").strip()
    subqueries = parse_subqueries(generated) if generated else ()
    if not subqueries:
        return (), [], generated or None
    _LOG.info("[query-prep] decomposed query into %d subqueries", len(subqueries))
    return (
        subqueries,
        [
            QueryEdit(
                STEP_DECOMPOSE,
                "subqueries",
                original=query,
                replacement="\n".join(subqueries),
            )
        ],
        generated,
    )
