"""HyDE query step: generate a hypothetical answer for dense retrieval."""

import logging

from llb.rag.query_prep.base import STEP_HYDE, QueryEdit, QueryGenerator

_LOG = logging.getLogger(__name__)


def apply_hyde(query: str, generator: QueryGenerator) -> tuple[str | None, list[QueryEdit]]:
    """Generate a short answer-like passage; blank output is a safe no-op."""
    hypothetical = (generator(query) or "").strip()
    if not hypothetical:
        return None, []
    _LOG.info("[query-prep] HyDE generated %d characters", len(hypothetical))
    return hypothetical, [
        QueryEdit(
            STEP_HYDE,
            "hypothetical_answer",
            original=query,
            replacement=hypothetical,
        )
    ]
