"""Rewrite step: an optional local-LLM query rewrite through an injected endpoint callable.

OFF by default and only present when explicitly requested. Records the rewritten query.
"""

import logging

from llb.rag.query_prep.base import STEP_REWRITE, QueryEdit, Rewriter

_LOG = logging.getLogger(__name__)


def apply_rewrite(query: str, rewriter: "Rewriter") -> tuple[str, list[QueryEdit], str | None]:
    """Rewrite the query through the injected local-LLM endpoint; record both forms.

    A blank or unchanged rewrite is a no-op (the original query is kept), so a degenerate model
    response never silently drops the question.
    """
    rewritten = (rewriter(query) or "").strip()
    if not rewritten or rewritten == query:
        return query, [], rewritten or None
    _LOG.info("[query-prep] llm rewrite %r -> %r", query, rewritten)
    return (
        rewritten,
        [QueryEdit(STEP_REWRITE, "rewrite", original=query, replacement=rewritten)],
        (rewritten),
    )
