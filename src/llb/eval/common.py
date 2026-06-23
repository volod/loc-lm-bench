"""Shared primitives for the LangGraph eval templates.

The design standardizes every eval flow on LangGraph (single-call, map-reduce, multi-hop)
so they share one status taxonomy and one untrusted-context renderer. Those shared pieces
live here; each template module (`graph.py`, `map_reduce.py`, `multi_hop.py`) imports them
instead of redefining them. Everything in this module is a pure function or constant, so it
is unit-testable WITHOUT langgraph installed (only the `build_*_graph` functions import it).

Failure taxonomy (design "distinct typed cases"): each case ends in exactly one status --
ok / empty / malformed / refusal / timeout / backend_error / retrieval_miss -- recorded
separately, never collapsed into a single "reliability failure".
"""

from llb.contracts import ChunkRecord

# Terminal case statuses (shared across all templates).
OK = "ok"
EMPTY = "empty"
MALFORMED = "malformed"
REFUSAL = "refusal"
RETRIEVAL_MISS = "retrieval_miss"
# transport tokens (timeout / backend_error) are passed through from ChatResult.error.

# Markers a model uses when it declines to answer (UA + EN).
_REFUSAL_MARKERS = (
    "не можу відповісти",
    "не маю можливості",
    "не можу допомогти",
    "вибачте, але я",
    "i cannot answer",
    "i can't answer",
    "i'm unable to",
    "as an ai",
)


def format_context(chunks: list[ChunkRecord]) -> str:
    """Render retrieved chunks as a delimited, numbered block (corpus is untrusted input)."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] ({chunk.get('doc_id', '?')})\n{chunk.get('text', '').strip()}")
    return "\n\n".join(parts)


def classify_response(text: str, error: str | None, expect_json: bool = False) -> str:
    """Map a raw model response to a terminal status."""
    if error:
        return error  # "timeout" | "backend_error", passed through verbatim
    if text is None or not text.strip():
        return EMPTY
    stripped = text.strip()
    low = stripped.lower()
    if any(marker in low for marker in _REFUSAL_MARKERS):
        return REFUSAL
    if expect_json:
        import json

        try:
            json.loads(stripped)
        except (ValueError, TypeError):
            return MALFORMED
    return OK
