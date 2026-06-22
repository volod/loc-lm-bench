"""Exact-evidence grounding shared across stages.

Reuses `frontier.ground_span` (exact substring, then casefold/whitespace-normalized match
mapped back to EXACT original offsets) and lifts the result into a canonical `SourceSpan`, so
extraction, the ontology, and the drafts all anchor to the same offset-exact representation.
A quote that is not in the document grounds to None and is dropped -- a label/fact can never
point at text that is not there.
"""

from llb.goldset.schema import SourceSpan
from llb.prep.frontier import ground_span


def ground_quote(doc_id: str, doc_text: str, quote: str) -> SourceSpan | None:
    """Ground `quote` against `doc_text`, returning an exact `SourceSpan` or None."""
    grounded = ground_span(doc_text, quote)
    if grounded is None:
        return None
    start, exact = grounded
    return SourceSpan(doc_id=doc_id, char_start=start, char_end=start + len(exact), text=exact)
