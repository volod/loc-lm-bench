"""Answer-side RAG quality: deterministic groundedness + citation validity (groundedness-citation-metrics).

Reference-answer overlap (`llb.scoring.correctness`) says whether the answer matches the gold
answer; it does NOT say whether the answer is SUPPORTED by the retrieved context, nor whether the
model's own `[i]` citations point at chunks that actually contain the claim. This module adds those
answer-side signals, deterministically and dependency-free (pure token/span overlap over the same
normalized text the correctness scorer uses), so they can stand as separate columns beside the
headline objective without a RAGAS dependency or a frontier judge.

Three signals:
  - groundedness fraction: the share of the answer's claims (sentence-ish units) that are supported
    by ANY retrieved chunk via token-overlap matching. A fully-supported answer scores 1.0; an
    answer whose claims are absent from the context scores near 0.0.
  - citation validity: of the `[i]` citations the model emitted, the share whose cited chunk
    actually supports the claim it annotates. A citation pointing at a chunk that lacks the claim is
    flagged invalid.
  - hallucinated-citation rate: the share of citations whose index is out of the retrieved range
    (points at a chunk that was never in the prompt).

The calibration-gated judge's faithfulness score remains the OPTIONAL secondary groundedness signal
(recorded elsewhere only when the judge is trusted); this deterministic scorer is the primary one.
"""

import re

from typing_extensions import TypedDict

from llb.core.contracts import ChunkRecord
from llb.scoring.correctness import normalize

# A claim counts toward groundedness only when it carries at least this many content tokens; shorter
# fragments (a lone number, an interjection) cannot be judged supported/unsupported reliably.
MIN_CLAIM_TOKENS = 2
# Share of a claim's content tokens that must appear in a chunk for the chunk to SUPPORT the claim.
GROUNDEDNESS_SUPPORT_THRESHOLD = 0.6

# `[i]` chunk citation (1-based prompt position, matching `format_context`'s numbering).
_CITATION_RE = re.compile(r"\[(\d+)\]")
# Sentence-ish claim boundaries: terminal punctuation (UA/EN) or a newline.
_CLAIM_SPLIT_RE = re.compile(r"[.!?;\n…]+")


def parse_citations(text: str) -> list[int]:
    """Every `[i]` citation index in `text`, in order (duplicates kept)."""
    return [int(m.group(1)) for m in _CITATION_RE.finditer(text or "")]


def strip_citations(text: str) -> str:
    """Remove `[i]` markers so citation numbers never leak into content-token matching."""
    return _CITATION_RE.sub(" ", text or "")


def split_claims(answer: str) -> list[str]:
    """Split an answer into sentence-ish claims (citations preserved for per-claim attribution)."""
    return [claim.strip() for claim in _CLAIM_SPLIT_RE.split(answer or "") if claim.strip()]


def _content_tokens(text: str) -> list[str]:
    """Normalized content tokens of `text` with `[i]` markers removed."""
    return normalize(strip_citations(text)).split()


def chunk_supports_claim(claim: str, chunk_text: str) -> bool:
    """True when at least `GROUNDEDNESS_SUPPORT_THRESHOLD` of the claim's content tokens are in the chunk.

    Token-recall overlap over normalized text -- deterministic, morphology-agnostic, and symmetric
    with the correctness scorer's normalization. A claim with no content tokens is unsupported.
    """
    claim_tokens = _content_tokens(claim)
    if not claim_tokens:
        return False
    chunk_tokens = set(normalize(chunk_text).split())
    covered = sum(1 for tok in claim_tokens if tok in chunk_tokens)
    return covered / len(claim_tokens) >= GROUNDEDNESS_SUPPORT_THRESHOLD


def _chunk_text(chunk: ChunkRecord) -> str:
    return str(chunk.get("text", ""))


def groundedness_fraction(answer: str, ordered_chunks: list[ChunkRecord]) -> float:
    """Share of the answer's countable claims supported by ANY retrieved chunk.

    Claims shorter than `MIN_CLAIM_TOKENS` content tokens are ignored (too short to judge). Returns
    0.0 when there is no countable claim (an empty or purely-fragmentary answer is not grounded).
    """
    claims = [c for c in split_claims(answer) if len(_content_tokens(c)) >= MIN_CLAIM_TOKENS]
    if not claims:
        return 0.0
    texts = [_chunk_text(c) for c in ordered_chunks]
    supported = sum(1 for claim in claims if any(chunk_supports_claim(claim, t) for t in texts))
    return supported / len(claims)


class CitationReport(TypedDict):
    """Citation-validity signals for one answer."""

    n_citations: int
    n_valid: int
    n_hallucinated: int
    citation_validity: float
    hallucinated_citation_rate: float


def citation_report(answer: str, ordered_chunks: list[ChunkRecord]) -> CitationReport:
    """Validate every `[i]` citation against the chunk it points at (prompt-position order).

    A citation is HALLUCINATED when its index is out of the retrieved range (no such chunk was in
    the prompt) and VALID when its in-range chunk supports the claim (the sentence carrying it). An
    in-range citation whose chunk lacks the claim is neither valid nor hallucinated -- it is a
    flagged invalid citation, lowering `citation_validity` without inflating the hallucination rate.
    """
    n = len(ordered_chunks)
    n_citations = 0
    n_valid = 0
    n_hallucinated = 0
    for claim in split_claims(answer):
        citations = parse_citations(claim)
        if not citations:
            continue
        for index in citations:
            n_citations += 1
            if index < 1 or index > n:
                n_hallucinated += 1
                continue
            if chunk_supports_claim(claim, _chunk_text(ordered_chunks[index - 1])):
                n_valid += 1
    return CitationReport(
        n_citations=n_citations,
        n_valid=n_valid,
        n_hallucinated=n_hallucinated,
        citation_validity=(n_valid / n_citations) if n_citations else 0.0,
        hallucinated_citation_rate=(n_hallucinated / n_citations) if n_citations else 0.0,
    )
