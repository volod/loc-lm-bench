"""Answer-side metrics read off the DECLARED envelope instead of scraped from prose.

Every number here has a free-text twin in `llb.scoring.groundedness`, and the two are deliberately
computed the same way -- same support threshold, same countable-claim rule, same prompt-layout
citation numbering. Only the SOURCE of the claims and citations differs: a declared `claims[]`
entry with its `citations` list, rather than a punctuation-split sentence with `[i]` markers
scraped out of it. That is what makes an envelope run comparable to a free-text run instead of a
different measurement wearing the same column names.

Reading the declaration removes two known distortions of the scraped version: a claim that spans
two sentences is one claim (the model said so) rather than two, and a `[i]` marker that happens to
sit in a neighboring sentence is attributed to the claim the model attached it to.
"""

from llb.core.contracts.rag import ChunkRecord
from llb.eval.answer_envelope.models import AnswerEnvelope, EnvelopeClaim
from llb.scoring.groundedness import (
    MIN_CLAIM_TOKENS,
    CitationReport,
    chunk_supports_claim,
    content_tokens,
)


def countable_claims(envelope: AnswerEnvelope) -> list[EnvelopeClaim]:
    """The declared claims long enough to judge (>= `MIN_CLAIM_TOKENS` content tokens).

    The same floor the free-text scorer applies, so a one-word claim is not counted as grounded or
    ungrounded on either path.
    """
    return [
        claim for claim in envelope.claims if len(content_tokens(claim.text)) >= MIN_CLAIM_TOKENS
    ]


def envelope_groundedness(envelope: AnswerEnvelope, ordered_chunks: list[ChunkRecord]) -> float:
    """Share of the declared countable claims supported by ANY chunk the prompt carried.

    An envelope with no countable claim scores 0.0 -- including a declared abstention, which
    asserts nothing the context could support. The abstention is recorded as its own flag, so the
    zero here never has to stand in for one.
    """
    claims = countable_claims(envelope)
    if not claims:
        return 0.0
    texts = [str(chunk.get("text", "")) for chunk in ordered_chunks]
    supported = sum(
        1 for claim in claims if any(chunk_supports_claim(claim.text, text) for text in texts)
    )
    return supported / len(claims)


def envelope_citation_report(
    envelope: AnswerEnvelope, ordered_chunks: list[ChunkRecord]
) -> CitationReport:
    """Validate each claim's declared citations against the chunk each index points at.

    Indices are PROMPT positions, so a run under `reverse_rank` is validated against the order the
    model actually saw -- exactly as the scraped `[i]` validation already is. An out-of-range index
    is hallucinated; an in-range index whose chunk does not support the claim is invalid but not
    hallucinated.
    """
    n = len(ordered_chunks)
    n_citations = 0
    n_valid = 0
    n_hallucinated = 0
    claims = countable_claims(envelope)
    for claim in envelope.claims:
        for index in claim.citations:
            n_citations += 1
            if index < 1 or index > n:
                n_hallucinated += 1
                continue
            if chunk_supports_claim(claim.text, str(ordered_chunks[index - 1].get("text", ""))):
                n_valid += 1
    n_covered = sum(1 for claim in claims if claim.citations)
    return CitationReport(
        n_citations=n_citations,
        n_valid=n_valid,
        n_hallucinated=n_hallucinated,
        n_claims=len(claims),
        n_covered_claims=n_covered,
        citation_validity=(n_valid / n_citations) if n_citations else 0.0,
        hallucinated_citation_rate=(n_hallucinated / n_citations) if n_citations else 0.0,
        citation_coverage=(n_covered / len(claims)) if claims else 0.0,
    )
