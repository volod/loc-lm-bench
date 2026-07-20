"""Tier 4 (`claim`): local-model adjudication of the candidate pairs the cheap tiers found.

Only candidate pairs reach the model, so cost is bounded by the semantic tier's blocking rather
than by corpus size. Each verdict is narrowed to the exact claim spans the model quoted, then two
deterministic post-processing rules apply:

  - a contradiction between two documents the governance fields can order becomes `superseded_by`,
    with side a the deprecated claim -- the one case where dates change the relation itself;
  - a verdict whose quoted claims cannot be located in their passages keeps the enclosing chunk
    span and is marked inexact, never dropped silently.

Because relations are assigned per claim PAIR, one revision can legitimately produce a
`superseded_by` for the fact it replaces and a `duplicate` for the fact it restates. That is the
partial-supersession case, and it needs no special handling here -- it falls out of adjudicating
pairs instead of documents.
"""

import logging
import time
from collections.abc import Iterable
from dataclasses import replace

from llb.conflicts.claim_prompt import AdjudicationError, adjudication_prompt, parse_adjudication
from llb.conflicts.constants import REL_CONTRADICTS, REL_SUPERSEDED_BY, TIER_CLAIM
from llb.conflicts.governance import SIDE_A, compare_editions
from llb.conflicts.models import ClaimRef, Finding, Staleness, TierStats
from llb.conflicts.semantic_tier import chunk_claim_ref
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord
from llb.prep.frontier_parsing import ground_span
from llb.prep.frontier_telemetry import LLMComplete

_LOG = logging.getLogger(__name__)

EVIDENCE_MODEL = "model"


def narrow_to_claim(chunk: ChunkRecord, claim_text: str, governance: JsonObject) -> ClaimRef:
    """Locate `claim_text` inside the chunk and return its exact corpus offsets.

    Falls back to the whole chunk span when the quote cannot be grounded, flagging the result
    rather than inventing offsets.
    """
    base = chunk_claim_ref(chunk, governance)
    if not claim_text:
        return replace(base, offsets_exact=False)
    located = ground_span(chunk["text"], claim_text)
    if located is None:
        return replace(base, offsets_exact=False)
    offset, exact = located
    start = int(chunk["char_start"]) + offset
    return ClaimRef(
        doc_id=base.doc_id,
        char_start=start,
        char_end=start + len(exact),
        text=exact,
        chunk_id=base.chunk_id,
        offsets_exact=True,
        governance=governance,
    )


def apply_supersession(
    relation: str, a: ClaimRef, b: ClaimRef
) -> tuple[str, ClaimRef, ClaimRef, Staleness]:
    """Promote a datable contradiction to `superseded_by`, with side a the deprecated claim."""
    staleness = compare_editions(a.governance, b.governance)
    if relation != REL_CONTRADICTS or staleness.newer_side is None:
        return relation, a, b, staleness
    if staleness.newer_side == SIDE_A:
        # a is the newer edition; flip so the deprecated claim stays on side a.
        return REL_SUPERSEDED_BY, b, a, Staleness(newer_side="b", basis=staleness.basis)
    return REL_SUPERSEDED_BY, a, b, staleness


def adjudicate_pairs(
    pairs: Iterable[tuple[int, int, float]],
    chunks: list[ChunkRecord],
    governance_by_doc: dict[str, JsonObject],
    complete: LLMComplete,
) -> tuple[list[Finding], TierStats]:
    """Ask the model for a relation per candidate pair and build the claim-level findings."""
    started = time.monotonic()
    stats = TierStats(tier=TIER_CLAIM)
    findings: list[Finding] = []
    failures = 0
    calls = 0

    for left, right, similarity in pairs:
        left_chunk, right_chunk = chunks[left], chunks[right]
        calls += 1
        try:
            verdict = parse_adjudication(
                complete(adjudication_prompt(left_chunk["text"], right_chunk["text"]))
            )
        except (AdjudicationError, RuntimeError) as exc:
            failures += 1
            _LOG.warning(
                "[conflicts] claim adjudication failed for %s vs %s: %s",
                left_chunk.get("chunk_id"),
                right_chunk.get("chunk_id"),
                exc,
            )
            continue
        a = narrow_to_claim(
            left_chunk, verdict["claim_a"], governance_by_doc.get(left_chunk["doc_id"], {})
        )
        b = narrow_to_claim(
            right_chunk, verdict["claim_b"], governance_by_doc.get(right_chunk["doc_id"], {})
        )
        relation, a, b, staleness = apply_supersession(verdict["relation"], a, b)
        findings.append(
            Finding(
                relation=relation,
                tier=TIER_CLAIM,
                a=a,
                b=b,
                score=verdict["confidence"],
                evidence=EVIDENCE_MODEL,
                staleness=staleness,
                rationale=verdict["rationale"] or f"model verdict at chunk cosine {similarity:.3f}",
            )
        )

    stats.candidate_pairs = calls
    stats.findings = len(findings)
    stats.seconds = time.monotonic() - started
    stats.extra = {"model_calls": calls, "unparsed_verdicts": failures}
    return findings, stats
