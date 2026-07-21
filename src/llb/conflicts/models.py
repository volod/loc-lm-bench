"""Findings, claim references, and governance/staleness records.

A finding is always a PAIR of claim references plus one relation. Both sides carry exact source
offsets so a downstream resolution lane can act on a span, not on a whole document. `offsets_exact`
records whether the span was located verbatim in the source (model-quoted claims that could not be
matched back fall back to the enclosing chunk span and say so).
"""

from dataclasses import dataclass, field
from typing import Any

from llb.core.contracts.common import JsonObject


@dataclass(frozen=True)
class ClaimRef:
    """One side of a finding: a span of one corpus document."""

    doc_id: str
    char_start: int
    char_end: int
    text: str
    chunk_id: str | None = None
    offsets_exact: bool = True
    governance: JsonObject = field(default_factory=dict)

    def payload(self) -> JsonObject:
        return {
            "doc_id": self.doc_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text": self.text,
            "chunk_id": self.chunk_id,
            "offsets_exact": self.offsets_exact,
            "governance": dict(self.governance),
        }


@dataclass(frozen=True)
class Staleness:
    """Which side of a finding is the newer edition, and what decided it.

    `newer_side` is `"a"`, `"b"`, or None when the governance fields cannot order the pair. It is
    orthogonal to the relation: a duplicate pair can be dated, and a contradiction need not be.
    """

    newer_side: str | None = None
    basis: str | None = None

    def payload(self) -> JsonObject:
        return {"newer_side": self.newer_side, "basis": self.basis}


@dataclass(frozen=True)
class Finding:
    """One claim-pair relation with its evidence."""

    relation: str
    tier: str
    a: ClaimRef
    b: ClaimRef
    score: float
    evidence: str
    staleness: Staleness = field(default_factory=Staleness)
    rationale: str = ""

    def key(self) -> tuple[str, int, int, str, int, int]:
        """Order-independent identity of the claim pair (used to suppress re-reporting)."""
        left = (self.a.doc_id, self.a.char_start, self.a.char_end)
        right = (self.b.doc_id, self.b.char_start, self.b.char_end)
        first, second = sorted([left, right])
        return (*first, *second)

    def doc_pair(self) -> tuple[str, str]:
        return tuple(sorted([self.a.doc_id, self.b.doc_id]))  # type: ignore[return-value]

    def payload(self) -> JsonObject:
        return {
            "relation": self.relation,
            "tier": self.tier,
            "score": round(float(self.score), 4),
            "evidence": self.evidence,
            "rationale": self.rationale,
            "staleness": self.staleness.payload(),
            "a": self.a.payload(),
            "b": self.b.payload(),
        }


@dataclass
class TierStats:
    """Per-tier accounting: what the tier looked at, what it found, and what it cost."""

    tier: str
    candidate_pairs: int = 0
    findings: int = 0
    seconds: float = 0.0
    extra: JsonObject = field(default_factory=dict)

    def payload(self) -> JsonObject:
        return {
            "tier": self.tier,
            "candidate_pairs": self.candidate_pairs,
            "findings": self.findings,
            "seconds": round(self.seconds, 3),
            **self.extra,
        }


@dataclass
class AuditResult:
    """Everything one `audit-corpus-conflicts` run produced."""

    effort: str
    corpus_root: str
    n_docs: int
    findings: list[Finding] = field(default_factory=list)
    tiers: list[TierStats] = field(default_factory=list)
    needles: JsonObject = field(default_factory=dict)
    tree_meta: JsonObject = field(default_factory=dict)
    params: JsonObject = field(default_factory=dict)

    def relation_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.relation] = counts.get(finding.relation, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> JsonObject:
        payload: dict[str, Any] = {
            "effort": self.effort,
            "corpus_root": self.corpus_root,
            "n_docs": self.n_docs,
            "n_findings": len(self.findings),
            "relations": self.relation_counts(),
            "tiers": [stat.payload() for stat in self.tiers],
            "params": dict(self.params),
        }
        if self.needles:
            payload["needles"] = dict(self.needles)
        if self.tree_meta:
            payload["tree"] = dict(self.tree_meta)
        return payload
