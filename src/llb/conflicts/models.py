"""Findings, claim references, and governance/staleness records.

A finding is always a PAIR of claim references plus one relation. Both sides carry exact source
offsets so a downstream resolution lane can act on a span, not on a whole document. `offsets_exact`
records whether the span was located verbatim in the source (model-quoted claims that could not be
matched back fall back to the enclosing chunk span and say so).
"""

from dataclasses import dataclass, field
from typing import Any

from llb.conflicts.constants import COVERAGE_FIELD, STAGE_INPUTS_FIELD
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
    claim_precision: JsonObject = field(default_factory=dict)
    tree_meta: JsonObject = field(default_factory=dict)
    params: JsonObject = field(default_factory=dict)
    # Whether this corpus could carry a dated supersession at all: the documents that record an
    # orderable governance field, the corpus's own document pairs `compare_editions` can order, and
    # the returned pairs it can order. Detection-side and policy-free, so it is recorded on every
    # run -- it is what tells a zero policy delta (a property of the KNOWLEDGE) apart from a pair
    # the candidate list never returned, apart from a corpus ingested without dates at all.
    governance_coverage: JsonObject = field(default_factory=dict)
    # What the stage attribution above was READ FROM, so a finished bundle can be re-read under a
    # changed rule without the store that run held (`stage_replay.py`). Empty only on a result that
    # never ran the corpus pass; the chunk half of it is absent below the semantic tier, which is
    # itself the `effort` reading.
    stage_inputs: JsonObject = field(default_factory=dict)
    # An opt-in TO REVIEW projection under a policy the operator named, computed ABOVE this layer
    # (`policy_projection.py`) and carried as plain data. Empty by default, and empty is the whole
    # point: the detector runs without a resolution policy, and the renderer reads this dict
    # without importing the resolution vocabulary that produced it.
    policy_projection: JsonObject = field(default_factory=dict)

    def rows(self) -> list["JsonObject"]:
        """The findings as `findings.jsonl` rows, in the order that file is written in.

        One implementation, because the sidecar's group ids, the projection's group ids, and the
        rows on disk must all be derived from the SAME ordering or they address different groups.
        """
        from llb.conflicts.census import finding_sort_key

        return [finding.payload() for finding in sorted(self.findings, key=finding_sort_key)]

    def relation_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.relation] = counts.get(finding.relation, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> JsonObject:
        # Imported here rather than at module scope: the census reads these models, so the
        # dependency runs one way and this file stays the leaf every conflicts module can import.
        from llb.conflicts.census import finding_census, relation_census
        from llb.conflicts.granularity import finding_granularity

        payload: dict[str, Any] = {
            "effort": self.effort,
            "corpus_root": self.corpus_root,
            "n_docs": self.n_docs,
            "n_findings": len(self.findings),
            "finding_census": finding_census(self.findings),
            # Both grouping rules, so a consumer reads the decision RANGE rather than the quoted
            # group count alone; `granularity.QUOTED_RULE` names which end the audit is built on.
            "group_granularity": finding_granularity(self.findings),
            "relations": self.relation_counts(),
            "relation_census": relation_census(self.findings),
            "tiers": [stat.payload() for stat in self.tiers],
            "params": dict(self.params),
        }
        if self.needles:
            payload["needles"] = dict(self.needles)
        if self.claim_precision:
            payload["claim_precision"] = dict(self.claim_precision)
        if self.tree_meta:
            payload["tree"] = dict(self.tree_meta)
        if self.governance_coverage:
            payload[COVERAGE_FIELD] = dict(self.governance_coverage)
        if self.stage_inputs:
            payload[STAGE_INPUTS_FIELD] = dict(self.stage_inputs)
        if self.policy_projection:
            payload["policy_projection"] = dict(self.policy_projection)
        return payload
