"""Tier 1 (`hash`): exact and normalization-equivalent duplicate documents.

One pass over the corpus, no model and no embeddings. Documents are grouped by content hash; each
group of size n yields n-1 findings chaining every later member to the group's first document, so
a group of five identical files reports four pairs rather than ten.

`raw` evidence means the files are byte-identical. `normalized` means they differ only in case,
whitespace, punctuation, apostrophe variant, or governance front matter -- the re-ingested-edition
case, which is exactly where the staleness fields decide which copy is current.
"""

from llb.conflicts.constants import REL_DUPLICATE, TIER_HASH
from llb.conflicts.corpus import CorpusDoc, whole_doc_span
from llb.conflicts.governance import compare_editions
from llb.conflicts.models import ClaimRef, Finding

EVIDENCE_RAW = "raw"
EVIDENCE_NORMALIZED = "normalized"


def _claim_ref(doc: CorpusDoc) -> ClaimRef:
    start, end = whole_doc_span(doc)
    return ClaimRef(
        doc_id=doc.doc_id,
        char_start=start,
        char_end=end,
        text=doc.text,
        governance=doc.governance,
    )


def _group_findings(group: list[CorpusDoc], evidence: str) -> list[Finding]:
    """Chain each later document in `group` to the first one."""
    anchor = group[0]
    findings: list[Finding] = []
    for other in group[1:]:
        a, b = _claim_ref(anchor), _claim_ref(other)
        findings.append(
            Finding(
                relation=REL_DUPLICATE,
                tier=TIER_HASH,
                a=a,
                b=b,
                score=1.0,
                evidence=evidence,
                staleness=compare_editions(a.governance, b.governance),
                rationale=(
                    "byte-identical document content"
                    if evidence == EVIDENCE_RAW
                    else "identical after Ukrainian normalization (case, whitespace, "
                    "punctuation, apostrophe variant, front matter)"
                ),
            )
        )
    return findings


def _grouped(docs: list[CorpusDoc], attribute: str) -> list[list[CorpusDoc]]:
    """Documents grouped by one hash attribute, groups of 2+ only, in corpus order."""
    buckets: dict[str, list[CorpusDoc]] = {}
    for doc in docs:
        buckets.setdefault(str(getattr(doc, attribute)), []).append(doc)
    return [group for group in buckets.values() if len(group) > 1]


def detect_hash_duplicates(docs: list[CorpusDoc]) -> tuple[list[Finding], set[tuple[str, str]]]:
    """Exact then normalization-equivalent duplicate documents.

    Returns the findings and the set of ALL document pairs the tier settled. Findings chain each
    group member to the group's first document, so a group of five reports four pairs rather than
    ten -- but duplication is transitive, so the settled set is the group's full closure. Without
    that distinction the later tiers would re-examine (and re-report) the pairs the chaining left
    implicit.
    """
    findings: list[Finding] = []
    settled: set[tuple[str, str]] = set()
    for group in _grouped(docs, "raw_sha"):
        findings.extend(_group_findings(group, EVIDENCE_RAW))
        settled |= _group_pairs(group)
    seen = {finding.key() for finding in findings}
    for group in _grouped(docs, "normalized_sha"):
        for finding in _group_findings(group, EVIDENCE_NORMALIZED):
            if finding.key() not in seen:
                seen.add(finding.key())
                findings.append(finding)
        settled |= _group_pairs(group)
    return findings, settled


def _group_pairs(group: list[CorpusDoc]) -> set[tuple[str, str]]:
    """Every unordered document pair within one duplicate group."""
    ids = sorted(doc.doc_id for doc in group)
    return {(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))}
