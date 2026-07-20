"""End-to-end tier orchestration over the fixture corpus, plus the report artifacts.

Runs the real pipeline against a StoreView of real chunk records and fake BoW vectors, so the
whole path is exercised without FAISS or an encoder. Each planted pair is asserted at the tier
that should find it, and the cumulative-effort contract is asserted directly: a cheaper effort
must never report something a more expensive one misses.
"""

import json

import pytest

from llb.conflicts.audit import AuditParams, run_audit
from llb.conflicts.constants import (
    REL_DUPLICATE,
    REL_SUBSUMED_BY,
    REL_SUPERSEDED_BY,
    TIER_CLAIM,
    TIER_HASH,
    TIER_LEXICAL,
    TIER_SEMANTIC,
    tiers_up_to,
)
from llb.conflicts.report import render_report, write_audit

from conflict_helpers import (
    DOC_2021,
    DOC_2021_COPY,
    DOC_2024,
    DOC_ARCHIVE,
    DOC_DEADLINE,
    DOC_EAPPEALS,
    FAKE_COS_THRESHOLD,
    FIXTURE_CORPUS,
    fake_store_view,
    relation_for,
)

CONTRADICTS = (
    '{"relation": "contradicts", "confidence": 0.95,'
    ' "claim_a": "становить тридцять календарних днів",'
    ' "claim_b": "становить п\'ятнадцять робочих днів", "rationale": "different deadlines"}'
)
SUBSUMED = (
    '{"relation": "subsumed_by", "confidence": 0.88,'
    ' "claim_a": "становить установлений законодавством строк",'
    ' "claim_b": "становить п\'ятнадцять робочих днів", "rationale": "vague vs specific"}'
)
DUPLICATE = (
    '{"relation": "duplicate", "confidence": 0.99, "claim_a": "", "claim_b": "",'
    ' "rationale": "restated unchanged"}'
)


def scripted(prompt: str) -> str:
    """Adjudicate by what the passages actually say, so order never matters."""
    if "тридцять" in prompt and "п'ятнадцять" in prompt:
        return CONTRADICTS
    if "установлений законодавством" in prompt:
        return SUBSUMED
    return DUPLICATE


def audit(effort: str, **kwargs):
    return run_audit(
        FIXTURE_CORPUS,
        AuditParams(effort=effort, cos_threshold=FAKE_COS_THRESHOLD, **kwargs),
        store=fake_store_view(),
        complete=scripted,
    )


def test_hash_effort_finds_only_the_identical_copies():
    result = audit(TIER_HASH)
    assert [stat.tier for stat in result.tiers] == [TIER_HASH]
    assert relation_for(result.findings, DOC_2021, DOC_2021_COPY) == {REL_DUPLICATE}
    assert not relation_for(result.findings, DOC_EAPPEALS, DOC_2024)


def test_lexical_effort_adds_the_subsumption():
    result = audit(TIER_LEXICAL)
    assert [stat.tier for stat in result.tiers] == [TIER_HASH, TIER_LEXICAL]
    assert relation_for(result.findings, DOC_EAPPEALS, DOC_2024) == {REL_SUBSUMED_BY}


def test_semantic_effort_pairs_the_revision_but_does_not_label_the_disagreement():
    """Cosine says "same topic", not "these agree"; the honest label waits for the claim tier."""
    result = audit(TIER_SEMANTIC)
    assert relation_for(result.findings, DOC_2021, DOC_2024) == {REL_DUPLICATE}
    semantic = next(stat for stat in result.tiers if stat.tier == TIER_SEMANTIC)
    assert semantic.extra["cross_document_pairs"] > 0


def test_claim_effort_recovers_partial_supersession():
    """The 2024 revision supersedes the deadline it changed and duplicates what it restated."""
    result = audit(TIER_CLAIM)
    relations = relation_for(result.findings, DOC_2021, DOC_2024)
    assert REL_SUPERSEDED_BY in relations, "the changed deadline must read as superseded"
    assert REL_DUPLICATE in relations, "the restated sections must read as duplicates"

    superseded = next(
        f
        for f in result.findings
        if f.relation == REL_SUPERSEDED_BY and f.doc_pair() == tuple(sorted([DOC_2021, DOC_2024]))
    )
    assert superseded.a.doc_id == DOC_2021, "the deprecated claim is side a"
    assert superseded.staleness.basis == "effective_date"


def test_claim_effort_labels_the_vague_restatement_as_subsumed():
    result = audit(TIER_CLAIM)
    assert REL_SUBSUMED_BY in relation_for(result.findings, DOC_DEADLINE, DOC_2024)


def test_unrelated_document_is_never_reported_at_any_effort():
    for effort in (TIER_HASH, TIER_LEXICAL, TIER_SEMANTIC, TIER_CLAIM):
        result = audit(effort)
        assert all(DOC_ARCHIVE not in f.doc_pair() for f in result.findings), effort


def test_metadata_and_stub_chunks_are_excluded_from_semantic_pairing():
    """Front matter and page-marker chunks are metadata, not claims, and must not pair."""
    result = audit(TIER_SEMANTIC)
    semantic = next(stat for stat in result.tiers if stat.tier == TIER_SEMANTIC)
    assert semantic.extra["excluded_chunks"] > 0
    assert semantic.extra["excluded_metadata_block_chunks"] == 2
    assert semantic.extra["excluded_front_matter_chunks"] > 0


def test_effort_is_cumulative():
    """Each tier's findings survive into every more expensive effort."""
    seen: set[tuple] = set()
    for effort in (TIER_HASH, TIER_LEXICAL, TIER_SEMANTIC):
        keys = {f.key() for f in audit(effort).findings}
        assert seen <= keys, f"{effort} dropped a finding a cheaper effort reported"
        seen = keys


def test_tiers_up_to_is_ordered():
    assert tiers_up_to(TIER_SEMANTIC) == (TIER_HASH, TIER_LEXICAL, TIER_SEMANTIC)
    with pytest.raises(ValueError, match="unknown effort tier"):
        tiers_up_to("free")


def test_audit_is_deterministic():
    first, second = audit(TIER_CLAIM), audit(TIER_CLAIM)
    assert [f.payload() for f in first.findings] == [f.payload() for f in second.findings]


def test_semantic_effort_requires_a_store():
    with pytest.raises(SystemExit, match="needs a built store"):
        run_audit(FIXTURE_CORPUS, AuditParams(effort=TIER_SEMANTIC))


def test_claim_effort_requires_an_endpoint():
    with pytest.raises(SystemExit, match="needs a model endpoint"):
        run_audit(FIXTURE_CORPUS, AuditParams(effort=TIER_CLAIM), store=fake_store_view())


def test_capped_claim_run_keeps_unadjudicated_pairs_as_provisional():
    """A budget cap must narrow what the model sees, never silently drop a detected pair."""
    full = audit(TIER_CLAIM)
    capped = audit(TIER_CLAIM, max_claim_pairs=1)
    assert len(capped.findings) == len(full.findings)
    assert {f.tier for f in capped.findings} >= {TIER_SEMANTIC, TIER_CLAIM}


def test_every_finding_carries_exact_offsets_into_its_source():
    result = audit(TIER_CLAIM)
    for finding in result.findings:
        for side in (finding.a, finding.b):
            source = (FIXTURE_CORPUS / side.doc_id).read_text(encoding="utf-8")
            assert 0 <= side.char_start < side.char_end <= len(source)
            if side.offsets_exact:
                assert source[side.char_start : side.char_end] == side.text


def test_missing_corpus_fails_with_a_clear_message(tmp_path):
    with pytest.raises(SystemExit, match="corpus root does not exist"):
        run_audit(tmp_path / "nope", AuditParams(effort=TIER_HASH))


def test_write_audit_persists_findings_report_and_summary(tmp_path):
    result = audit(TIER_CLAIM)
    paths = write_audit(tmp_path / "run", result)
    rows = [
        json.loads(line)
        for line in paths["findings"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == len(result.findings)
    assert {"relation", "tier", "a", "b", "staleness"} <= set(rows[0])
    assert rows[0]["a"]["doc_id"]

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["n_findings"] == len(result.findings)
    assert summary["effort"] == TIER_CLAIM
    assert paths["tree_meta"].is_file()


def test_report_leads_with_actionable_relations():
    result = audit(TIER_CLAIM)
    report = render_report(result)
    assert "# Corpus conflict audit" in report
    body = report.split("## Findings", 1)[1]
    first_row = next(line for line in body.splitlines() if line.startswith("| `"))
    assert REL_SUPERSEDED_BY in first_row or "contradicts" in first_row


def test_report_renders_for_a_clean_corpus(tmp_path):
    (tmp_path / "only.md").write_text("Єдиний документ без жодних конфліктів.", encoding="utf-8")
    result = run_audit(tmp_path, AuditParams(effort=TIER_LEXICAL))
    assert result.findings == []
    assert "No conflicting" in render_report(result)


def test_centering_is_skipped_on_a_corpus_too_small_to_estimate_a_mean():
    """The fixture has 23 chunks; a "mean direction" from that is noise, not a correction."""
    result = audit(TIER_SEMANTIC)
    assert result.tree_meta["centered"] is False
    assert result.params["center_vectors"] is True, (
        "the request was honored-if-possible, not ignored"
    )


def test_low_content_chunks_never_pair():
    """A page marker matching another page marker is a conversion artifact, not a conflict."""
    from llb.conflicts.semantic_tier import claim_token_count, content_ordinals

    chunks = [
        {
            "doc_id": "a.md",
            "char_start": 0,
            "char_end": 60,
            "text": "<!-- source_pdf: a.pdf page: 3 parser: pymupdf4llm -->\n3",
        },
        {
            "doc_id": "b.md",
            "char_start": 0,
            "char_end": 60,
            "text": "<!-- source_pdf: b.pdf page: 7 parser: pymupdf4llm -->\n7",
        },
    ]
    assert claim_token_count(chunks[0]["text"]) < 25
    assert content_ordinals(chunks, {}) == set()
