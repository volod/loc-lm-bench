"""Reversible corpus-conflict resolution policy and overlay."""

import json
import shutil

import pytest
from typer.testing import CliRunner

from llb.cli.app import app
from llb.conflicts.audit import AuditParams, run_audit
from llb.conflicts.constants import (
    CONFLICT_OVERLAY_FILE,
    EFFECT_REPORT_FILE,
    FINDINGS_FILE,
    RESOLUTION_PLAN_FILE,
    REVIEW_RECORDS_FILE,
)
from llb.conflicts.overlay import applied_overlay_path
from llb.conflicts.report import write_audit
from llb.conflicts.resolution_io import (
    create_resolution_artifacts,
    install_overlay,
    rollback_overlay,
)
from llb.conflicts.resolution_policy import (
    ACTION_DROP_DUPLICATE,
    ACTION_ESCALATE,
    ACTION_PREFER_NEWER,
    POLICY_CONSERVATIVE,
    POLICY_PREFER_NEWER,
    resolve_finding,
)
from llb.prep.corpus_governance import corpus_doc_fingerprints
from llb.rag.chunking.corpus import chunk_corpus
from llb.review.adapters.conflicts import ConflictResolutionAdapter
from llb.review.registry import open_review

from conflict_helpers import FIXTURE_CORPUS


def _audit_copy(tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURE_CORPUS, corpus)
    audit_dir = tmp_path / "audit"
    write_audit(audit_dir, run_audit(corpus, AuditParams(effort="hash")))
    return corpus, audit_dir


def _finding(relation="contradicts", newer_side=None):
    return {
        "relation": relation,
        "tier": "claim",
        "score": 0.9,
        "evidence": "test",
        "staleness": {"newer_side": newer_side, "basis": "effective_date"},
        "a": {
            "doc_id": "a.md",
            "char_start": 0,
            "char_end": 3,
            "text": "old",
            "offsets_exact": True,
        },
        "b": {
            "doc_id": "b.md",
            "char_start": 0,
            "char_end": 3,
            "text": "new",
            "offsets_exact": True,
        },
    }


def test_policy_drops_duplicates_and_escalates_undated_contradictions():
    duplicate = resolve_finding(_finding("duplicate"), POLICY_CONSERVATIVE)
    contradiction = resolve_finding(_finding(), POLICY_CONSERVATIVE)
    assert duplicate["action"] == ACTION_DROP_DUPLICATE
    assert duplicate["status"] == "accepted"
    assert contradiction["action"] == ACTION_ESCALATE
    assert contradiction["status"] == "review_required"


def test_semantic_duplicate_candidate_requires_review_before_suppression():
    candidate = _finding("duplicate")
    candidate["tier"] = "semantic"
    item = resolve_finding(candidate, POLICY_CONSERVATIVE)
    assert item["action"] == ACTION_ESCALATE
    assert item["status"] == "review_required"


def test_prefer_newer_suppresses_the_governance_ordered_older_side():
    item = resolve_finding(_finding("superseded_by", newer_side="b"), POLICY_PREFER_NEWER)
    assert item["action"] == ACTION_PREFER_NEWER
    assert item["target_side"] == "a"
    assert item["target_doc_id"] == "a.md"


def test_apply_changes_only_overlay_affected_fingerprints_and_rollback_restores(tmp_path):
    corpus, audit_dir = _audit_copy(tmp_path)
    before = corpus_doc_fingerprints(corpus)
    source_bytes = {path.name: path.read_bytes() for path in corpus.glob("*.md")}
    plan, overlay, _ = create_resolution_artifacts(
        audit_dir / FINDINGS_FILE,
        audit_dir,
        policy=POLICY_CONSERVATIVE,
        corpus_root=corpus,
    )
    path = install_overlay(corpus, overlay, plan)
    after = corpus_doc_fingerprints(corpus)
    affected = set(overlay["documents"])
    assert affected
    assert {doc_id for doc_id in before if before[doc_id] != after[doc_id]} == affected
    assert {path.name: path.read_bytes() for path in corpus.glob("*.md")} == source_bytes
    assert rollback_overlay(corpus) == path
    assert corpus_doc_fingerprints(corpus) == before


def test_overlay_suppresses_redundant_documents_without_editing_source(tmp_path):
    corpus, audit_dir = _audit_copy(tmp_path)
    baseline = chunk_corpus(corpus, "heading", 600, 0)
    plan, overlay, _ = create_resolution_artifacts(
        audit_dir / FINDINGS_FILE,
        audit_dir,
        policy=POLICY_CONSERVATIVE,
        corpus_root=corpus,
    )
    install_overlay(corpus, overlay, plan)
    resolved = chunk_corpus(corpus, "heading", 600, 0)
    suppressed = {
        doc_id
        for doc_id, directive in overlay["documents"].items()
        if directive["suppress_document"]
    }
    assert suppressed
    assert not suppressed & {str(chunk["doc_id"]) for chunk in resolved}
    assert len(resolved) < len(baseline)


def test_install_refuses_findings_stale_against_source_text(tmp_path):
    corpus, audit_dir = _audit_copy(tmp_path)
    plan, overlay, _ = create_resolution_artifacts(
        audit_dir / FINDINGS_FILE,
        audit_dir,
        policy=POLICY_CONSERVATIVE,
        corpus_root=corpus,
    )
    target = corpus / str(plan["items"][0]["a"]["doc_id"])
    target.write_text(target.read_text(encoding="utf-8") + "changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since audit"):
        install_overlay(corpus, overlay, plan)


def test_cli_writes_plan_overlay_effect_and_applies(tmp_path):
    corpus, audit_dir = _audit_copy(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "resolve-corpus-conflicts",
            "--findings",
            str(audit_dir / FINDINGS_FILE),
            "--corpus",
            str(corpus),
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    for name in (
        RESOLUTION_PLAN_FILE,
        CONFLICT_OVERLAY_FILE,
        REVIEW_RECORDS_FILE,
        EFFECT_REPORT_FILE,
    ):
        assert (audit_dir / name).is_file()
    assert applied_overlay_path(corpus).is_file()
    assert "MEASUREMENT REQUIRED" in (audit_dir / EFFECT_REPORT_FILE).read_text(encoding="utf-8")


def test_conflict_review_adapter_persists_a_typed_decision(tmp_path):
    path = tmp_path / REVIEW_RECORDS_FILE
    row = {
        "review_type": "corpus_conflict_resolution",
        "finding_id": "abc",
        "relation": "contradicts",
        "rationale": "needs review",
        "a": {"doc_id": "a.md", "text": "old"},
        "b": {"doc_id": "b.md", "text": "new"},
        "staleness": {},
        "resolution_decision": "",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    adapter = open_review(path)
    assert isinstance(adapter, ConflictResolutionAdapter)
    adapter.apply(0, "drop_a")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["resolution_decision"] == "drop_a"


def test_reviewed_decision_updates_plan_action_counts(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("old", encoding="utf-8")
    (corpus / "b.md").write_text("new", encoding="utf-8")
    findings = tmp_path / FINDINGS_FILE
    findings.write_text(json.dumps(_finding()) + "\n", encoding="utf-8")
    reviewed = tmp_path / REVIEW_RECORDS_FILE
    initial, _, _ = create_resolution_artifacts(
        findings,
        tmp_path / "initial",
        policy=POLICY_CONSERVATIVE,
        corpus_root=corpus,
    )
    reviewed.write_text(
        json.dumps(
            {
                "finding_id": initial["items"][0]["finding_id"],
                "resolution_decision": "drop_a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    resolved, _, _ = create_resolution_artifacts(
        findings,
        tmp_path / "resolved",
        policy=POLICY_CONSERVATIVE,
        corpus_root=corpus,
        reviewed=reviewed,
    )
    assert resolved["action_counts"] == {"drop_duplicate": 1}
