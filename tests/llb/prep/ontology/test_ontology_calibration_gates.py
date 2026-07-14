"""Tests for ontology calibration gates."""

import logging
from pathlib import Path
from llb.goldset.schema import GoldItem
from llb.prep.ontology.artifacts.report import write_calibration_artifacts
from llb.prep.ontology.constants import (
    PROVENANCE_KIND,
)
from llb.prep.ontology.models import (
    Claim,
    DocExtraction,
    DocRecord,
    Entity,
    OntologyCandidate,
)
from tests.llb.prep.ontology._ontology_fixtures import DOC1
from test_ontology_draft import _draft, _grounded_span, fake_endpoint


def test_calibration_gates_pass_without_sro_facts(tmp_path):
    # a corpus rich in entities/claims but with ZERO SRO facts still yields a usable gold set: the
    # roll-up passes on grounded extractions + a non-empty gold set, and does NOT require facts.
    out = tmp_path / "bundle"
    out.mkdir()
    span = _grounded_span("a.md", DOC1, "Київ")
    docs = [DocRecord(doc_id="a.md", text=DOC1, sha256="x", n_chars=len(DOC1))]
    extraction = DocExtraction(
        doc_id="a.md",
        entities=[Entity(name="Київ", type="LOC", mentions=[span])],
        claims=[Claim(text="Київ є столицею", evidence=span)],
    )
    item = GoldItem(
        id="q1",
        question="Що згадано у документі?",
        reference_answer="Київ",
        source_doc_id="a.md",
        source_spans=[span],
        provenance=PROVENANCE_KIND,
        split="final",
    )
    report = write_calibration_artifacts(
        out, docs, [extraction], OntologyCandidate(), [item], elapsed_s=0.0, settings={}
    )
    gates = report["gates"]
    assert gates["nonzero_grounded_facts"] is False  # no SRO facts...
    assert gates["nonzero_grounded_extractions"] is True  # ...but entities + claims are grounded
    assert gates["nonzero_draft_items"] is True
    assert gates["pdf_citation_gate_applicable"] is False
    assert gates["passed"] is True


def test_calibration_gates_fail_on_empty_draft(tmp_path):
    # no grounded evidence and no drafted items -> the roll-up fails so the operator does not accept
    out = tmp_path / "bundle"
    out.mkdir()
    docs = [DocRecord(doc_id="a.md", text=DOC1, sha256="x", n_chars=len(DOC1))]
    report = write_calibration_artifacts(
        out,
        docs,
        [DocExtraction(doc_id="a.md")],
        OntologyCandidate(),
        [],
        elapsed_s=0.0,
        settings={},
    )
    gates = report["gates"]
    assert gates["nonzero_grounded_extractions"] is False
    assert gates["nonzero_draft_items"] is False
    assert gates["passed"] is False


def test_pipeline_warns_and_names_the_blocking_gate(caplog):
    # the pipeline acts on the roll-up: a failing required gate is a WARNING that names it, so a
    # PDF run with no citation-valid needle is flagged (informational gates never appear here)
    from llb.prep.ontology.pipeline.bundle import _log_calibration_gates

    report = {
        "gates": {
            "nonzero_grounded_extractions": True,
            "nonzero_grounded_facts": False,  # informational: must NOT be named as a blocker
            "nonzero_draft_items": True,
            "has_citation_valid_needles": False,
            "pdf_citation_gate_applicable": True,
            "passed": False,
        }
    }
    with caplog.at_level(logging.WARNING):
        _log_calibration_gates(report, Path("/tmp/bundle"))
    assert "NOT passed" in caplog.text
    assert "has_citation_valid_needles" in caplog.text
    assert "nonzero_grounded_facts" not in caplog.text


def test_full_flow_does_not_write_when_write_false(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc1.md").write_text(DOC1, encoding="utf-8")
    out = tmp_path / "bundle"
    result = _draft(
        corpus,
        fake_endpoint,
        out_dir=out,
        write=False,
    )
    assert not out.exists() and len(result.items) >= 0
