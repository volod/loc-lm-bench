"""Frontier ontology lane consent, budget abort, phase routing, and shared-seed comparison."""

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import llb.prep.ontology.endpoint as endpoint_runtime
from llb.cli import app
from llb.prep.frontier_telemetry import DraftBudgetExceeded
from llb.goldset.verify_base import load_worksheet, write_worksheet_rows
from llb.prep.ontology.compare import compare_drafters, refresh_comparison_acceptance
from llb.prep.ontology.compare_gate import finalize_comparison
from llb.prep.ontology.endpoint_config import (
    EndpointCompleters,
    EndpointConfig,
    EndpointPlan,
)
from llb.prep.ontology.pipeline.run import draft_goldset
from tests.llb.prep.ontology.test_ontology_draft import DOC1, DOC2, fake_endpoint


def test_committed_frontier_probe_corpus_matches_provenance():
    root = Path(__file__).parents[4] / "samples" / "text_analysis_bundle_uk"
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))

    assert provenance["frontier_ua_probe"] is True
    assert provenance["egress_classification"] == "synthetic-repo-fixture"
    assert provenance["n_docs"] == 2
    for document in provenance["documents"]:
        payload = (root / document["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == document["sha256"]


def _corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc1.md").write_text(DOC1, encoding="utf-8")
    (corpus / "doc2.md").write_text(DOC2, encoding="utf-8")
    return corpus


def test_frontier_cli_decline_stops_before_completer_build(tmp_path, monkeypatch):
    corpus = _corpus(tmp_path)
    built = []
    monkeypatch.setattr(
        endpoint_runtime,
        "litellm_complete",
        lambda *args, **kwargs: built.append(True) or fake_endpoint,
    )
    result = CliRunner().invoke(
        app,
        [
            "prepare-goldset-draft",
            "--corpus-root",
            str(corpus),
            "--model",
            "provider/model",
            "--endpoint",
            "frontier",
        ],
        input="n\n",
    )
    assert result.exit_code == 2
    assert str(corpus) in result.output and "provider/model" in result.output
    assert built == []


def test_frontier_spend_abort_writes_inspectable_provenance(tmp_path, monkeypatch):
    corpus = _corpus(tmp_path)
    out = tmp_path / "aborted"

    def fake_factory(model, temperature, log):
        def complete(prompt):
            log.record(model, 10, 5, 0.6, latency_s=0.25)
            return fake_endpoint(prompt)

        return complete

    monkeypatch.setattr(endpoint_runtime, "litellm_complete", fake_factory)
    frontier = EndpointConfig(
        kind="frontier",
        model="provider/model",
        egress_consent=True,
        max_usd=0.5,
        max_calls=20,
    )
    with pytest.raises(DraftBudgetExceeded, match="spend budget exceeded"):
        draft_goldset(corpus, EndpointPlan.single(frontier), out_dir=out, max_items=4)

    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["status"] == "aborted"
    assert provenance["abort"]["resumable"] is True
    assert provenance["endpoint"]["calls"] == 1
    assert provenance["endpoint"]["cost_usd"] == pytest.approx(0.6)
    assert provenance["endpoint"]["latency_s"] == pytest.approx(0.25)
    assert not (out / "goldset.jsonl").exists()


def test_split_phase_plan_uses_distinct_completers_and_records_routes(tmp_path):
    corpus = _corpus(tmp_path)
    out = tmp_path / "mixed"
    calls = {"extraction": 0, "drafting": 0}

    def extraction(prompt):
        calls["extraction"] += 1
        return fake_endpoint(prompt)

    def drafting(prompt):
        calls["drafting"] += 1
        return fake_endpoint(prompt)

    local = EndpointConfig(kind="local", model="local-model")
    frontier = EndpointConfig(
        kind="frontier", model="provider/model", egress_consent=True, max_calls=20
    )
    draft_goldset(
        corpus,
        EndpointPlan(extraction=local, drafting=frontier),
        completers=EndpointCompleters(extraction=extraction, drafting=drafting),
        out_dir=out,
        max_items=4,
    )
    assert calls["extraction"] == 2 and calls["drafting"] > 0
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    stages = provenance["endpoint"]["stages"]
    assert stages["extraction"]["kind"] == "local"
    assert stages["drafting"]["kind"] == "frontier"


def test_compare_drafters_uses_identical_seed_objects_and_writes_rankings(tmp_path):
    corpus = _corpus(tmp_path)
    out = tmp_path / "comparison"
    local = EndpointConfig(kind="local", model="local-model")
    frontier = EndpointConfig(
        kind="frontier", model="provider/model", egress_consent=True, max_calls=20
    )
    report = compare_drafters(
        corpus,
        local,
        frontier,
        seeds=4,
        out_dir=out,
        local_completers=EndpointCompleters.single(fake_endpoint),
        frontier_complete=fake_endpoint,
    )
    local_lane = report["lanes"]["local"]
    frontier_lane = report["lanes"]["frontier"]
    assert local_lane["seeds"] == frontier_lane["seeds"]
    assert len(report["shared_seed_fingerprints"]) == local_lane["seeds"]
    assert set(report["rankings"]["kept_yield"]) == {"local", "frontier"}
    assert (out / "comparison.json").is_file()
    assert (out / "local" / "verify_sample.csv").is_file()
    assert (out / "frontier" / "verify_sample.csv").is_file()

    for lane in ("local", "frontier"):
        worksheet = out / lane / "verify_sample.csv"
        rows, fields = load_worksheet(worksheet)
        for row in rows:
            row["decision"] = "accept"
        write_worksheet_rows(worksheet, rows, fields)
    refreshed = refresh_comparison_acceptance(
        out / "comparison.json",
        out / "local" / "verify_sample.csv",
        out / "frontier" / "verify_sample.csv",
    )
    assert refreshed["lanes"]["local"]["verify_sample"]["accept_rate"] == 1.0
    finalized = finalize_comparison(out / "comparison.json")
    assert finalized["finalization"]["passed"] is True
    frontier_progress = finalized["finalization"]["worksheet_progress"]["frontier"]
    assert frontier_progress["decided"] == frontier_progress["total"]
    assert frontier_progress["total"] == frontier_lane["kept"]


def test_finalize_comparison_reports_incomplete_human_review(tmp_path):
    corpus = _corpus(tmp_path)
    out = tmp_path / "comparison"
    local = EndpointConfig(kind="local", model="local-model")
    frontier = EndpointConfig(
        kind="frontier", model="provider/model", egress_consent=True, max_calls=20
    )
    compare_drafters(
        corpus,
        local,
        frontier,
        seeds=4,
        out_dir=out,
        local_completers=EndpointCompleters.single(fake_endpoint),
        frontier_complete=fake_endpoint,
    )

    finalized = finalize_comparison(out / "comparison.json")

    assert finalized["finalization"]["passed"] is False
    assert finalized["finalization"]["checks"]["worksheets_complete"] is False
