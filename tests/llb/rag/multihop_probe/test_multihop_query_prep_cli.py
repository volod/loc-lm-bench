"""CLI wiring for the paired per-hop query-prep probe."""

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from llb.main import app


def test_probe_cli_builds_query_prep_and_writes_the_paired_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = SimpleNamespace(
        query_prep=["decompose"],
        data_dir=tmp_path,
        retrieval_backend="faiss",
        fingerprint=lambda: {"query_prep": ["decompose"], "model": "model"},
    )
    store = object()
    pipeline = object()
    observed: dict[str, object] = {}
    report = {
        "conversion": {
            "cohorts": {
                "query": {"all_spans_gained": 2, "n": 8},
                "budget": {"span_coverage_regressed": 1, "n": 19},
            }
        }
    }

    monkeypatch.setattr("llb.cli.rag.multihop_probe.load_config", lambda *_a, **_kw: cfg)
    monkeypatch.setattr(
        "llb.cli.rag.multihop_probe.fusion_inputs.evidence_items",
        lambda *_a, **_kw: [SimpleNamespace(question_type="multi-hop")],
    )
    monkeypatch.setattr("llb.executor.runner_retrieval._load_store", lambda _cfg: store)
    monkeypatch.setattr(
        "llb.cli.rag.multihop_probe.resolve_query_prep_endpoint",
        lambda *_a, **_kw: (cfg, None, {"model": "model", "backend": "ollama"}),
    )
    monkeypatch.setattr(
        "llb.executor.runner_retrieval.build_query_prep", lambda *_a, **_kw: pipeline
    )

    def compare(got_store, got_items, got_pipeline, **kwargs):
        observed.update(
            store=got_store,
            items=got_items,
            pipeline=got_pipeline,
            kwargs=kwargs,
        )
        return report

    monkeypatch.setattr("llb.rag.multihop_probe.compare_multihop_query_prep", compare)
    monkeypatch.setattr(
        "llb.rag.multihop_probe.format_query_prep_probe_report", lambda _report: "paired\n"
    )
    out_dir = tmp_path / "probe"
    result = CliRunner().invoke(
        app,
        [
            "probe-multihop-hops",
            "--goldset",
            str(tmp_path / "goldset.jsonl"),
            "--query-prep",
            "decompose",
            "--query-prep-model",
            "model",
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (observed["store"], observed["pipeline"]) == (store, pipeline)
    assert observed["kwargs"]["budgets"] == (10, 25, 50)
    assert "query conversion 2/8; budget cost 1/19" in result.output
    artifact = json.loads((out_dir / "probe.json").read_text(encoding="utf-8"))
    assert artifact["endpoint"] == {"model": "model", "backend": "ollama"}
    assert (out_dir / "report.md").read_text(encoding="utf-8") == "paired\n"


def test_probe_cli_refuses_empty_focus_before_loading_the_store(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = SimpleNamespace(query_prep=[])
    monkeypatch.setattr("llb.cli.rag.multihop_probe.load_config", lambda *_a, **_kw: cfg)
    monkeypatch.setattr(
        "llb.cli.rag.multihop_probe.fusion_inputs.evidence_items",
        lambda *_a, **_kw: [SimpleNamespace(question_type="factoid")],
    )

    result = CliRunner().invoke(
        app,
        [
            "probe-multihop-hops",
            "--goldset",
            str(tmp_path / "goldset.jsonl"),
        ],
    )

    assert result.exit_code == 2
    assert "probe focus slice is empty: multi-hop" in result.output
