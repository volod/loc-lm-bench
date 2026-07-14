"""Sequential local comparison, adaptive model selection, and report analytics."""

from pathlib import Path

from llb.inference.serving_selection import GpuTierInfo
from llb.prep.ontology.compare_analysis import comparison_statistics, format_comparison_statistics
from llb.prep.ontology.compare_gate import finalize_comparison
from llb.prep.ontology.endpoint_config import EndpointCompleters, EndpointConfig
from llb.goldset.verify_base import load_worksheet, write_worksheet_rows
from llb.prep.ontology.local_compare import compare_local_drafters
from llb.prep.ontology.local_compare_models import (
    LOCAL_COMPARE_PROFILES,
    select_local_compare_models,
)
from tests.llb.prep.ontology.test_ontology_draft import DOC1, DOC2, fake_endpoint


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc1.md").write_text(DOC1, encoding="utf-8")
    (corpus / "doc2.md").write_text(DOC2, encoding="utf-8")
    return corpus


def test_adaptive_model_profiles_cover_supported_gpu_tiers():
    for tier, profile in LOCAL_COMPARE_PROFILES.items():
        baseline, probe, num_ctx, selection = select_local_compare_models(
            "http://unused/v1",
            gpu=GpuTierInfo(tier, tier * 1024, f"gpu-{tier}", True),
            installed={profile.baseline_model, profile.probe_model},
        )
        assert (baseline, probe, num_ctx) == (
            profile.baseline_model,
            profile.probe_model,
            profile.num_ctx,
        )
        assert selection["policy"] == "gpu-tier-qwen-gemma"


def test_sequential_local_compare_unloads_between_models_and_writes_stats(tmp_path):
    unloads: list[list[str] | None] = []
    endpoint = {
        "kind": "local",
        "backend": "ollama",
        "base_url": "http://ollama/v1",
        "think": False,
    }
    baseline = EndpointConfig(model="qwen3:14b", **endpoint)
    probe = EndpointConfig(model="gemma4:e4b", **endpoint)

    report = compare_local_drafters(
        _corpus(tmp_path),
        baseline,
        probe,
        seeds=4,
        out_dir=tmp_path / "comparison",
        resource_selection={"policy": "test"},
        baseline_completers=EndpointCompleters.single(fake_endpoint),
        probe_complete=fake_endpoint,
        unload=lambda _url, models: unloads.append(models) or list(models or []),
    )

    assert unloads == [None, ["qwen3:14b"], None]
    assert report["lane_order"] == ["baseline", "probe"]
    assert report["execution"]["model_order"] == ["qwen3:14b", "gemma4:e4b"]
    assert report["execution"]["unload_between_lanes"] is True
    stats = comparison_statistics(report)
    assert stats["lanes"]["baseline"]["model"] == "qwen3:14b"
    assert stats["lanes"]["probe"]["model"] == "gemma4:e4b"
    rendered = format_comparison_statistics(stats)
    assert "sequential-local-ollama" in rendered
    assert "probe_minus_baseline_parse_rate" in rendered

    for lane in ("baseline", "probe"):
        worksheet = tmp_path / "comparison" / lane / "verify_sample.csv"
        rows, fields = load_worksheet(worksheet)
        for row in rows:
            row["decision"] = "accept"
        write_worksheet_rows(worksheet, rows, fields)
    finalized = finalize_comparison(tmp_path / "comparison" / "comparison.json")
    assert finalized["finalization"]["passed"] is True
    assert finalized["finalization"]["checks"]["model_unload_between_lanes"] is True


def test_adaptive_selection_reports_missing_model():
    profile = LOCAL_COMPARE_PROFILES[12]
    try:
        select_local_compare_models(
            "http://unused/v1",
            gpu=GpuTierInfo(12, 12288, "gpu", True),
            installed={profile.baseline_model},
        )
    except RuntimeError as exc:
        assert profile.probe_model in str(exc)
        assert "ollama pull" in str(exc)
    else:
        raise AssertionError("missing adaptive model must fail before the run")
