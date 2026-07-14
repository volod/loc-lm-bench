"""Recommendation summary: host-adaptive picks from final-split run bundles."""

import json

import pytest

from llb.board.recommend.build import (
    build_recommendation,
    load_run_summaries,
)
from llb.board.recommend.model import HostInfo, RunSummary
from llb.board.recommend.render import (
    format_summary_md,
    recommendation_payload,
)
from llb.board.runs import RunRecord
from llb.scoring.leaderboard import ModelResult

MAMAYLM_V2_27B = "mamaylm-v2-27b"
MAMAYLM_V2_12B = "mamaylm-v2-12b"


def _summary(model, obj, tok_s, vram, qpw, *, reliability=1.0, n=82, backend="ollama"):
    result = ModelResult(
        model=model,
        backend=backend,
        objective_score=obj,
        n_cases=n,
        reliability=reliability,
        tokens_per_s=tok_s,
        peak_vram_mb=vram,
        case_objectives=[obj] * n,
    )
    config = {
        "strategy": "recursive",
        "chunk_size": 800,
        "chunk_overlap": 120,
        "top_k": 5,
        "retrieval_mode": "flat",
        "model": model,
        "backend": backend,
    }
    record = RunRecord(
        result=result, config=config, run_dir=f"/runs/{model}", created_at="", split="final"
    )
    return RunSummary(record, quality_per_watt=qpw, mean_power_w=100.0, recall_at_k=0.95, mrr=0.83)


# A 5-model cohort mirroring the real 16 GiB committed-goldset sweep.
COHORT = [
    _summary(MAMAYLM_V2_27B, 0.546, 8.0, 16170, 0.053),
    _summary("lapa", 0.505, 29.2, 9454, 0.115),
    _summary(MAMAYLM_V2_12B, 0.500, 30.1, 9342, 0.115),
    _summary("qwen3.6", 0.471, 24.7, 15505, 0.216, reliability=0.951),
    _summary("mistral", 0.399, 12.9, 15907, 0.048),
]


def test_recommendation_picks_quality_efficiency_and_speed():
    rec = build_recommendation(COHORT, HostInfo(16, 16380, "RTX 4060 Ti", True))
    assert rec.best_quality.model == MAMAYLM_V2_27B  # highest objective
    assert rec.best_efficiency.model == "qwen3.6"  # highest quality/W
    assert rec.fastest.model == MAMAYLM_V2_12B  # highest tok/s
    assert rec.recall_at_k == 0.95 and rec.top_k == 5


def test_recommended_for_host_is_vram_adaptive():
    # 16 GiB: the 27B (16170 MiB) blows the 0.92 budget -> recommend the best that fits (lapa).
    at_16 = build_recommendation(COHORT, HostInfo(16, 16380, "g", True))
    assert at_16.recommended_for_host.model == "lapa"
    # 24 GiB budget: the 27B now fits with headroom and wins on accuracy.
    at_24 = build_recommendation(COHORT, HostInfo(24, 24 * 1024, "g", True))
    assert at_24.recommended_for_host.model == MAMAYLM_V2_27B
    # unknown VRAM (total 0) cannot filter -> falls back to the top-accuracy Pareto model.
    at_unknown = build_recommendation(COHORT, HostInfo(16, 0, "", False))
    assert at_unknown.recommended_for_host.model == MAMAYLM_V2_27B


def test_recommended_for_host_respects_performance_floor():
    # Both fit VRAM and are Pareto-optimal (one wins quality, one wins speed). Quality optimization
    # alone picks the accurate-but-slow model; the good-enough-performance floor flips it.
    cohort = [
        _summary("accurate-slow", 0.60, 5.0, 9000, 0.05),
        _summary("fast-enough", 0.50, 30.0, 9000, 0.20),
    ]
    host = HostInfo(16, 16380, "g", True)
    assert build_recommendation(cohort, host).recommended_for_host.model == "accurate-slow"

    rec = build_recommendation(cohort, host, min_tokens_per_s=15.0)
    assert rec.recommended_for_host.model == "fast-enough"
    md = format_summary_md(rec)
    assert "clears the 15 tok/s performance floor" in md
    assert "below the 15 tok/s floor (traded away for speed): accurate-slow" in md


def test_performance_floor_off_by_default_leaves_summary_unchanged():
    md = format_summary_md(build_recommendation(COHORT, HostInfo(16, 16380, "g", True)))
    assert "performance floor" not in md
    assert "traded away for speed" not in md


def test_format_summary_md_has_sections_and_picks():
    md = format_summary_md(build_recommendation(COHORT, HostInfo(16, 16380, "RTX 4060 Ti", True)))
    assert "# loc-lm-bench recommendation summary" in md
    assert "Recommended for this host: **lapa**" in md
    assert f"Best RAG accuracy: **{MAMAYLM_V2_27B}**" in md
    assert "recall@5" in md and "chunk_size=800" in md
    assert "| model | backend | objective |" in md  # comparison table
    assert "(final split, 82 cases)" in md  # uniform cohort -> single count
    assert "Excluded (off-cohort" not in md  # nothing excluded when all runs share the cohort
    assert "best RAG top_k 5" in md  # host pick surfaces its retrieval depth (RAG-grid use case)


def test_recommendation_payload_keeps_exact_model_ids_and_metrics():
    rec = build_recommendation(COHORT, HostInfo(16, 16380, "RTX 4060 Ti", True))
    payload = recommendation_payload(rec)

    recommended = payload["selection"]["recommended_for_host"]
    assert recommended["model"] == "lapa"
    assert recommended["tokens_per_s"] == 29.2
    assert payload["selection"]["best_quality"]["model"] == MAMAYLM_V2_27B
    assert payload["candidates"][0]["model"] == MAMAYLM_V2_27B
    assert payload["candidates"][0]["top_k"] == 5


def test_recommendation_ranks_only_dominant_cohort():
    # 5 models at n=82 + a single n=20 platform-matrix row: rank only the n=82 cohort and name the
    # rest as excluded, so the n=20 row cannot win a pick (e.g. "fastest") on a smaller sample.
    mixed = [*COHORT, _summary("gemma-e4b", 0.42, 60.1, 11227, 0.21, n=20)]
    rec = build_recommendation(mixed, HostInfo(16, 16380, "g", True))
    assert [s.result.n_cases for s in rec.summaries] == [82] * 5  # only the cohort is ranked
    assert {s.model for s in rec.excluded} == {"gemma-e4b"}
    assert rec.fastest.model == MAMAYLM_V2_12B  # 30.1 tok/s, not the excluded 60.1 row
    md = format_summary_md(rec)
    assert "(final split, 82 cases)" in md
    assert "Excluded (off-cohort, not ranked): gemma-e4b n=20" in md
    assert "--min-cases" in md


def test_select_cohort_breaks_ties_on_larger_n_cases():
    from llb.board.recommend.build import select_cohort

    # Two cohorts of equal model count (2 each): the larger-n cohort is the more robust comparison.
    summaries = [
        _summary("a50", 0.5, 10, 9000, 0.1, n=50),
        _summary("b50", 0.4, 10, 9000, 0.1, n=50),
        _summary("c82", 0.5, 10, 9000, 0.1, n=82),
        _summary("d82", 0.4, 10, 9000, 0.1, n=82),
    ]
    cohort, excluded = select_cohort(summaries)
    assert {s.result.n_cases for s in cohort} == {82}
    assert {s.model for s in excluded} == {"a50", "b50"}


def test_build_recommendation_requires_runs():
    with pytest.raises(ValueError, match="no final-split"):
        build_recommendation([], HostInfo(16, 16380, "g", True))


def test_load_run_summaries_filters_partial_before_dedup(tmp_path):
    # A 3-case smoke run scores higher than the full 82-case run of the SAME model; the min_cases
    # filter must drop it BEFORE best-per-model, so the full run represents the model (not the smoke).
    def write_bundle(name, model, n, obj):
        d = tmp_path / name
        d.mkdir()
        (d / "manifest.json").write_text(
            json.dumps(
                {
                    "split": "final",
                    "n_cases": n,
                    "config": {"model": model, "backend": "ollama", "top_k": 5},
                    "metrics": {
                        "objective_score": obj,
                        "reliability": 1.0,
                        "tokens_per_s": 10.0,
                        "quality_per_watt": 0.1,
                    },
                    "telemetry": {"peak_vram_mb": 9000},
                    "retrieval": {"recall": 0.9, "mrr": 0.8},
                }
            ),
            encoding="utf-8",
        )

    write_bundle("smoke", "modelA", 3, 0.99)
    write_bundle("full", "modelA", 82, 0.50)

    kept = load_run_summaries(tmp_path, min_cases=50)
    assert len(kept) == 1
    assert kept[0].result.n_cases == 82 and kept[0].result.objective_score == 0.50


def _cell(model, top_k, obj, tok_s=10.0, vram=9000, recall=0.9, n=82):
    result = ModelResult(
        model=model,
        backend="ollama",
        objective_score=obj,
        n_cases=n,
        reliability=1.0,
        tokens_per_s=tok_s,
        peak_vram_mb=vram,
        case_objectives=[obj] * n,
    )
    config = {
        "strategy": "recursive",
        "chunk_size": 800,
        "chunk_overlap": 120,
        "top_k": top_k,
        "retrieval_mode": "flat",
        "model": model,
        "backend": "ollama",
    }
    record = RunRecord(
        result=result, config=config, run_dir=f"/runs/{model}-{top_k}", created_at="", split="final"
    )
    return RunSummary(
        record, quality_per_watt=0.1, mean_power_w=100.0, recall_at_k=recall, mrr=0.83
    )
