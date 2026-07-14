"""Two-stage Optuna tuner (Optuna tuning): pure search-space/fit logic + Optuna-driven studies."""

import pytest

from llb.core.config import RunConfig
from llb.core.contracts import ModelSpec
from llb.optimize.tuner import make_objective
from llb.optimize.tuning_space import (
    EXTENDED_STRATEGIES,
    STRATEGIES,
    estimate_prompt_tokens,
    fits_context,
    suggest_overrides,
)

VLLM_BASE = {"backend": "vllm", "vllm_host": "http://localhost:8000", "vllm_port": 8000}

SMALL_CTX_SPEC: ModelSpec = {
    "name": "m",
    "backend": "vllm",
    "source": "org/m",
    "params_b": 3.0,
    "quant": "q4_k_m",
    "n_layers": 28,
    "kv_dim": 1024,
    "max_context": 2048,
}


class FakeTrial:
    def __init__(self, vals):
        self.vals = vals
        self.attrs = {}

    def suggest_categorical(self, name, choices):
        return self.vals[name]

    def suggest_int(self, name, lo, hi, step=1):
        return self.vals[name]

    def suggest_float(self, name, lo, hi, step=None):
        return self.vals[name]

    def set_user_attr(self, key, value):
        self.attrs[key] = value


def test_suggest_overrides_flat_keeps_overlap_below_size():
    over = suggest_overrides(
        FakeTrial(
            {
                "strategy": "markdown",
                "chunk_size": 800,
                "overlap_frac": 0.25,
                "retrieval_mode": "flat",
                "top_k": 6,
            }
        )
    )
    assert over["strategy"] == "markdown" and over["chunk_size"] == 800
    assert over["chunk_overlap"] == 200 and over["chunk_overlap"] < over["chunk_size"]
    assert "child_chunk_size" not in over  # flat mode


def test_suggest_overrides_extended_chunkers_behind_the_flag():
    # Default search space excludes the corpus-chunking additions; `strategies=` opts them in.
    captured: dict[str, list] = {}

    class RecordingTrial(FakeTrial):
        def suggest_categorical(self, name, choices):
            captured.setdefault(name, list(choices))
            return super().suggest_categorical(name, choices)

    vals = {
        "strategy": "late",
        "chunk_size": 512,
        "overlap_frac": 0.1,
        "retrieval_mode": "flat",
        "top_k": 5,
    }
    over = suggest_overrides(RecordingTrial(vals), strategies=EXTENDED_STRATEGIES)
    assert {"page", "heading", "late"} <= set(captured["strategy"])
    assert over["strategy"] == "late"

    captured.clear()
    suggest_overrides(RecordingTrial({**vals, "strategy": "markdown"}))
    assert captured["strategy"] == STRATEGIES  # default space unchanged


def test_suggest_overrides_parent_child_clamps_child_below_size():
    over = suggest_overrides(
        FakeTrial(
            {
                "strategy": "recursive",
                "chunk_size": 300,
                "overlap_frac": 0.0,
                "retrieval_mode": "parent_child",
                "top_k": 4,
                "child_chunk_size": 600,  # bigger than chunk_size -> must be clamped
            }
        )
    )
    assert over["child_chunk_size"] < over["chunk_size"]


def test_suggest_overrides_hybrid_samples_fusion_knobs():
    over = suggest_overrides(
        FakeTrial(
            {
                "strategy": "recursive",
                "chunk_size": 512,
                "overlap_frac": 0.1,
                "retrieval_mode": "hybrid",
                "top_k": 5,
                "fusion_weight": 0.4,
                "fusion_candidates": 40,
            }
        )
    )
    assert over["retrieval_mode"] == "hybrid"
    assert over["fusion_weight"] == 0.4 and over["fusion_candidates"] == 40


def test_suggest_overrides_flat_never_samples_fusion_knobs():
    over = suggest_overrides(
        FakeTrial(
            {
                "strategy": "recursive",
                "chunk_size": 512,
                "overlap_frac": 0.1,
                "retrieval_mode": "flat",
                "top_k": 5,
            }
        )
    )
    assert "fusion_weight" not in over and "fusion_candidates" not in over


def test_suggest_overrides_rerank_axes_only_behind_the_flag():
    # rerank-context-order: no `--reranker` -> the axes are never sampled; with it, the on/off
    # categorical gates the candidate-depth axis (off-trial samples no dead depth parameter).
    vals = {
        "strategy": "recursive",
        "chunk_size": 512,
        "overlap_frac": 0.1,
        "retrieval_mode": "flat",
        "top_k": 5,
    }
    assert "reranker" not in suggest_overrides(FakeTrial(vals))

    off = suggest_overrides(
        FakeTrial({**vals, "use_reranker": False}), reranker="BAAI/bge-reranker-v2-m3"
    )
    assert "reranker" not in off and "rerank_candidates" not in off

    on = suggest_overrides(
        FakeTrial({**vals, "use_reranker": True, "rerank_candidates": 30}),
        reranker="BAAI/bge-reranker-v2-m3",
    )
    assert on["reranker"] == "BAAI/bge-reranker-v2-m3" and on["rerank_candidates"] == 30


def test_estimate_prompt_tokens_grows_with_topk_and_size():
    base = RunConfig(max_tokens=128)
    big = base.with_overrides(top_k=12, chunk_size=1200)
    small = base.with_overrides(top_k=3, chunk_size=256)
    assert estimate_prompt_tokens(big) > estimate_prompt_tokens(small)


def test_fits_context_prunes_when_retrieved_context_too_big():
    base = RunConfig(max_tokens=128)
    big = base.with_overrides(top_k=12, chunk_size=1200)  # ~4800+ tok > 2048
    small = base.with_overrides(top_k=3, chunk_size=256)  # well under 2048
    assert fits_context(big, SMALL_CTX_SPEC, 0, 0) is False
    assert fits_context(small, SMALL_CTX_SPEC, 0, 0) is True
    assert fits_context(big, None, 0, 0) is True  # no spec -> cannot judge -> not pruned


def test_objective_prunes_over_context_trial(tmp_path):
    optuna = pytest.importorskip("optuna")
    base = RunConfig(max_tokens=128, data_dir=tmp_path)
    objective = make_objective(base, lambda _c: 1.0, model_spec=SMALL_CTX_SPEC)
    trial = optuna.trial.FixedTrial(
        {
            "strategy": "recursive",
            "chunk_size": 1280,
            "overlap_frac": 0.1,
            "retrieval_mode": "flat",
            "top_k": 12,
        }
    )
    with pytest.raises(optuna.TrialPruned):
        objective(trial)


# --- Optuna tuning backend-aware Optuna: serving params, measured OOM prune, throughput tie-break ----

BASE_OVERRIDES = {
    "strategy": "markdown",
    "chunk_size": 800,
    "overlap_frac": 0.1,
    "retrieval_mode": "flat",
    "top_k": 6,
    "gpu_memory_utilization": 0.8,
    "max_model_len": 8192,
}


# --- isolation reclaim reuse: each trial runs through the executor's isolate_cell -----------------------

_GPU = [{"index": 0, "temp_c": 40, "power_w": 100.0, "sm_clock_mhz": 2000, "mem_clock_mhz": 9000}]
