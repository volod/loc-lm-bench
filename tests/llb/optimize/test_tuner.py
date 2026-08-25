"""Two-stage Optuna tuner (Optuna tuning): pure search-space/fit logic + Optuna-driven studies."""

import pytest

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec
from llb.backends.served_window import (
    BUDGET_SOURCE_DECLARED,
    BUDGET_SOURCE_SERVED,
    BUDGET_SOURCE_UNBOUNDED,
)
from llb.optimize.tuner import make_objective
from llb.optimize.tuner_runtime import resolve_study_window
from llb.backends.context_fit import bound_max_context
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

# The shape the defect lives in: a model card advertising a huge window on a backend serving 4096.
BIG_CTX_SPEC: ModelSpec = {**SMALL_CTX_SPEC, "max_context": 131072}


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


def test_suggest_overrides_fused_backend_samples_graph_weight():
    over = suggest_overrides(
        FakeTrial(
            {
                "strategy": "recursive",
                "chunk_size": 512,
                "overlap_frac": 0.1,
                "retrieval_mode": "flat",
                "top_k": 5,
                "graph_weight": 0.3,
            }
        ),
        retrieval_backend="fused",
    )
    assert over["graph_weight"] == 0.3


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


# --- the served window the prune is priced against ------------------------------------------


def test_fits_context_prunes_on_the_served_window_the_declared_one_admits():
    """The whole point: a probed 4096 must prune a trial a declared 131072 would keep.

    An unpinned Ollama serves `num_ctx` 4096 however large a window the model card advertises, so
    the declared side alone keeps a configuration whose prompt the backend silently truncates.
    """
    base = RunConfig(max_tokens=128)
    big = base.with_overrides(top_k=12, chunk_size=1280)  # 15,360 chars ~ 5,760 tok
    assert fits_context(big, BIG_CTX_SPEC, 0, 0) is True
    assert fits_context(big, BIG_CTX_SPEC, 0, 0, 4096) is False


def test_the_declared_window_still_prunes_when_the_backend_serves_a_larger_one():
    """The mirror direction: a generous backend must not widen what the model card declares."""
    base = RunConfig(max_tokens=128)
    big = base.with_overrides(top_k=12, chunk_size=1200)
    assert fits_context(big, SMALL_CTX_SPEC, 0, 0, 131072) is False
    assert fits_context(big.with_overrides(top_k=3, chunk_size=256), SMALL_CTX_SPEC, 0, 0, 131072)


def test_a_served_window_bounds_a_model_the_roster_does_not_price():
    """A probe IS a measurement, so it bounds the prompt whether or not the manifest lists it."""
    big = RunConfig(max_tokens=128).with_overrides(top_k=12, chunk_size=1280)
    assert fits_context(big, None, 0, 0) is True  # no spec, no probe -> cannot judge
    assert fits_context(big, None, 0, 0, 4096) is False


def test_bound_max_context_names_which_side_bound_the_prune():
    base = RunConfig(max_tokens=128)
    assert bound_max_context(base, BIG_CTX_SPEC, 0, 0, 4096) == (4096, BUDGET_SOURCE_SERVED)
    assert bound_max_context(base, SMALL_CTX_SPEC, 0, 0, 131072) == (2048, BUDGET_SOURCE_DECLARED)
    assert bound_max_context(base, None, 0, 0, None) == (0, BUDGET_SOURCE_UNBOUNDED)


def test_objective_prunes_on_the_served_window_and_names_it(tmp_path):
    optuna = pytest.importorskip("optuna")
    base = RunConfig(max_tokens=128, data_dir=tmp_path)
    sampled = {
        "strategy": "recursive",
        "chunk_size": 1280,
        "overlap_frac": 0.1,
        "retrieval_mode": "flat",
        "top_k": 12,
    }
    kept = make_objective(base, lambda _c: 1.0, model_spec=BIG_CTX_SPEC)
    assert kept(optuna.trial.FixedTrial(sampled)) == 1.0

    pruned = make_objective(
        base, lambda _c: 1.0, model_spec=BIG_CTX_SPEC, served_max_model_len=4096
    )
    with pytest.raises(optuna.TrialPruned, match="served window of 4096 tok"):
        pruned(optuna.trial.FixedTrial(sampled))


def test_resolve_study_window_takes_an_injected_probe_without_touching_a_backend():
    """The CI seam: `probe=False` with an explicit served window never opens a socket."""
    base = RunConfig(max_tokens=128)
    served, provenance = resolve_study_window(
        base, model_spec=BIG_CTX_SPEC, vram_mib=0, ram_mib=0, served_max_model_len=4096
    )
    assert served == 4096
    assert provenance == {
        "declared_max_model_len": 131072,
        "served_max_model_len": 4096,
        "budget_source": BUDGET_SOURCE_SERVED,
    }

    unprobed, unprobed_provenance = resolve_study_window(
        base, model_spec=BIG_CTX_SPEC, vram_mib=0, ram_mib=0, probe=False
    )
    assert unprobed is None
    assert unprobed_provenance["budget_source"] == BUDGET_SOURCE_DECLARED


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
