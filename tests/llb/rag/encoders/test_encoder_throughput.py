"""CI coverage for encoder throughput decomposition (injected clock + fake encoders)."""

import pytest

from llb.rag.encoders.throughput import (
    STOP_MAX_PASSES,
    STOP_MAX_SECONDS,
    STOP_PRECISION,
    measure_encoder_throughput,
    relative_precision,
)
from llb.rag.encoders.throughput_report import format_host_summary, render_host_markdown
from llb.rag.encoders.throughput_summary import build_host_summary, ordering_survives, rate_order


def _step_clock(values: list[float]):
    idx = {"i": 0}

    def clock() -> float:
        value = values[idx["i"]]
        idx["i"] += 1
        return float(value)

    return clock


def test_relative_precision_empty_and_single():
    assert relative_precision([]) == (0.0, 0.0, 0.0)
    assert relative_precision([4.0]) == (4.0, 0.0, 0.0)


def test_relative_precision_iqr_over_median():
    median, iqr, precision = relative_precision([10.0, 12.0, 14.0, 16.0])
    assert median == pytest.approx(13.0)
    assert iqr == pytest.approx(4.0)  # median([10,12])=11, median([14,16])=15
    assert precision == pytest.approx(4.0 / 13.0)


def test_measure_stops_on_precision_with_injected_clock():
    # clock advances 1s per call; load=1, first=1, each warm=1 -> precision 0 after 3 identical.
    # calls: load(2) + first(2) + budget_start(1) + 3*(start,end)
    values = [0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
    loads: list[int] = []
    encodes: list[int] = []

    profile = measure_encoder_throughput(
        "fake-model",
        ["a", "b", "c", "d"],
        load=lambda: loads.append(1),
        encode=lambda texts: encodes.append(len(texts)),
        device="cuda",
        clock=_step_clock(values),
        target_relative_precision=0.05,
        min_warm_passes=3,
        max_warm_passes=10,
        max_warm_seconds=100.0,
    )
    assert loads == [1]
    assert encodes == [4, 4, 4, 4]  # first + 3 warm
    assert profile["stopping_reason"] == STOP_PRECISION
    assert profile["warm_passes"] == 3
    assert profile["warm_relative_precision"] == 0.0
    assert profile["warm_chunks_per_s"] == pytest.approx(4.0)
    assert profile["compile_estimate_seconds"] == pytest.approx(0.0)
    assert profile["one_pass_chunks_per_s"] == pytest.approx(2.0)  # 4 texts / (1+1)


def test_measure_stops_on_max_passes_when_noisy():
    # Alternating warm times 1.0 and 3.0 keep IQR/median above the target.
    values = [
        0.0,
        1.0,  # load
        1.0,
        2.0,  # first
        2.0,  # warm_budget_start
        2.0,
        3.0,  # warm 1 (=1s)
        3.0,
        6.0,  # warm 2 (=3s)
        6.0,
        7.0,  # warm 3 (=1s)
        7.0,
        10.0,  # warm 4 (=3s)
        10.0,
        11.0,  # warm 5 (=1s)
    ]
    profile = measure_encoder_throughput(
        "noisy",
        ["x"] * 10,
        load=lambda: None,
        encode=lambda _t: None,
        device="cpu",
        clock=_step_clock(values),
        target_relative_precision=0.01,
        min_warm_passes=3,
        max_warm_passes=5,
        max_warm_seconds=1000.0,
    )
    assert profile["stopping_reason"] == STOP_MAX_PASSES
    assert profile["warm_passes"] == 5
    assert profile["warm_relative_precision"] > 0.01


def test_measure_stops_on_max_seconds_cap():
    # Each warm takes 10s; after warm 3, ended - budget_start = 30 >= 25.
    values = [
        0.0,
        1.0,  # load
        1.0,
        2.0,  # first
        2.0,  # warm_budget_start
        2.0,
        12.0,  # warm 1
        12.0,
        22.0,  # warm 2
        22.0,
        32.0,  # warm 3
    ]
    profile = measure_encoder_throughput(
        "capped",
        ["y"] * 5,
        load=lambda: None,
        encode=lambda _t: None,
        device="cuda",
        clock=_step_clock(values),
        target_relative_precision=0.001,
        min_warm_passes=10,
        max_warm_passes=10,
        max_warm_seconds=25.0,
    )
    assert profile["stopping_reason"] == STOP_MAX_SECONDS
    assert profile["warm_passes"] == 3


def test_ordering_survives_and_host_summary():
    fast = measure_encoder_throughput(
        "fast",
        ["a"] * 10,
        load=lambda: None,
        encode=lambda _t: None,
        device="cuda",
        clock=_step_clock([0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]),
        min_warm_passes=3,
        max_warm_passes=3,
    )
    slow = measure_encoder_throughput(
        "slow",
        ["a"] * 10,
        load=lambda: None,
        encode=lambda _t: None,
        device="cuda",
        clock=_step_clock([0, 2, 2, 12, 12, 12, 22, 22, 32, 32, 42]),
        min_warm_passes=3,
        max_warm_passes=3,
    )
    profiles = [slow, fast]
    assert rate_order(profiles, warm=True) == ["fast", "slow"]
    assert ordering_survives(profiles) is True
    summary = build_host_summary(profiles, corpus_n_texts=10)
    assert summary["ordering_survives"] is True
    assert summary["by_device"]["cuda"]["ordering_survives"] is True
    assert "MATCHES" in summary["verdict"]
    text = format_host_summary(summary)
    assert "fast" in text and "warm_c/s" in text
    md = render_host_markdown(summary)
    assert md.startswith("# Encoder throughput")
    assert "ordering survives warm measurement: `True`" in md


def test_ordering_does_not_survive_when_warm_flips():
    # one-pass: a is faster (cheap load+first); warm: b is faster.
    a = measure_encoder_throughput(
        "a",
        ["t"] * 8,
        load=lambda: None,
        encode=lambda _t: None,
        device="cuda",
        clock=_step_clock([0, 0.5, 0.5, 1.0, 1.0, 1.0, 5.0, 5.0, 9.0, 9.0, 13.0]),
        min_warm_passes=3,
        max_warm_passes=3,
    )
    b = measure_encoder_throughput(
        "b",
        ["t"] * 8,
        load=lambda: None,
        encode=lambda _t: None,
        device="cuda",
        clock=_step_clock([0, 5.0, 5.0, 15.0, 15.0, 15.0, 16.0, 16.0, 17.0, 17.0, 18.0]),
        min_warm_passes=3,
        max_warm_passes=3,
    )
    assert rate_order([a, b], warm=False)[0] == "a"
    assert rate_order([a, b], warm=True)[0] == "b"
    assert ordering_survives([a, b]) is False
    summary = build_host_summary([a, b], corpus_n_texts=8)
    assert summary["ordering_survives"] is False
    assert "DIFFERS" in summary["verdict"]


def test_host_summary_orders_per_device_not_mixed():
    # CUDA warm order flips; CPU does not. Headline follows CUDA.
    cuda_a = measure_encoder_throughput(
        "a",
        ["t"] * 8,
        load=lambda: None,
        encode=lambda _t: None,
        device="cuda",
        clock=_step_clock([0, 0.5, 0.5, 1.0, 1.0, 1.0, 5.0, 5.0, 9.0, 9.0, 13.0]),
        min_warm_passes=3,
        max_warm_passes=3,
    )
    cuda_b = measure_encoder_throughput(
        "b",
        ["t"] * 8,
        load=lambda: None,
        encode=lambda _t: None,
        device="cuda",
        clock=_step_clock([0, 5.0, 5.0, 15.0, 15.0, 15.0, 16.0, 16.0, 17.0, 17.0, 18.0]),
        min_warm_passes=3,
        max_warm_passes=3,
    )
    cpu_a = measure_encoder_throughput(
        "a",
        ["t"] * 8,
        load=lambda: None,
        encode=lambda _t: None,
        device="cpu",
        clock=_step_clock([0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]),
        min_warm_passes=3,
        max_warm_passes=3,
    )
    cpu_b = measure_encoder_throughput(
        "b",
        ["t"] * 8,
        load=lambda: None,
        encode=lambda _t: None,
        device="cpu",
        clock=_step_clock([0, 2, 2, 12, 12, 12, 22, 22, 32, 32, 42]),
        min_warm_passes=3,
        max_warm_passes=3,
    )
    summary = build_host_summary([cuda_a, cuda_b, cpu_a, cpu_b], corpus_n_texts=8)
    assert summary["by_device"]["cuda"]["ordering_survives"] is False
    assert summary["by_device"]["cpu"]["ordering_survives"] is True
    assert summary["ordering_survives"] is False  # headline = cuda
    assert summary["warm_order"] == ["b", "a"]


def test_faster_than_baseline_names_warm_winners():
    base = measure_encoder_throughput(
        "base",
        ["t"] * 8,
        load=lambda: None,
        encode=lambda _t: None,
        device="cuda",
        clock=_step_clock([0, 1, 1, 2, 2, 2, 4, 4, 6, 6, 8]),
        min_warm_passes=3,
        max_warm_passes=3,
    )
    cheap = measure_encoder_throughput(
        "cheap",
        ["t"] * 8,
        load=lambda: None,
        encode=lambda _t: None,
        device="cuda",
        clock=_step_clock([0, 1, 1, 2, 2, 2, 2.5, 2.5, 3.0, 3.0, 3.5]),
        min_warm_passes=3,
        max_warm_passes=3,
    )
    slow = measure_encoder_throughput(
        "slow",
        ["t"] * 8,
        load=lambda: None,
        encode=lambda _t: None,
        device="cuda",
        clock=_step_clock([0, 1, 1, 2, 2, 2, 12, 12, 22, 22, 32]),
        min_warm_passes=3,
        max_warm_passes=3,
    )
    summary = build_host_summary([base, cheap, slow], corpus_n_texts=8, baseline_model="base")
    assert [row["model"] for row in summary["faster_than_baseline"]] == ["cheap"]
    assert summary["faster_than_baseline"][0]["speedup_vs_baseline"] == pytest.approx(4.0)
    assert "Faster than baseline" in summary["verdict"]
    assert "cheap=" in format_host_summary(summary)
    assert "faster than baseline" in render_host_markdown(summary)
