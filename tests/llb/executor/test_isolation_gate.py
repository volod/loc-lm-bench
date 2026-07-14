"""Tests for isolation gate."""

import pytest
from llb.executor.isolation import (
    isolate_cell,
    run_sweep,
)
from llb.executor.vram import VERDICT_BASELINE_SHIFT, VERDICT_RECLAIMED, VramNotReclaimed
from test_isolation import cfg, gpu


def test_isolate_cell_reclaimed_runs_work_and_cools_down():
    out, iso = isolate_cell(
        lambda: "result",
        backend="vllm",
        vram_reader=lambda: 1000,  # constant -> reclaimed within tolerance
        pid_usage_reader=lambda: {100: 500},
        gpu_sampler=lambda: gpu(40),
        sleep=lambda _s: None,
    )
    assert out == "result"
    assert iso["vram_verdict"] == VERDICT_RECLAIMED and iso["vram_residual_mb"] == 0


def test_isolate_cell_aborts_on_attributed_leak():
    reads = iter([1000] + [9000] * 100)  # baseline 1000, then stuck high -> residual 8000
    usage = iter([{100: 500}, {100: 500, 200: 3000}])  # a NEW pid (200) still holds VRAM
    with pytest.raises(VramNotReclaimed, match="leaked"):
        isolate_cell(
            lambda: "x",
            backend="vllm",
            vram_reader=lambda: next(reads),
            pid_usage_reader=lambda: next(usage),
            gpu_sampler=lambda: gpu(40),
            sleep=lambda _s: None,
        )


def test_isolate_cell_tolerates_baseline_shift():
    reads = iter([1000] + [9000] * 100)  # residual 8000
    usage = iter([{100: 500}, {100: 8500}])  # only the PRE-EXISTING pid 100 grew (no new pid)
    out, iso = isolate_cell(
        lambda: "done",
        backend="vllm",
        vram_reader=lambda: next(reads),
        pid_usage_reader=lambda: next(usage),
        gpu_sampler=lambda: gpu(40),
        sleep=lambda _s: None,
    )
    assert out == "done"
    assert iso["vram_verdict"] == VERDICT_BASELINE_SHIFT and iso["vram_residual_mb"] == 8000


def test_run_sweep_aborts_on_attributed_leak(tmp_path):
    reads = iter([1000] + [9000] * 200)
    usage = iter([{100: 500}, {100: 500, 200: 4000}])  # leaked launched pid 200
    with pytest.raises(VramNotReclaimed):
        run_sweep(
            [cfg(tmp_path, backend="vllm")],
            sweep_id="leak",
            data_dir=tmp_path,
            cell_runner=lambda c, s: "dir",
            vram_reader=lambda: next(reads),
            pid_usage_reader=lambda: next(usage),
            gpu_sampler=lambda: gpu(40),
            sleep=lambda _s: None,
        )


def test_run_sweep_completes_through_baseline_shift(tmp_path):
    reads = iter([1000] + [9000] * 200)
    usage = iter([{100: 500}, {100: 8500}])  # unrelated process grew -> not a leak
    report = run_sweep(
        [cfg(tmp_path, backend="vllm")],
        sweep_id="bshift",
        data_dir=tmp_path,
        cell_runner=lambda c, s: str(tmp_path / "d"),
        vram_reader=lambda: next(reads),
        pid_usage_reader=lambda: next(usage),
        gpu_sampler=lambda: gpu(40),
        sleep=lambda _s: None,
    )
    assert report["completed"] == 1 and report["results"][0]["vram_residual_mb"] == 8000
