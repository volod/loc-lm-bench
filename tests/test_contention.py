import pytest

from llb.executor.contention import (
    ACTION_ABORT,
    ACTION_DERATE,
    ACTION_OK,
    apply_contention_guard,
    evict_ollama,
    model_weight_floor_mb,
    plan_guard,
    resident_users,
)

# A 16 GB card with the configured gpu-memory-utilization default.
TOTAL = 16000
REQ = 0.85
E4B_FLOOR = 9800  # the M4.1 embedding-aware weight floor (MiB), ~ the measured 9.8 GiB


def test_plan_guard_no_contention_keeps_requested():
    report = plan_guard(TOTAL, free_mb=15500, requested_util=REQ, weight_floor_mb=E4B_FLOOR)
    assert report["action"] == ACTION_OK
    assert report["safe_util"] == REQ and not report["derated"] and report["fits"]


def test_plan_guard_derates_under_contention():
    # Mild contention: ~13 GB free, a small model -> util drops below 0.85 but still fits.
    report = plan_guard(
        TOTAL, free_mb=13000, requested_util=REQ, weight_floor_mb=6000, overhead_mb=512
    )
    assert report["action"] == ACTION_DERATE and report["derated"]
    assert report["safe_util"] < REQ
    assert report["safe_util"] * TOTAL <= 13000  # never exceeds free VRAM
    assert report["target_mb"] >= 6000 + 512  # still holds weights + KV


def test_plan_guard_aborts_when_overhead_leaves_no_kv():
    # The live M4.2 finding: at ~2.75 GB contention, free ~12.4 GB cannot hold E4B's weights
    # (~10 GB) + vLLM's ~2 GB serving overhead + KV, so the guard must ABORT, not derate into the
    # "No available memory for the cache blocks" failure.
    report = plan_guard(16380, free_mb=12398, requested_util=0.80, weight_floor_mb=10049)
    assert report["action"] == ACTION_ABORT and not report["fits"]
    assert "serving overhead" in report["note"]


def test_plan_guard_aborts_when_free_cannot_hold_weights():
    report = plan_guard(TOTAL, free_mb=3000, requested_util=REQ, weight_floor_mb=E4B_FLOOR)
    assert report["action"] == ACTION_ABORT and not report["fits"]
    assert "Free VRAM" in report["note"]  # actionable message


def test_plan_guard_records_residents_in_note():
    residents = [{"pid": 4242, "used_mb": 2800}]
    report = plan_guard(TOTAL, 13000, REQ, E4B_FLOOR, residents)
    assert report["residents"] == residents
    assert "4242" in report["note"] and "2800" in report["note"]


def test_resident_users_sorts_and_excludes_self():
    usage = {10: 2800, 20: 500, 99: 4000, 7: 0}
    out = resident_users(usage, exclude={99})
    assert out == [
        {"pid": 10, "used_mb": 2800},
        {"pid": 20, "used_mb": 500},
    ]  # 99 excluded, 0 dropped


def test_apply_guard_no_gpu_returns_none():
    assert (
        apply_contention_guard(
            requested_util=REQ, weight_floor_mb=E4B_FLOOR, gpu_reader=lambda: None
        )
        is None
    )


def test_apply_guard_derates_with_resident_process():
    report = apply_contention_guard(
        requested_util=REQ,
        weight_floor_mb=E4B_FLOOR,
        gpu_reader=lambda: (TOTAL, 13000),
        process_reader=lambda: {4242: 2800},
        own_pids=set(),
    )
    assert report is not None and report["action"] == ACTION_DERATE
    assert report["residents"] == [{"pid": 4242, "used_mb": 2800}]


def test_apply_guard_evict_then_replan():
    state = {"evicted": False}

    def gpu_reader() -> tuple[int, int]:
        return (TOTAL, 13000 if state["evicted"] else 2000)

    def evict_fn(_host: str) -> None:
        state["evicted"] = True

    report = apply_contention_guard(
        requested_util=REQ,
        weight_floor_mb=E4B_FLOOR,
        gpu_reader=gpu_reader,
        evict=True,
        evict_fn=evict_fn,
        wait_timeout_s=0.0,  # evict frees immediately; no need to poll
        sleep=lambda _s: None,
    )
    assert state["evicted"]
    assert report is not None and report["free_mb"] == 13000 and report["fits"]


def test_apply_guard_wait_polls_until_free():
    reads = [2000, 2000, 14200, 14200, 14200]
    calls = {"i": 0}

    def gpu_reader() -> tuple[int, int]:
        free = reads[min(calls["i"], len(reads) - 1)]
        calls["i"] += 1
        return (TOTAL, free)

    report = apply_contention_guard(
        requested_util=REQ,
        weight_floor_mb=E4B_FLOOR,
        gpu_reader=gpu_reader,
        wait=True,
        wait_timeout_s=10.0,
        poll_s=1.0,
        sleep=lambda _s: None,
    )
    assert report is not None and report["free_mb"] == 14200
    assert report["action"] == ACTION_OK  # waiting freed enough; no derate needed


def test_evict_ollama_unloads_each_running_model():
    posted: list[dict] = []
    evict_ollama(
        "http://localhost:11434",
        http_get=lambda _url: {"models": [{"name": "llama3.2:3b"}, {"model": "gemma:2b"}]},
        http_post=lambda _url, payload: posted.append(payload),
    )
    assert posted == [
        {"model": "llama3.2:3b", "keep_alive": 0},
        {"model": "gemma:2b", "keep_alive": 0},
    ]


def test_evict_ollama_swallows_errors():
    def boom(_url: str) -> dict:
        raise OSError("connection refused")

    evict_ollama("http://localhost:11434", http_get=boom)  # must not raise


def test_model_weight_floor_uses_embedding_aware_estimate():
    # Integration with M4.1: the E4B floor must reflect the corrected ~9.8 GiB, not the flat 4.2.
    floor = model_weight_floor_mb("google/gemma-4-E4B-it-qat-w4a16-ct")
    assert 9500 <= floor <= 10500
    assert model_weight_floor_mb("does/not-exist") == 0.0


def test_runner_guard_applies_derate_and_aborts(monkeypatch):
    from llb.backends.vllm import VllmLauncher
    from llb.config import RunConfig
    from llb.executor import contention, runner

    launcher = VllmLauncher("google/gemma-4-E4B-it-qat-w4a16-ct", gpu_memory_utilization=0.85)
    cfg = RunConfig(model="google/gemma-4-E4B-it-qat-w4a16-ct", backend="vllm")

    derate = plan_guard(TOTAL, 13000, 0.85, E4B_FLOOR)
    monkeypatch.setattr(contention, "apply_contention_guard", lambda **_kw: derate)
    report = runner._guard_vllm_contention(cfg, launcher, evict=False, wait=False)
    assert report is not None and report["derated"]
    assert launcher.gpu_memory_utilization == derate["safe_util"]  # launcher actually adjusted
    assert launcher.meta["gpu_memory_utilization"] == derate["safe_util"]

    abort = plan_guard(TOTAL, 3000, 0.85, E4B_FLOOR)
    monkeypatch.setattr(contention, "apply_contention_guard", lambda **_kw: abort)
    with pytest.raises(SystemExit):
        runner._guard_vllm_contention(cfg, launcher, evict=False, wait=False)
