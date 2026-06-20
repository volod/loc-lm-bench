"""Telemetry: throughput protocol, tokenizer efficiency, VRAM peak (fakes; no GPU)."""

from llb.backends.base import ChatResult
from llb.backends.telemetry import (
    VramSampler,
    collect_telemetry,
    measure_throughput,
    tokens_per_char,
)


def test_tokens_per_char():
    assert tokens_per_char(5, "abcde") == 1.0
    assert tokens_per_char(2, "abcd") == 0.5
    assert tokens_per_char(3, "") == 0.0


class FakeChat:
    """Records calls; returns a fixed-rate ChatResult, plus counts warmup vs measured."""

    def __init__(self, completion_tokens=10, latency=0.5, text="x" * 20):
        self.calls = 0
        self._r = ChatResult(text=text, completion_tokens=completion_tokens, latency_s=latency)

    def __call__(self, messages, max_tokens, temperature, timeout):
        self.calls += 1
        return self._r


def test_measure_throughput_discards_warmup_and_computes_rate():
    chat = FakeChat(completion_tokens=10, latency=0.5, text="y" * 20)
    res = measure_throughput(chat, ["p1", "p2"], max_new_tokens=64, warmup=1, passes=1)
    assert chat.calls == 4  # 1 warmup pass (2) + 1 measured pass (2)
    assert res.n_measured == 2 and res.n_failed == 0
    assert res.steady_tokens_per_s == 20.0  # 10 tokens / 0.5 s
    assert res.tokens_per_char == 0.5  # 10 tokens / 20 chars


def test_measure_throughput_counts_failures():
    def chat(messages, max_tokens, temperature, timeout):
        return ChatResult(text="", error="timeout")

    res = measure_throughput(chat, ["p"], warmup=0)
    assert res.n_measured == 0 and res.n_failed == 1
    assert res.steady_tokens_per_s == 0.0


def test_vram_sampler_tracks_peak():
    values = iter([2000, 9000, 4000])
    sampler = VramSampler(reader=lambda: next(values))
    for _ in range(3):
        sampler.sample()
    assert sampler.peak_mb == 9000


def test_vram_sampler_no_reader_is_noop():
    with VramSampler(reader=None) as s:
        pass
    assert s.peak_mb == 0


def test_collect_telemetry_assembles_record():
    class FakeLauncher:
        load_time_s = 3.5
        meta = {"backend": "vllm", "gpu_memory_utilization": 0.85}

        def chat(self, messages, max_tokens, temperature, timeout):
            return ChatResult(text="z" * 10, completion_tokens=5, latency_s=0.25)

        def served_context(self):
            return 4096

    report = collect_telemetry(
        FakeLauncher(), requested_context=8192, vram_reader=lambda: 7000, warmup=0
    )
    assert report["steady_tokens_per_s"] == 20.0  # 5 tokens / 0.25 s
    assert report["served_context"] == 4096 and report["requested_context"] == 8192
    assert report["load_time_s"] == 3.5
    assert report["backend"] == "vllm"
    assert report["peak_vram_mb"] == 7000
