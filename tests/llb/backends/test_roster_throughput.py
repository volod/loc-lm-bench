"""Roster throughput protocol: measured placement, run sizing, and the doc row (fakes; no GPU)."""

import json

import pytest

from llb.backends.base import BackendLauncher, ChatResult
from llb.backends.ollama import OllamaPlacement, parse_ollama_placement
from llb.backends.telemetry_samplers import LastValueSampler
from llb.backends.roster_throughput import (
    PROTOCOL_CONTEXT,
    warm_load,
    ThroughputRow,
    markdown_row,
    markdown_table,
    measure_entry,
    minutes_per_100_cases,
    placement_label,
)


def _ps_body(name: str, size: int, size_vram: int) -> str:
    return json.dumps(
        {"models": [{"name": name, "model": name, "size": size, "size_vram": size_vram}]}
    )


def test_placement_reports_the_split_ollama_itself_prints() -> None:
    # 17 GB total with 9.5 GB resident -> 44% CPU / 56% GPU, the shape `ollama ps` shows.
    split = parse_ollama_placement(_ps_body("qwen3.6:27b", 17_000, 9_520), "qwen3.6:27b")
    assert split == OllamaPlacement(total_bytes=17_000, vram_bytes=9_520)
    assert (split.cpu_percent, split.gpu_percent) == (44, 56)
    assert split.label == "offload 44%/56% CPU/GPU"


def test_placement_labels_the_two_ends() -> None:
    assert parse_ollama_placement(_ps_body("m", 100, 100), "m").label == "GPU-resident"
    assert parse_ollama_placement(_ps_body("m", 100, 0), "m").label == "CPU-only"


def test_placement_matches_a_tagless_name_and_ignores_other_models() -> None:
    assert parse_ollama_placement(_ps_body("m:latest", 100, 50), "m") is not None
    assert parse_ollama_placement(_ps_body("other:27b", 100, 50), "m") is None


@pytest.mark.parametrize("body", ["not json", json.dumps({"models": "nope"}), json.dumps({})])
def test_placement_is_best_effort_on_a_bad_body(body: str) -> None:
    assert parse_ollama_placement(body, "m") is None


def test_minutes_per_100_is_decode_only_and_zero_when_unknown() -> None:
    # 100 cases * 256 tokens / 25.6 tok/s = 1000 s = 16.67 min
    assert minutes_per_100_cases(25.6) == pytest.approx(16.667, abs=0.01)
    assert minutes_per_100_cases(0.0) == 0.0


class _FakeLauncher(BackendLauncher):
    """A launcher that answers at a fixed rate and reports its own placement."""

    def __init__(self, split: OllamaPlacement | None = None):
        super().__init__(model="fake", meta={"backend": "ollama"})
        self.started = self.stopped = False
        self._split = split

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def chat(self, messages, max_tokens, temperature, timeout):  # type: ignore[no-untyped-def]
        return ChatResult(text="x" * 40, completion_tokens=20, latency_s=0.5)

    def placement(self) -> OllamaPlacement | None:
        return self._split


def test_measure_entry_runs_the_protocol_and_stops_the_launcher() -> None:
    launcher = _FakeLauncher(OllamaPlacement(total_bytes=100, vram_bytes=56))
    seen: dict[str, object] = {}

    def factory(backend: str, source: str, context: int) -> BackendLauncher:
        seen.update(backend=backend, source=source, context=context)
        return launcher

    row = measure_entry("qwen3.8-27b", "ollama", "qwen3.8:27b", launcher_factory=factory)

    assert seen == {"backend": "ollama", "source": "qwen3.8:27b", "context": PROTOCOL_CONTEXT}
    assert launcher.started and launcher.stopped
    assert row.tokens_per_s == 40.0  # 20 tokens / 0.5 s
    assert row.tokens_per_char == 0.5  # 20 tokens / 40 chars
    assert row.placement == "offload 44%/56% CPU/GPU"
    assert row.telemetry["requested_context"] == PROTOCOL_CONTEXT
    assert row.telemetry["max_new_tokens"] == 128 and row.telemetry["n_warmup"] == 1


def test_last_value_sampler_keeps_the_last_reading_that_existed() -> None:
    readings = iter([None, OllamaPlacement(total_bytes=100, vram_bytes=72), None])
    sampler = LastValueSampler(lambda: next(readings))
    for _ in range(3):
        sampler.sample()
    # The signal vanished before the run ended; what the run was served under still stands.
    assert sampler.value == OllamaPlacement(total_bytes=100, vram_bytes=72)


def test_placement_label_prefers_what_was_observed_during_the_run() -> None:
    # The launcher reports nothing now -- Ollama evicted the runner as the last request returned.
    evicted = _FakeLauncher()
    observed = OllamaPlacement(total_bytes=100, vram_bytes=72)
    assert placement_label(evicted, "ollama", observed) == "offload 28%/72% CPU/GPU"


def test_measure_entry_record_carries_the_protocol_beside_the_reading() -> None:
    row = measure_entry(
        "qwen3.8-27b", "ollama", "qwen3.8:27b", launcher_factory=lambda *_: _FakeLauncher()
    )
    record = row.as_record()
    assert record["name"] == "qwen3.8-27b" and record["served_artifact"] == "qwen3.8:27b"
    assert record["protocol"] == {
        "max_new_tokens": 128,
        "warmup": 1,
        "context": PROTOCOL_CONTEXT,
        "sizing_cases": 100,
        "sizing_output_tokens": 256,
    }
    assert record["telemetry"]["steady_tokens_per_s"] == 40.0


def test_vllm_without_a_split_is_gpu_resident_by_construction() -> None:
    assert placement_label(BackendLauncher(model="m"), "vllm") == "GPU-resident"
    assert placement_label(_FakeLauncher(), "ollama") == "unknown"


def _row(name: str, rate: float) -> ThroughputRow:
    return ThroughputRow(
        name=name,
        backend="ollama",
        served_artifact=f"{name}:tag",
        placement="GPU-resident",
        telemetry={"steady_tokens_per_s": rate, "tokens_per_char": 0.35, "peak_vram_mb": 15233},  # type: ignore[typeddict-item]
    )


def test_markdown_row_matches_the_baseline_table_shape() -> None:
    assert markdown_row(_row("qwen3.8-27b", 4.59)) == (
        "| `qwen3.8-27b` | `qwen3.8-27b:tag` | ollama | 4.59 | 0.350 | 93.0 | 15233 | GPU-resident |"
    )


def test_markdown_table_orders_fastest_first() -> None:
    lines = markdown_table([_row("slow", 4.59), _row("fast", 63.45)]).splitlines()
    assert lines[0].startswith("| model |") and lines[1].startswith("| --- |")
    assert "`fast`" in lines[2] and "`slow`" in lines[3]


class _DaemonLauncher(_FakeLauncher):
    """An Ollama-shaped launcher: it knows its window only after a request has loaded the model."""

    def __init__(self) -> None:
        super().__init__()
        self.warmed = 0

    def served_context(self) -> int | None:
        return PROTOCOL_CONTEXT if self.warmed else None

    def ensure_num_ctx(self, timeout: float = 120.0) -> int | None:
        self.warmed += 1
        return PROTOCOL_CONTEXT


def test_warm_load_times_the_load_a_daemon_backend_would_charge_to_the_first_pass() -> None:
    launcher = _DaemonLauncher()
    warm_load(launcher)
    assert launcher.warmed == 1
    assert launcher.load_time_s is not None and launcher.load_time_s >= 0.0


def test_warm_load_leaves_a_launcher_that_timed_its_own_startup_alone() -> None:
    launcher = _DaemonLauncher()
    launcher.load_time_s = 112.0  # vLLM: measured from launch to readiness
    warm_load(launcher)
    assert launcher.warmed == 0 and launcher.load_time_s == 112.0
