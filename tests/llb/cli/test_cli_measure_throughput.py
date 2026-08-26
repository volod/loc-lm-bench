"""`measure-throughput` CLI wiring: model selection, forced serving choice, record file."""

import json

import pytest
import typer

from llb.backends.roster_throughput import ThroughputRow
from llb.cli.models import throughput


def _specs() -> list[dict]:
    return [{"name": "qwen3.8-27b"}, {"name": "qwen3.6-27b"}, {"name": "gemma-4-e4b-it-w4a16"}]


def test_empty_selection_measures_every_manifest_entry() -> None:
    assert throughput._selected_specs(_specs(), "  ") == _specs()


def test_selection_keeps_the_order_the_names_were_given_in() -> None:
    picked = throughput._selected_specs(_specs(), "gemma-4-e4b-it-w4a16, qwen3.8-27b")
    assert [s["name"] for s in picked] == ["gemma-4-e4b-it-w4a16", "qwen3.8-27b"]


def test_unknown_model_name_is_a_usage_error_not_a_silent_skip() -> None:
    with pytest.raises(typer.Exit):
        throughput._selected_specs(_specs(), "qwen3.9-27b")


def test_forced_backend_and_source_skip_host_resolution() -> None:
    # No GPU / no probes touched: both halves are given, so nothing is resolved.
    assert throughput._serving_choice(
        {"name": "qwen3.8-27b"}, "ollama", "qwen3.8:27b", offline=True
    ) == ("ollama", "qwen3.8:27b")


def test_records_are_written_as_one_json_array_per_measured_model(tmp_path) -> None:
    row = ThroughputRow(
        name="qwen3.8-27b",
        backend="ollama",
        served_artifact="qwen3.8:27b",
        placement="offload 44%/56% CPU/GPU",
        telemetry={"steady_tokens_per_s": 4.59, "tokens_per_char": 0.35, "peak_vram_mb": 15233},  # type: ignore[typeddict-item]
    )
    out = tmp_path / "nested" / "rows.json"
    throughput._write_records(out, [row])
    records = json.loads(out.read_text(encoding="utf-8"))
    assert [r["name"] for r in records] == ["qwen3.8-27b"]
    assert records[0]["placement"] == "offload 44%/56% CPU/GPU"
    assert records[0]["minutes_per_100_cases"] == pytest.approx(93.0, abs=0.1)
