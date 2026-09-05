"""Run bundle heads for tests, written the way a producer writes one.

Every board, recommendation, and profile lane now admits a run only through `llb.run-manifest`, so
a hand-rolled mapping that omits the fields a real bundle always records is not a weaker fixture --
it is a bundle the board is right to refuse. Building the fixture through `RunManifest` is what
keeps a test's idea of a run and a producer's the same one, and it is why a column added to the
manifest reaches these fixtures without each of them being edited.
"""

import json
from pathlib import Path

from llb.core.contracts.hardware import TelemetryReport
from llb.tracking.manifest import RunManifest

ENV = {"python": "3.13.0", "platform": "Linux-test"}


def telemetry(**overrides: object) -> TelemetryReport:
    """One complete telemetry report, with only the fields a test cares about overridden."""
    report: TelemetryReport = {
        "steady_tokens_per_s": 20.0,
        "mean_completion_tokens": 64.0,
        "tokens_per_char": 0.25,
        "max_new_tokens": 256,
        "n_warmup": 1,
        "n_measured": 3,
        "n_failed": 0,
        "load_time_s": None,
        "peak_vram_mb": None,
        "requested_context": None,
        "served_context": None,
        "backend": "ollama",
        "gpu_memory_utilization": None,
        "n_gpu_layers": None,
        "gpus": [],
    }
    report.update(overrides)  # type: ignore[typeddict-item]
    return report


def manifest_payload(run_dir: Path, **fields: object) -> dict[str, object]:
    """One bundle head as JSON, with the identity fields a test does not care about defaulted."""
    defaults: dict[str, object] = {
        "run_id": run_dir.name,
        "run_name": run_dir.name,
        "created_at": "2026-06-21T00:00:00Z",
        "config": {},
        "env": ENV,
    }
    return RunManifest(**{**defaults, **fields}).model_dump(mode="json")  # type: ignore[arg-type]


def write_manifest(run_dir: Path, **fields: object) -> Path:
    """Write one bundle head into `run_dir` and return its path."""
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest_payload(run_dir, **fields)), encoding="utf-8")
    return path
