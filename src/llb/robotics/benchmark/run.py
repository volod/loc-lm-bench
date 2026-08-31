"""Compose and execute the paired robotics RAG benchmark operation bundle."""

import shutil
import time
from pathlib import Path
from typing import Any

from llb.backends.base import ChatResult
from llb.backends.ollama import OllamaLauncher
from llb.backends.telemetry_samplers import (
    LastValueSampler,
    PowerSampler,
    VramSampler,
    nvidia_smi_power_reader,
)
from llb.bench.common import new_run_timestamp
from llb.core.contracts.common import ChatMessage
from llb.core.paths import resolve_data_dir
from llb.robotics.benchmark.constants import (
    LANE_NO_RETRIEVAL,
    LANE_REFERENCE,
    LANE_RETRIEVAL,
    METHOD_NAME,
)
from llb.robotics.benchmark.design import load_design
from llb.robotics.benchmark.execution import evaluate_task
from llb.robotics.benchmark.metrics import aggregate, paired_verdict
from llb.robotics.benchmark.models import BenchmarkTask
from llb.robotics.benchmark.profile import latest_profile, load_measured_profile
from llb.robotics.benchmark.report import write_bundle
from llb.robotics.benchmark.retrieval import build_benchmark_store, retrieve_context
from llb.robotics.digests import file_digest, value_digest
from llb.robotics.emulator_fixture import load_emulator_fixture


def validate_benchmark_design(design_path: Path) -> dict[str, object]:
    design, tasks = load_design(design_path)
    return {
        "benchmark_id": design.benchmark_id,
        "task_count": len(tasks),
        "fault_classes": list(design.mandatory_fault_classes),
        "minimum_detectable_gain": design.minimum_detectable_gain,
        "minimum_evidence_count": design.minimum_evidence_count,
    }


def _vram_reader():  # type: ignore[no-untyped-def]
    try:
        from llb.executor.vram import nvml_reader

        return nvml_reader()
    except (Exception, SystemExit):
        return None


def _telemetry(
    launcher: OllamaLauncher,
    fixture: Any,
    tasks: tuple[Any, ...],
    lane: str,
    retrieve: Any,
    complete: Any,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    vram = VramSampler(_vram_reader())
    power = PowerSampler(nvidia_smi_power_reader())
    placement = LastValueSampler(launcher.placement)
    started = time.monotonic()
    with vram, power, placement:
        rows = [
            evaluate_task(
                fixture,
                task,
                lane,
                retrieve=retrieve,
                complete=complete,
            )
            for task in tasks
        ]
    placed = placement.value
    return rows, {
        "elapsed_s": round(time.monotonic() - started, 3),
        "peak_vram_mb": vram.peak_mb or None,
        "mean_power_w": round(power.mean_w, 2) if power.mean_w is not None else None,
        "peak_power_w": round(power.peak_w, 2) if power.peak_w is not None else None,
        "power_samples": len(power.samples),
        "placement": placed.label if placed is not None else None,
        "placement_total_bytes": placed.total_bytes if placed is not None else None,
        "placement_vram_bytes": placed.vram_bytes if placed is not None else None,
    }


def run_benchmark(
    *,
    design_path: Path,
    emulator_path: Path,
    hflow_fixture: Path,
    manual_corpus: Path,
    model: str,
    backend: str = "ollama",
    agent_profile: Path | None = None,
    data_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    if backend != "ollama":
        raise ValueError("the robotics benchmark currently supports the local Ollama backend")
    root = resolve_data_dir(data_dir)
    profile_path = agent_profile or latest_profile(root)
    profile = load_measured_profile(profile_path, model=model, backend=backend)
    measured = profile["measured_fields"]
    if not isinstance(measured, dict) or not isinstance(measured.get("context_budget"), int):
        raise ValueError("robotics benchmark requires a measured context_budget profile field")
    design, tasks = load_design(design_path)
    fixture = load_emulator_fixture(emulator_path)
    _run_id, timestamp = new_run_timestamp()
    final_dir = root / METHOD_NAME / timestamp
    staging = final_dir.with_name(f".{final_dir.name}.staging")
    staging.mkdir(parents=True, exist_ok=False)
    shutil.copy2(design_path, staging / "design.json")
    shutil.copy2(design_path.parent / design.task_ledger, staging / "tasks.jsonl")
    store, store_identity, evidence_by_doc = build_benchmark_store(
        staging,
        hflow_fixture=hflow_fixture,
        manual_corpus=manual_corpus,
        embedding_model=design.embedding_model,
        strategy=design.chunk_strategy,
        chunk_size=design.chunk_size,
        chunk_overlap=design.chunk_overlap,
    )

    def retrieve_off(_task: BenchmarkTask) -> list[dict[str, Any]]:
        return []

    def retrieve_on(task: BenchmarkTask) -> list[dict[str, Any]]:
        return retrieve_context(store, evidence_by_doc, task, design.top_k)

    launcher = OllamaLauncher(
        model,
        num_ctx=int(measured["context_budget"]),
        seed=design.seed,
    )
    launcher.start()
    try:

        def complete(messages: list[ChatMessage]) -> ChatResult:
            return launcher.chat(
                messages,
                max_tokens=design.generation_max_tokens,
                temperature=0.0,
                timeout=design.request_timeout_s,
            )

        off, off_telemetry = _telemetry(
            launcher, fixture, tasks, LANE_NO_RETRIEVAL, retrieve_off, complete
        )
        on, on_telemetry = _telemetry(
            launcher, fixture, tasks, LANE_RETRIEVAL, retrieve_on, complete
        )
    finally:
        launcher.stop()
    reference = [
        evaluate_task(
            fixture,
            task,
            LANE_REFERENCE,
            retrieve=retrieve_on,
            complete=None,
        )
        for task in tasks
    ]
    lane_rows = {
        LANE_NO_RETRIEVAL: off,
        LANE_RETRIEVAL: on,
        LANE_REFERENCE: reference,
    }
    fingerprints = {
        "design": file_digest(design_path),
        "task_ledger": design.task_ledger_sha256,
        "profile": profile["sha256"],
        "corpus": store_identity["corpus_fingerprint"],
        "store": store_identity["store_fingerprint"],
        "model": value_digest({"model": model, "backend": backend}),
        "adapter": value_digest({"adapter": measured.get("adapter")}),
        "drivers": sorted(device.reference.reference_digest for device in fixture.devices),
        "policy": fixture.policy.policy_digest,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": timestamp,
        "verdict": "complete",
        "model": model,
        "backend": backend,
        "design": design.model_dump(mode="json"),
        "profile": profile,
        "fingerprints": fingerprints,
        "store": store_identity,
        "lanes": {lane: aggregate(rows) for lane, rows in lane_rows.items()},
        "telemetry": {
            LANE_NO_RETRIEVAL: off_telemetry,
            LANE_RETRIEVAL: on_telemetry,
        },
        "paired_verdict": paired_verdict(design, off, on),
        "scope": "protocol-neutral-emulator",
    }
    write_bundle(staging, report, lane_rows)
    staging.replace(final_dir)
    return final_dir, report
