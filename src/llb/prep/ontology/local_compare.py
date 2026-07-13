"""Sequential exact-seed comparison of two local Ollama drafters."""

from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path

from llb.core.paths import resolve_data_dir
from llb.prep.frontier_telemetry import LLMComplete
from llb.prep.ontology.compare_artifacts import (
    draft_shared_seeds,
    lane_report,
    verification_report,
    write_comparison_report,
)
from llb.prep.ontology.endpoint_config import EndpointCompleters, EndpointConfig, EndpointPlan
from llb.prep.ontology.ollama_lifecycle import unload_models
from llb.prep.ontology.pipeline.run import draft_goldset

LOCAL_COMPARE_METHOD_DIR = "draft-compare-local"
BASELINE_LANE = "baseline"
PROBE_LANE = "probe"


def default_local_compare_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return resolve_data_dir() / LOCAL_COMPARE_METHOD_DIR / stamp


def compare_local_drafters(
    corpus_root: Path | str,
    baseline_endpoint: EndpointConfig,
    probe_endpoint: EndpointConfig,
    *,
    seeds: int,
    seed: int = 13,
    out_dir: Path | str | None = None,
    resource_selection: dict[str, object] | None = None,
    baseline_completers: EndpointCompleters | None = None,
    probe_complete: LLMComplete | None = None,
    unload: Callable[[str, list[str] | None], list[str]] = unload_models,
) -> dict[str, object]:
    """Run baseline then probe locally, unloading Ollama models at every boundary."""
    for endpoint in (baseline_endpoint, probe_endpoint):
        if endpoint.kind != "local" or endpoint.backend != "ollama":
            raise ValueError("sequential local comparison requires local Ollama endpoints")
    if baseline_endpoint.base_url != probe_endpoint.base_url:
        raise ValueError("sequential local comparison requires one shared Ollama host")

    root = Path(out_dir) if out_dir is not None else default_local_compare_dir()
    unload(baseline_endpoint.base_url, None)
    try:
        baseline = draft_goldset(
            corpus_root,
            EndpointPlan.single(baseline_endpoint),
            completers=baseline_completers,
            max_items=seeds,
            seed=seed,
            out_dir=root / BASELINE_LANE,
        )
        unload(baseline_endpoint.base_url, [baseline_endpoint.model])
        probe = draft_shared_seeds(
            baseline,
            probe_endpoint,
            baseline_endpoint,
            root / PROBE_LANE,
            seed,
            PROBE_LANE,
            complete=probe_complete,
        )
    finally:
        unload(baseline_endpoint.base_url, None)

    sample_n = min(seeds, max(len(baseline.items), len(probe.items)))
    lanes = {
        BASELINE_LANE: lane_report(
            baseline,
            verification_report(root / BASELINE_LANE, None, sample_n, seed),
        ),
        PROBE_LANE: lane_report(
            probe,
            verification_report(root / PROBE_LANE, None, sample_n, seed),
        ),
    }
    return write_comparison_report(
        root,
        corpus_root,
        baseline,
        lanes,
        kind="local-draft-comparison",
        lane_order=[BASELINE_LANE, PROBE_LANE],
        execution={
            "mode": "sequential-local-ollama",
            "model_order": [baseline_endpoint.model, probe_endpoint.model],
            "unload_between_lanes": True,
            "shared_extraction_model": baseline_endpoint.model,
            "resource_selection": resource_selection or {},
        },
    )
