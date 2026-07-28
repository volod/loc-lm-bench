"""Experiment-derived acceptance planning for reviewed chain fixtures."""

import json
import math
from pathlib import Path
from typing import Any

DEFAULT_MIN_RETENTION_FRACTION = 0.50
SAMPLE_MANIFEST_FILE = "sample_manifest.json"


def retention_gate_plan(
    bundle: Path,
    *,
    min_chains: int | None,
    min_retention_fraction: float,
) -> dict[str, Any]:
    """Derive the promotion minimum from the number of reviewed sample rows."""
    if not 0.0 < min_retention_fraction <= 1.0:
        raise ValueError("minimum retention fraction must be between zero and one")
    sample_path = bundle / SAMPLE_MANIFEST_FILE
    reviewed_n: int | None = None
    if sample_path.is_file():
        try:
            payload = json.loads(sample_path.read_text(encoding="utf-8"))
            value = payload.get("sample_size")
            if isinstance(value, int) and value > 0:
                reviewed_n = value
        except (OSError, json.JSONDecodeError):
            reviewed_n = None
    derived = math.ceil(reviewed_n * min_retention_fraction) if reviewed_n is not None else None
    if min_chains is not None and min_chains < 1:
        raise ValueError("min_chains must be at least 1")
    if min_chains is None and derived is None:
        raise ValueError(
            "cannot derive the accepted-chain minimum without sample_manifest.json; "
            "pass --min-chains as an explicit override"
        )
    selected = min_chains if min_chains is not None else derived
    return {
        "gate_id": "chain-review-retention",
        "classification": "inferential_gate",
        "method": "relative-reviewed-sample-retention",
        "assumptions": {
            "reviewed_sample_size": reviewed_n,
            "minimum_retention_fraction": min_retention_fraction,
        },
        "derived_target": derived,
        "operator_override": min_chains,
        "selected_target": selected,
        "override_meets_derived_target": (
            min_chains is None or derived is None or min_chains >= derived
        ),
    }
