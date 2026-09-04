"""Publish and read the retrieval comparison and routing-calibration sidecars.

These two are the artifacts an adoption decision is re-read from months later, and they are the
largest records this project writes: a comparison carries every lane, every slice, and every
item's paired draw. So the sidecar keeps the exact payload its renderers already read and gains
only its identity -- and the check is run over the BYTES about to be written, not over the
in-memory mapping, because what a later reader gets back is the JSON and nothing else.
"""

import json
from pathlib import Path

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.io import json_document
from llb.artifacts.registry import ContractRegistry
from llb.core.contracts.common import JsonObject
from llb.core.contracts.retrieval_graph.calibration import RoutingCalibrationReport
from llb.core.contracts.retrieval_graph.comparison import RetrievalComparisonReport


def write_sidecar(
    path: Path, schema_id: str, report: JsonObject, registry: ContractRegistry = DEFAULT_REGISTRY
) -> None:
    """Validate the encoded sidecar against its contract, then publish it with its identity."""
    definition = registry.definition(schema_id)
    identified = {
        "schema_id": schema_id,
        "schema_version": definition.current_version,
        **{
            key: value
            for key, value in report.items()
            if key not in {"schema_id", "schema_version"}
        },
    }
    text = json.dumps(identified, ensure_ascii=False, indent=2)
    registry.read_as(schema_id, json.loads(text), source=str(path))
    path.write_text(text, encoding="utf-8")


def read_sidecar(
    path: Path, schema_id: str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> RetrievalComparisonReport | RoutingCalibrationReport:
    """Read one sidecar, current or pre-contract, at the current contract version."""
    read = registry.read_as(schema_id, json_document(path), source=str(path))
    assert isinstance(read, (RetrievalComparisonReport, RoutingCalibrationReport))
    return read
