"""Read the board's miss analysis through its registered contracts."""

from pathlib import Path

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.io import json_document
from llb.artifacts.registry import ContractRegistry
from llb.core.contracts.run_bundle.board import MISS_ANALYSIS_SCHEMA_ID, MissAnalysisDocument


def read_miss_analysis(
    path: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> MissAnalysisDocument:
    """One `analysis.json`, current or pre-contract, at the current contract version."""
    path = Path(path)
    read = registry.read_as(MISS_ANALYSIS_SCHEMA_ID, json_document(path), source=str(path))
    assert isinstance(read, MissAnalysisDocument)
    return read
