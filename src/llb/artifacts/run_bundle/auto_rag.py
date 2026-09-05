"""Read the auto-RAG orchestration trail through its registered contracts."""

from pathlib import Path

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.io import json_document
from llb.artifacts.registry import ContractRegistry
from llb.core.contracts.run_bundle.auto_rag import (
    AUTO_RAG_MANIFEST_SCHEMA_ID,
    AUTO_RAG_STAGE_RESULT_SCHEMA_ID,
    RAG_RECOMMENDATION_SCHEMA_ID,
    AutoRagManifestDocument,
    AutoRagStageResult,
    RagRecommendationDocument,
)


def read_auto_rag_manifest(
    path: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> AutoRagManifestDocument:
    """The pinned settings a resume must still match, current or pre-contract."""
    path = Path(path)
    read = registry.read_as(AUTO_RAG_MANIFEST_SCHEMA_ID, json_document(path), source=str(path))
    assert isinstance(read, AutoRagManifestDocument)
    return read


def read_stage_result(
    path: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> AutoRagStageResult:
    """One durably completed stage, read at the current contract version."""
    path = Path(path)
    read = registry.read_as(AUTO_RAG_STAGE_RESULT_SCHEMA_ID, json_document(path), source=str(path))
    assert isinstance(read, AutoRagStageResult)
    return read


def read_recommendation(
    path: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> RagRecommendationDocument:
    """The selected configuration, read from the YAML an operator copies from."""
    import yaml

    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected one recommendation record")
    read = registry.read_as(RAG_RECOMMENDATION_SCHEMA_ID, payload, source=str(path))
    assert isinstance(read, RagRecommendationDocument)
    return read
