"""Read the resume and abort records an interrupted run left behind."""

from pathlib import Path

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.io import json_document
from llb.artifacts.registry import ContractRegistry
from llb.core.contracts.run_bundle.journals import (
    CASE_PROGRESS_META_SCHEMA_ID,
    JUDGE_BUDGET_ABORT_SCHEMA_ID,
    CaseProgressMeta,
    JudgeBudgetAbort,
)


def read_progress_meta(
    path: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> CaseProgressMeta:
    """The pinned identity a `--resume` compares itself against, current or pre-contract."""
    path = Path(path)
    read = registry.read_as(CASE_PROGRESS_META_SCHEMA_ID, json_document(path), source=str(path))
    assert isinstance(read, CaseProgressMeta)
    return read


def read_budget_abort(
    path: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> JudgeBudgetAbort:
    """The record a judge writes when it stops on its declared spend or call budget."""
    path = Path(path)
    read = registry.read_as(JUDGE_BUDGET_ABORT_SCHEMA_ID, json_document(path), source=str(path))
    assert isinstance(read, JudgeBudgetAbort)
    return read
