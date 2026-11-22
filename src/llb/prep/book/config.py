"""The run request, its preflight, and the `config.json` a resume reads back.

`preflight` is the whole of `--dry-run`: it opens the source, parses the page ranges, checks every
configured stage (registered, its inputs configured before it, its parameters, its model present)
and the `--reuse-from` run, and returns every problem at once rather than the first. A real run calls
the same function and refuses to start a stage when it reports anything.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llb.core.contracts.data_prep.book import BookOcrRunConfig, BookSourceRecord, BookStageSpec
from llb.core.fsutil import atomic_write_text
from llb.prep.book.constants import (
    CONFIG_FILENAME,
    CONTRACT_VERSION,
    DEFAULT_BATCH_PAGES,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_PAGES,
    DEFAULT_SOURCE_POLICY,
    DEFAULT_STAGES,
    RUN_CONFIG_SCHEMA_ID,
    SOURCE_POLICIES,
)
from llb.prep.book.inventory import PdfOpenError, inventory_document
from llb.prep.book.models import OllamaLister, model_problem
from llb.prep.book.ranges import PageRangeError, batches, parse_page_ranges
from llb.prep.book.source import sha256_file
from llb.prep.book.stages import StageRegistry, resolved_inputs
from llb.prep.pdf.model import PdfDiagnostics

_FILE_KEYS = ("pages", "batch_pages", "max_attempts", "source_policy", "stages")


@dataclass(frozen=True)
class RunRequest:
    """What the operator asked for, before anything is validated."""

    source: Path
    pages: str = DEFAULT_PAGES
    batch_pages: int = DEFAULT_BATCH_PAGES
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    source_policy: str = DEFAULT_SOURCE_POLICY
    stages: tuple[BookStageSpec, ...] = tuple(BookStageSpec(name=n) for n in DEFAULT_STAGES)
    reuse_from: Path | None = None


@dataclass(frozen=True)
class Preflight:
    """The validated plan, or the reasons there is none."""

    errors: list[str] = field(default_factory=list)
    sha256: str | None = None
    page_count: int | None = None
    pages: list[int] = field(default_factory=list)
    batches: list[list[int]] = field(default_factory=list)
    diagnostics: PdfDiagnostics | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def stage_specs(raw: Sequence[Any]) -> tuple[BookStageSpec, ...]:
    """Stage specs from a config file: a bare name or a `{name, params, model}` mapping each."""
    specs = []
    for item in raw:
        if isinstance(item, str):
            specs.append(BookStageSpec(name=item))
        elif isinstance(item, dict):
            specs.append(BookStageSpec.model_validate(item))
        else:
            raise ValueError(f"stage entry {item!r} is neither a name nor a mapping")
    return tuple(specs)


def load_request_file(path: Path) -> dict[str, Any]:
    """The run-shaping keys of a YAML or JSON config file, refusing any key it does not know."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: a book-ocr config is a mapping")
    unknown = sorted(set(payload) - set(_FILE_KEYS))
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}; allowed: {list(_FILE_KEYS)}")
    if "stages" in payload:
        payload["stages"] = stage_specs(payload["stages"])
    return payload


def _one_stage_errors(
    spec: BookStageSpec,
    registry: StageRegistry,
    configured_before: Sequence[str],
    ollama_lister: OllamaLister | None,
) -> list[str]:
    """Problems with one registered stage, given the stages configured ahead of it."""
    stage = registry.get(spec.name)
    errors = []
    if spec.name in configured_before:
        errors.append(f"stage {spec.name!r} is configured twice")
    needed = resolved_inputs(stage, configured_before, spec.params)
    missing = [name for name in needed if name not in configured_before]
    if missing:
        errors.append(f"stage {spec.name!r} reads {missing}, which must be configured before it")
    extra = getattr(stage, "validate_configured", None)
    if extra is not None:
        errors.extend(extra(configured_before))
    order = getattr(stage, "validate_order", None)
    if order is not None:
        errors.extend(order(spec.params, configured_before))
    errors.extend(stage.validate_params(spec.params))
    if stage.model_required and not spec.model:
        errors.append(f"stage {spec.name!r} needs a model")
    problem = model_problem(spec.model, ollama_lister) if spec.model else None
    if problem:
        errors.append(f"stage {spec.name!r}: {problem}")
    return errors


def stage_errors(
    specs: Sequence[BookStageSpec], registry: StageRegistry, ollama_lister: OllamaLister | None
) -> list[str]:
    """Every problem with the configured stage list: names, order, parameters, and models."""
    errors: list[str] = []
    seen: list[str] = []
    for spec in specs:
        if spec.name not in registry:
            errors.append(f"unknown stage {spec.name!r}; registered: {registry.names()}")
            continue
        errors.extend(_one_stage_errors(spec, registry, seen, ollama_lister))
        seen.append(spec.name)
    return errors


def _option_errors(request: RunRequest) -> list[str]:
    errors = []
    if request.source_policy not in SOURCE_POLICIES:
        errors.append(
            f"source policy must be one of {SOURCE_POLICIES}, got {request.source_policy!r}"
        )
    if request.batch_pages < 1:
        errors.append(f"--batch-pages must be at least 1, got {request.batch_pages}")
    if request.max_attempts < 1:
        errors.append(f"--max-attempts must be at least 1, got {request.max_attempts}")
    if request.reuse_from is not None and not (request.reuse_from / CONFIG_FILENAME).is_file():
        errors.append(
            f"--reuse-from {request.reuse_from} is not a book-ocr run (no {CONFIG_FILENAME})"
        )
    return errors


def preflight(
    request: RunRequest, registry: StageRegistry, ollama_lister: OllamaLister | None = None
) -> Preflight:
    """Validate the source, ranges, options, stages, and models without running a stage."""
    errors = _option_errors(request) + stage_errors(request.stages, registry, ollama_lister)
    if not request.source.is_file():
        return Preflight(errors=[f"source PDF {request.source} does not exist", *errors])
    try:
        document = inventory_document(request.source)
    except PdfOpenError as exc:
        return Preflight(errors=[str(exc), *errors])
    pages: list[int] = []
    try:
        pages = parse_page_ranges(request.pages, document.page_count)
    except PageRangeError as exc:
        errors.append(str(exc))
    plan = batches(pages, request.batch_pages) if pages and request.batch_pages >= 1 else []
    return Preflight(
        errors=errors,
        sha256=sha256_file(request.source),
        page_count=document.page_count,
        pages=pages,
        batches=plan,
        diagnostics=document.diagnostics,
    )


def build_config(
    request: RunRequest, plan: Preflight, source: BookSourceRecord, run_id: str
) -> BookOcrRunConfig:
    return BookOcrRunConfig(
        schema_id=RUN_CONFIG_SCHEMA_ID,
        schema_version=CONTRACT_VERSION,
        kind="book-ocr-run-config",
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=source,
        pages_spec=request.pages,
        pages=plan.pages,
        batch_pages=request.batch_pages,
        max_attempts=request.max_attempts,
        stages=list(request.stages),
        reuse_from=str(request.reuse_from.resolve()) if request.reuse_from else None,
    )


def write_config(run_dir: Path, config: BookOcrRunConfig) -> None:
    atomic_write_text(
        run_dir / CONFIG_FILENAME,
        json.dumps(config.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
    )


def read_config(run_dir: Path) -> BookOcrRunConfig:
    """The recorded configuration of `run_dir`, refusing a directory that is not a run."""
    path = run_dir / CONFIG_FILENAME
    if not path.is_file():
        raise ValueError(f"{run_dir} is not a book-ocr run (no {CONFIG_FILENAME})")
    try:
        return BookOcrRunConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ValueError(f"{path} is not a readable book-ocr run config: {exc}") from exc
