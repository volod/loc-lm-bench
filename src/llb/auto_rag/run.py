"""Resumable top-level auto-RAG stage machine."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from llb.auto_rag.journal import AutoRagJournal
from llb.auto_rag.models import STAGES, AutoRagSettings, AutoRagStatus
from llb.auto_rag.stages import DEFAULT_STAGES
from llb.auto_rag.verification import VerificationPending

Stage = Callable[[AutoRagSettings, dict[str, dict[str, Any]]], dict[str, Any]]
AfterStage = Callable[[str, dict[str, Any]], None]


class AutoRagPaused(RuntimeError):
    """A human gate has durable pending work and the run can be resumed later."""


def run_auto_rag(
    settings: AutoRagSettings,
    *,
    stages: Mapping[str, Stage] | None = None,
    after_stage: AfterStage | None = None,
) -> AutoRagStatus:
    """Run or resume all stages, publishing one atomic completion marker per boundary."""
    journal = AutoRagJournal(settings.run_dir, {"settings": settings.manifest_payload()})
    resumed = journal.open()
    runners = {**DEFAULT_STAGES, **(stages or {})}
    outputs: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        prior = journal.load(stage)
        if prior is not None:
            outputs[stage] = prior
            continue
        journal.event(stage, "started")
        try:
            result = runners[stage](settings, outputs)
        except VerificationPending as exc:
            journal.event(stage, "paused", reason=str(exc))
            raise AutoRagPaused(str(exc)) from exc
        except BaseException as exc:
            journal.event(stage, "failed", error_type=type(exc).__name__, reason=str(exc))
            raise
        journal.complete(stage, result)
        outputs[stage] = result
        if after_stage is not None:
            after_stage(stage, result)
    journal.event("run", "completed")
    recommendation = outputs["recommendation"]
    return AutoRagStatus(
        run_dir=settings.run_dir,
        completed=tuple(STAGES),
        recommendation=_optional_path(recommendation.get("recommendation")),
        report=_optional_path(recommendation.get("report")),
        resumed=resumed,
    )


def _optional_path(value: object) -> Path | None:
    return Path(value) if isinstance(value, str) and value else None
