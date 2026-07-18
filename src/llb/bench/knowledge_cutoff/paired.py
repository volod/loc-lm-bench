"""Run source-aligned English/Ukrainian knowledge-cutoff lanes."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.bench.common import LLMComplete, Mirror, persist_category_run
from llb.bench.common_backend import ThroughputMeter
from llb.bench.knowledge_cutoff.data import EventSource, LoadedEvents, load_events
from llb.bench.knowledge_cutoff.paired_report import paired_artifacts, paired_statistics
from llb.bench.knowledge_cutoff.run import KnowledgeCutoffRun, run_knowledge_cutoff
from llb.bench.knowledge_cutoff.translation_artifacts import MANIFEST_FILENAME
from llb.bench.knowledge_cutoff.translation_review import (
    REVIEWED_EN_FILENAME,
    REVIEWED_UK_FILENAME,
    REVIEW_SUMMARY_FILENAME,
)
from llb.core.contracts.runs import RunMetrics, RunPaths

METHOD = "knowledge-cutoff-bilingual"


@dataclass(slots=True)
class BilingualCutoffRun:
    english: KnowledgeCutoffRun
    ukrainian: KnowledgeCutoffRun
    paired: dict[str, object]
    report: dict[str, object]
    paths: RunPaths | None


def load_reviewed_lanes(bundle_dir: Path) -> tuple[LoadedEvents, LoadedEvents, dict[str, Any]]:
    summary_path = bundle_dir / REVIEW_SUMMARY_FILENAME
    if not summary_path.is_file():
        raise ValueError("reviewed translation bundle is not frozen; run the freeze command")
    review = json.loads(summary_path.read_text(encoding="utf-8"))
    if not review.get("complete") or not review.get("reviewer"):
        raise ValueError("translation bundle lacks complete bilingual reviewer sign-off")
    manifest = json.loads((bundle_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    english_local = load_events(path=bundle_dir / REVIEWED_EN_FILENAME)
    ukrainian_local = load_events(path=bundle_dir / REVIEWED_UK_FILENAME)
    if [event.id for event in english_local.events] != [
        event.id for event in ukrainian_local.events
    ]:
        raise ValueError("reviewed English and Ukrainian event ids are not aligned")
    for source, translated in zip(english_local.events, ukrainian_local.events, strict=True):
        if source.mcq_answer != translated.mcq_answer or len(source.mcq_choices) != len(
            translated.mcq_choices
        ):
            raise ValueError(f"{source.id}: translated answer identity is not preserved")
    event_source = EventSource(
        "reviewed-translation",
        str(manifest["dataset"]),
        manifest.get("requested_revision"),
        str(manifest["resolved_revision"]),
        "events",
        "train",
        str(manifest["license"]),
    )
    return (
        LoadedEvents(english_local.events, event_source),
        LoadedEvents(ukrainian_local.events, event_source),
        review,
    )


def run_bilingual_cutoff(
    bundle_dir: Path,
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    data_dir: Path | str | None = None,
    threshold: float = 0.5,
    optuna_trials: int = 200,
    seed: int = 42,
    persist: bool = True,
    mirror: Mirror | None = None,
    meter: ThroughputMeter | None = None,
) -> BilingualCutoffRun:
    english_events, ukrainian_events, review = load_reviewed_lanes(bundle_dir)
    english = run_knowledge_cutoff(
        english_events,
        model=model,
        backend=backend,
        complete=complete,
        threshold=threshold,
        optuna_trials=optuna_trials,
        seed=seed,
        persist=False,
        meter=meter,
    )
    ukrainian = run_knowledge_cutoff(
        ukrainian_events,
        model=model,
        backend=backend,
        complete=complete,
        threshold=threshold,
        optuna_trials=optuna_trials,
        seed=seed,
        persist=False,
        meter=meter,
    )
    paired = paired_statistics(english.rows, ukrainian.rows, seed=seed)
    report: dict[str, object] = {
        "schema_version": 1,
        "benchmark": METHOD,
        "model": model,
        "backend": backend,
        "review": review,
        "paired": paired,
        "english": english.report,
        "ukrainian": ukrainian.report,
    }
    paths = None
    if persist and data_dir is not None:
        rows = [
            {**row, "language": language}
            for language, result in (("en", english), ("uk", ukrainian))
            for row in result.rows
        ]
        metrics: RunMetrics = {
            "objective_score": ukrainian.summary.eligible_accuracy,
            "reliability": min(english.summary.parse_rate, ukrainian.summary.parse_rate),
            "tokens_per_s": meter.tokens_per_s if meter is not None else 0.0,
        }
        config: dict[str, Any] = {
            "model": model,
            "backend": backend,
            "category": METHOD,
            "dataset_revision": review["resolved_revision"],
            "translation_reviewer": review["reviewer"],
            "translation_bundle": str(bundle_dir),
            "accepted_rows": review["accepted_rows"],
            "optuna_trials": optuna_trials,
            "seed": seed,
        }
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=METHOD,
            config=config,
            metrics=metrics,
            case_rows=rows,
            mirror=mirror,
            artifacts=paired_artifacts(report),
        )
    return BilingualCutoffRun(english, ukrainian, paired, report, paths)
