"""Project-native runner and canonical MLOps persistence."""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llb.bench.common import (
    LLMComplete,
    Mirror,
    ThroughputMeter,
    complete_all,
    persist_category_run,
)
from llb.bench.knowledge_cutoff.data import UPSTREAM_PROJECT, LoadedEvents
from llb.bench.knowledge_cutoff.fit import DEFAULT_SEED, DEFAULT_TRIALS, DecayFit, fit_decay
from llb.bench.knowledge_cutoff.probe import parse_answer, prepare_probe
from llb.bench.knowledge_cutoff.report import build_report, report_artifacts
from llb.bench.knowledge_cutoff.score import DEFAULT_THRESHOLD, CutoffSummary, case_row, summarize
from llb.core.contracts import RunMetrics, RunPaths

METHOD = "knowledge-cutoff"
_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class KnowledgeCutoffRun:
    summary: CutoffSummary
    fit: DecayFit
    rows: list[dict[str, Any]]
    report: dict[str, object]
    paths: RunPaths | None
    report_json: str | None
    report_markdown: str | None


def _source_dict(loaded: LoadedEvents) -> dict[str, object]:
    source = asdict(loaded.source)
    source["upstream_project"] = UPSTREAM_PROJECT
    return source


def run_knowledge_cutoff(
    loaded: LoadedEvents,
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    data_dir: Path | str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    optuna_trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    run_name: str = METHOD,
    persist: bool = True,
    mirror: Mirror | None = None,
    meter: ThroughputMeter | None = None,
) -> KnowledgeCutoffRun:
    if not loaded.events:
        raise ValueError("no knowledge-cutoff events provided")
    probes = [prepare_probe(event) for event in loaded.events]
    outputs = complete_all(
        complete,
        [probe.prompt for probe in probes],
        label=METHOD,
        logger=_LOG,
    )
    rows = [
        case_row(
            event,
            response=output,
            selected=parse_answer(output),
            expected=probe.expected,
            choice_order=probe.choice_order,
        )
        for event, probe, output in zip(loaded.events, probes, outputs, strict=True)
    ]
    summary = summarize(rows, threshold=threshold)
    fit = fit_decay(summary, trials=optuna_trials, seed=seed)
    source = _source_dict(loaded)
    report = build_report(
        model=model,
        backend=backend,
        source=source,
        summary=summary.to_dict(),
        fit=fit.to_dict(),
        n_events=len(rows),
    )

    paths: RunPaths | None = None
    report_json = report_markdown = None
    if persist and data_dir is not None:
        tokens_per_s = meter.tokens_per_s if meter is not None else 0.0
        metrics: RunMetrics = {
            "objective_score": summary.eligible_accuracy,
            "reliability": summary.parse_rate,
            "tokens_per_s": tokens_per_s,
        }
        config: dict[str, Any] = {
            "model": model,
            "backend": backend,
            "category": METHOD,
            "probe": "position-balanced-mcq",
            "threshold": threshold,
            "optuna_trials": optuna_trials,
            "optuna_seed": seed,
            "effective_cutoff": fit.effective_cutoff,
            "cutoff_ordinal": fit.cutoff_ordinal,
            "fit_nll": fit.negative_log_likelihood,
            "fit_scale_months": fit.scale_months,
            "fit_ceiling": fit.ceiling,
            "dataset_id": loaded.source.identity,
            "dataset_revision": loaded.source.resolved_revision,
            "dataset_license": loaded.source.license,
            "n_eligible": sum(point.n for point in summary.curve),
        }
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=run_name,
            config=config,
            metrics=metrics,
            case_rows=rows,
            mirror=mirror,
            artifacts=report_artifacts(report),
        )
        out_dir = Path(paths["manifest"]).parent
        report_json = str(out_dir / "report.json")
        report_markdown = str(out_dir / "report.md")
        _LOG.info("[%s] %s cutoff=%s -> %s", METHOD, model, fit.effective_cutoff, report_markdown)
    return KnowledgeCutoffRun(summary, fit, rows, report, paths, report_json, report_markdown)
