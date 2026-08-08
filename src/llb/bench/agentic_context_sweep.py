"""Run the context-policy constant sweep: one axis at a time, everything else held fixed.

Three constants decide what `observation_cap` and `keep_last_n` do:
`DEFAULT_OBSERVATION_CAP_CHARS`, `OBSERVATION_HEAD_SHARE`, and `DEFAULT_KEEP_LAST_N`. This lane
holds the model, the task set, and the policy FIXED within each axis and varies only one constant,
pairs every non-shipped setting against the shipped value over SHARED bootstrap index sets, and
states a pin / expose / inapplicable verdict per axis. Defaults are not rewritten here: a separated
favorable delta is an expose recommendation, not an automatic ship.

The vocabulary (axes, grids, cell records) lives in `agentic_context_sweep_model`; the pairing and
the verdict cut live in `agentic_context_sweep_verdict`.
"""

import logging
from collections.abc import Sequence
from copy import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

from llb.bench.agentic.context import ContextPolicy
from llb.bench.agentic.context_budget import ContextBudget, unbounded_budget
from llb.bench.agentic.model import DEFAULT_MAX_STEPS, AgenticTask
from llb.bench.agentic_context import run_policy, task_set_digest
from llb.bench.agentic_context_report import (
    METRIC_PROMPT_TOKENS,
    PolicyReport,
    policy_config,
    policy_metrics,
)
from llb.bench.agentic_context_sweep_model import (
    AXES,
    METHOD,
    AxisVerdict,
    ConstantSweepRun,
    SettingReport,
    SweepSetting,
    grid_for_axes,
    parse_axes,
)
from llb.bench.agentic_context_sweep_verdict import (
    decide_axis_verdict,
    format_sweep_table,
    pair_against_shipped,
)
from llb.bench.common import LLMComplete, Mirror, persist_category_run, verified_data_config
from llb.bench.common_backend import ThroughputMeter

_LOG = logging.getLogger(__name__)


def _run_setting(
    setting: SweepSetting,
    tasks: list[AgenticTask],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    max_steps: int,
    budget: ContextBudget,
    meter: ThroughputMeter | None,
) -> PolicyReport:
    """Run one swept cell and log what it measured."""
    _LOG.info(
        "[agentic-context-sweep] running %s policy=%s overrides=%s",
        setting.label,
        setting.policy_name,
        setting.overrides,
    )
    report = run_policy(
        tasks,
        ContextPolicy(name=setting.policy_name, **setting.overrides),
        model=model,
        backend=backend,
        complete=complete,
        max_steps=max_steps,
        budget=budget,
    )
    if meter is not None:
        report.result = replace(report.result, tokens_per_s=meter.tokens_per_s)
    _LOG.info(
        "[agentic-context-sweep] done %s completion=%.3f prompt-tok=%.0f overflow=%d",
        setting.label,
        report.result.objective_score,
        report.metric_mean(METRIC_PROMPT_TOKENS),
        report.n_context_overflow,
    )
    return report


def _scored_cells(
    grid: list[SweepSetting],
    tasks: list[AgenticTask],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    max_steps: int,
    budget: ContextBudget,
    meter: ThroughputMeter | None,
) -> list[SettingReport]:
    """Score every cell in the grid, running identical (policy, overrides) cells ONCE.

    The shipped `observation_cap` cell is both the cap-axis and the head-share-axis baseline, so it
    would otherwise be measured twice and the two axes would be read against two different runs of
    the same setting. Each cell still carries its own shallow copy, so persisting one cell's paths
    cannot clobber a peer that shared the batch.
    """
    cells: list[SettingReport] = []
    cache: dict[tuple[str, tuple[tuple[str, Any], ...]], PolicyReport] = {}
    for setting in grid:
        key = (setting.policy_name, tuple(sorted(setting.overrides.items())))
        report = cache.get(key)
        if report is None:
            report = _run_setting(
                setting,
                tasks,
                model=model,
                backend=backend,
                complete=complete,
                max_steps=max_steps,
                budget=budget,
                meter=meter,
            )
            cache[key] = report
        else:
            _LOG.info("[agentic-context-sweep] reuse %s (identical to a prior cell)", setting.label)
        cells.append(SettingReport(setting=setting, report=copy(report), paired={}))
    return cells


def run_constant_sweep(
    tasks: list[AgenticTask],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    settings: list[SweepSetting] | None = None,
    axes: Sequence[str] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    budget: ContextBudget | None = None,
    data_dir: Path | str | None = None,
    persist: bool = True,
    mirror: Mirror | None = None,
    data_verified: bool = False,
    verification_ref: str | None = None,
    meter: ThroughputMeter | None = None,
) -> ConstantSweepRun:
    """Walk the constant grids, pair against shipped defaults, and cut per-axis verdicts."""
    if not tasks:
        raise SystemExit("no agentic tasks provided")
    budget = budget if budget is not None else unbounded_budget()
    grid = settings if settings is not None else grid_for_axes(parse_axes(axes))
    present_axes = tuple(dict.fromkeys(setting.axis for setting in grid))
    digest = task_set_digest(tasks)
    cells = _scored_cells(
        grid,
        tasks,
        model=model,
        backend=backend,
        complete=complete,
        max_steps=max_steps,
        budget=budget,
        meter=meter,
    )
    pair_against_shipped(cells)
    verdicts = [
        decide_axis_verdict(axis, [cell for cell in cells if cell.setting.axis == axis])
        for axis in present_axes
    ]
    table = format_sweep_table(cells, verdicts)
    if persist and data_dir is not None:
        _persist(
            cells,
            verdicts=verdicts,
            data_dir=data_dir,
            model=model,
            backend=backend,
            digest=digest,
            max_steps=max_steps,
            max_prompt_chars=budget.max_prompt_chars,
            verification_cfg=verified_data_config(
                data_verified=data_verified, verification_ref=verification_ref
            ),
            mirror=mirror,
            budget_provenance=budget.provenance(),
            table=table,
            axes=present_axes,
        )
    return ConstantSweepRun(
        model=model,
        backend=backend,
        settings=cells,
        verdicts=verdicts,
        table=table,
        task_set_digest=digest,
        max_prompt_chars=budget.max_prompt_chars,
    )


def _persist(
    cells: list[SettingReport],
    *,
    verdicts: list[AxisVerdict],
    data_dir: Path | str,
    model: str,
    backend: str,
    digest: str,
    max_steps: int,
    max_prompt_chars: int,
    verification_cfg: dict[str, object],
    mirror: Mirror | None,
    budget_provenance: dict[str, Any] | None,
    table: str,
    axes: Sequence[str] | None = None,
) -> None:
    """Persist one bundle per setting plus a sweep-summary manifest under the method root."""
    for cell in cells:
        config = {
            **policy_config(
                cell.report,
                model=model,
                backend=backend,
                task_digest=digest,
                policies=[cell.setting.policy_name],
                max_steps=max_steps,
                max_prompt_chars=max_prompt_chars,
                policy_settings={
                    **cell.setting.overrides,
                    "sweep_axis": cell.setting.axis,
                    "sweep_label": cell.setting.label,
                    "sweep_shipped": cell.setting.is_shipped,
                    "paired_vs_shipped": cell.paired or None,
                },
                budget_provenance=budget_provenance,
            ),
            **verification_cfg,
            "category": METHOD,
        }
        cell.report.paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=f"agentic-context-sweep-{cell.setting.label}",
            config=config,
            metrics=policy_metrics(cell.report),
            case_rows=cell.report.rows,
            mirror=mirror,
        )
    summary = {
        "model": model,
        "backend": backend,
        "category": METHOD,
        "task_set_digest": digest,
        "max_steps": max_steps,
        "max_prompt_chars": max_prompt_chars,
        "axes": list(axes) if axes is not None else list(AXES),
        "verdicts": [
            {
                "axis": v.axis,
                "shipped_value": v.shipped_value,
                "verdict": v.verdict,
                "reason": v.reason,
            }
            for v in verdicts
        ],
        "table": table,
        **(budget_provenance or {}),
        **verification_cfg,
    }
    persist_category_run(
        method=METHOD,
        data_dir=data_dir,
        run_name="agentic-context-sweep-summary",
        config=summary,
        metrics={"objective_score": 0.0, "reliability": 1.0, "tokens_per_s": 0.0},
        case_rows=[],
        mirror=mirror,
    )
