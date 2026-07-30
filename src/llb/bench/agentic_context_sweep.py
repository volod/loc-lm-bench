"""Constant sweep for agent context-policy knobs -- pin or expose each shipped default.

Three constants decide what `observation_cap` and `keep_last_n` do:
`DEFAULT_OBSERVATION_CAP_CHARS`, `OBSERVATION_HEAD_SHARE`, and `DEFAULT_KEEP_LAST_N`. This lane
holds the model, the task set, and the policy FIXED within each axis and varies only one constant,
pairs every non-shipped setting against the shipped value over SHARED bootstrap index sets, and
states a pin / expose / inapplicable verdict per axis. Defaults are not rewritten here: a separated
favorable delta is an expose recommendation, not an automatic ship.
"""

import logging
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from llb.bench.agentic.context import (
    DEFAULT_KEEP_LAST_N,
    DEFAULT_OBSERVATION_CAP_CHARS,
    OBSERVATION_HEAD_SHARE,
    POLICY_KEEP_LAST_N,
    POLICY_OBSERVATION_CAP,
    ContextPolicy,
)
from llb.bench.agentic.context_budget import ContextBudget, unbounded_budget
from llb.bench.agentic.model import DEFAULT_MAX_STEPS, AgenticTask
from llb.bench.agentic_context import run_policy, task_set_digest
from llb.bench.agentic_context_report import (
    METRIC_COMPLETION,
    METRIC_PROMPT_TOKENS,
    METRICS,
    PolicyReport,
    policy_config,
    policy_metrics,
)
from llb.bench.common import LLMComplete, Mirror, persist_category_run, verified_data_config
from llb.bench.common_backend import ThroughputMeter
from llb.rag.fusion_evidence.evidence_gate import (
    DEFAULT_CONFIDENCE,
    READING_FLAT,
    READING_SEPARATED,
    apply_evidence_gate,
)
from llb.rag.fusion_evidence.paired import PairedComparison, paired_comparison
from llb.rag.fusion_evidence.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_index_sets

_LOG = logging.getLogger(__name__)

METHOD = "agentic-context-sweep"

# Measured grids. Cap and head-share ride under `observation_cap`; keep-last-n under its own policy.
CAP_GRID: tuple[int, ...] = (400, 800, 1600)
HEAD_SHARE_GRID: tuple[float, ...] = (0.5, 0.6, 0.7)
KEEP_LAST_N_GRID: tuple[int, ...] = (1, 2, 3)
# Long-transcript keep grid: same cells, but the lane runs them alone at a higher max_steps over
# medium-observation pipeline tasks (see `agentic_long_transcript`).
KEEP_LONG_TRANSCRIPT_GRID: tuple[int, ...] = KEEP_LAST_N_GRID

AXIS_CAP = "observation_cap_chars"
AXIS_HEAD = "observation_head_share"
AXIS_KEEP = "keep_last_n"
AXES: tuple[str, ...] = (AXIS_CAP, AXIS_HEAD, AXIS_KEEP)

VERDICT_PIN = "pin"
VERDICT_EXPOSE = "expose"
VERDICT_INAPPLICABLE = "inapplicable"


@dataclass(frozen=True, slots=True)
class SweepSetting:
    """One cell of the constant grid: which axis, which policy, and the full override map."""

    axis: str
    label: str
    policy_name: str
    overrides: dict[str, Any]
    is_shipped: bool


@dataclass(slots=True)
class SettingReport:
    """One setting's scored outcome plus its paired delta against the shipped cell of its axis."""

    setting: SweepSetting
    report: PolicyReport
    paired: dict[str, PairedComparison]


@dataclass(slots=True)
class AxisVerdict:
    """Pin / expose / inapplicable for one constant, with the evidence sentence."""

    axis: str
    shipped_value: Any
    verdict: str
    reason: str


@dataclass(slots=True)
class ConstantSweepRun:
    """Outcome of one three-axis constant sweep for a fixed model and task set."""

    model: str
    backend: str
    settings: list[SettingReport]
    verdicts: list[AxisVerdict]
    table: str
    task_set_digest: str
    max_prompt_chars: int


def shipped_value(axis: str) -> Any:
    """The currently shipped default for one axis."""
    return {
        AXIS_CAP: DEFAULT_OBSERVATION_CAP_CHARS,
        AXIS_HEAD: OBSERVATION_HEAD_SHARE,
        AXIS_KEEP: DEFAULT_KEEP_LAST_N,
    }[axis]


def default_grid() -> list[SweepSetting]:
    """The three one-dimensional grids the CUDA evidence run walks."""
    return grid_for_axes(AXES)


def keep_long_transcript_grid() -> list[SweepSetting]:
    """Keep-only grid for the long-transcript lane (keep=1/2/3 under `keep_last_n`)."""
    return grid_for_axes((AXIS_KEEP,), keep_values=KEEP_LONG_TRANSCRIPT_GRID)


def grid_for_axes(
    axes: Sequence[str],
    *,
    keep_values: Sequence[int] | None = None,
) -> list[SweepSetting]:
    """Build the requested one-dimensional grids; unknown axis names raise."""
    unknown = [a for a in axes if a not in AXES]
    if unknown:
        raise SystemExit(f"unknown sweep axes: {unknown}; choose from {AXES}")
    keep_values = tuple(keep_values) if keep_values is not None else KEEP_LAST_N_GRID
    settings: list[SweepSetting] = []
    if AXIS_CAP in axes:
        for cap in CAP_GRID:
            settings.append(
                SweepSetting(
                    axis=AXIS_CAP,
                    label=f"cap={cap}",
                    policy_name=POLICY_OBSERVATION_CAP,
                    overrides={
                        "observation_cap_chars": cap,
                        "observation_head_share": OBSERVATION_HEAD_SHARE,
                        "keep_last_n": DEFAULT_KEEP_LAST_N,
                    },
                    is_shipped=cap == DEFAULT_OBSERVATION_CAP_CHARS,
                )
            )
    if AXIS_HEAD in axes:
        for share in HEAD_SHARE_GRID:
            settings.append(
                SweepSetting(
                    axis=AXIS_HEAD,
                    label=f"head={share}",
                    policy_name=POLICY_OBSERVATION_CAP,
                    overrides={
                        "observation_cap_chars": DEFAULT_OBSERVATION_CAP_CHARS,
                        "observation_head_share": share,
                        "keep_last_n": DEFAULT_KEEP_LAST_N,
                    },
                    is_shipped=share == OBSERVATION_HEAD_SHARE,
                )
            )
    if AXIS_KEEP in axes:
        for keep in keep_values:
            settings.append(
                SweepSetting(
                    axis=AXIS_KEEP,
                    label=f"keep={keep}",
                    policy_name=POLICY_KEEP_LAST_N,
                    overrides={
                        "observation_cap_chars": DEFAULT_OBSERVATION_CAP_CHARS,
                        "observation_head_share": OBSERVATION_HEAD_SHARE,
                        "keep_last_n": keep,
                    },
                    is_shipped=keep == DEFAULT_KEEP_LAST_N,
                )
            )
    return settings


def parse_axes(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse a comma-separated axes string (or pass-through a sequence) into known axis names."""
    if raw is None:
        return AXES
    if isinstance(raw, str):
        parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    else:
        parts = tuple(raw)
    return parts if parts else AXES


def pair_against_shipped(
    settings: list[SettingReport],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> None:
    """Attach each non-shipped cell's paired deltas against its axis's shipped cell, in place."""
    for axis in AXES:
        cells = [s for s in settings if s.setting.axis == axis]
        baseline = next((c for c in cells if c.setting.is_shipped), None)
        if baseline is None:
            continue
        n_items = len(baseline.report.case_success)
        index_sets = bootstrap_index_sets(n_items, resamples, seed)
        for cell in cells:
            if cell.setting.is_shipped or len(cell.report.case_success) != n_items:
                cell.paired = {}
                continue
            cell.paired = {
                metric: paired_comparison(
                    cell.report.vector(metric),
                    baseline.report.vector(metric),
                    index_sets,
                    confidence,
                )
                for metric in METRICS
            }


def _metric_reading(paired: PairedComparison | None) -> str:
    if paired is None:
        return READING_FLAT
    delta = paired["delta"]
    reading = READING_SEPARATED if delta["lo"] > 0.0 or delta["hi"] < 0.0 else READING_FLAT
    return apply_evidence_gate(
        reading,
        discordant=paired["wins"] + paired["losses"],
        confidence=DEFAULT_CONFIDENCE,
    )


def _delta_cell(paired: PairedComparison | None) -> str:
    if paired is None:
        return "baseline"
    delta = paired["delta"]
    return f"{delta['mean']:+.3f} [{delta['lo']:+.3f}, {delta['hi']:+.3f}]"


def decide_axis_verdict(axis: str, cells: Sequence[SettingReport]) -> AxisVerdict:
    """Cut pin / expose / inapplicable for one axis from its paired cells."""
    shipped = shipped_value(axis)
    baseline = next((c for c in cells if c.setting.is_shipped), None)
    if baseline is None:
        return AxisVerdict(axis, shipped, VERDICT_EXPOSE, "shipped cell missing from the grid")

    # keep_last_n is for long transcripts: if every cell overflows identically and completion is
    # flat, the single fat observation that blew the prompt sits inside every kept window.
    if axis == AXIS_KEEP:
        overflows = {c.report.n_context_overflow for c in cells}
        completions = {round(c.report.result.objective_score, 6) for c in cells}
        keep_labels = tuple(c.setting.label for c in cells)
        if len(overflows) == 1 and len(completions) == 1 and baseline.report.n_context_overflow > 0:
            return AxisVerdict(
                axis,
                shipped,
                VERDICT_INAPPLICABLE,
                (
                    f"every keep in {keep_labels} completed "
                    f"{baseline.report.result.objective_score:.3f} with "
                    f"{baseline.report.n_context_overflow} overflows -- the oversized observation "
                    "that blows the prompt stays inside the kept window at this max_steps; "
                    "keep_last_n is a long-transcript policy this task shape cannot exercise"
                ),
            )

    worse: list[str] = []
    for cell in cells:
        if cell.setting.is_shipped:
            continue
        completion = cell.paired.get(METRIC_COMPLETION)
        prompt = cell.paired.get(METRIC_PROMPT_TOKENS)
        if _metric_reading(completion) == READING_SEPARATED and completion is not None:
            if completion["delta"]["mean"] > 0:
                return AxisVerdict(
                    axis,
                    shipped,
                    VERDICT_EXPOSE,
                    (
                        f"{cell.setting.label} separates on completion vs shipped "
                        f"({_delta_cell(completion)}); consider adopting it or keeping the knob "
                        "operator-visible" + _long_transcript_note(axis, cells)
                    ),
                )
            worse.append(f"{cell.setting.label} ({_delta_cell(completion)})")
        if (
            _metric_reading(completion) == READING_FLAT
            and prompt is not None
            and _metric_reading(prompt) == READING_SEPARATED
            and prompt["delta"]["mean"] < 0
        ):
            # Same completion, clearly cheaper prompt -- still an expose (cost win), not a silent
            # default rewrite: the operator chooses whether the token saving is worth the knob.
            return AxisVerdict(
                axis,
                shipped,
                VERDICT_EXPOSE,
                (
                    f"{cell.setting.label} is flat on completion but separates cheaper on prompt "
                    f"tokens ({_delta_cell(prompt)}); expose the knob and consider the cheaper cell"
                    + _long_transcript_note(axis, cells)
                ),
            )

    if worse:
        return AxisVerdict(
            axis,
            shipped,
            VERDICT_PIN,
            (
                f"shipped {axis}={shipped}: no alternative improves completion; "
                f"worse cells {', '.join(worse)}; keep the measured default"
                + _long_transcript_note(axis, cells)
            ),
        )
    return AxisVerdict(
        axis,
        shipped,
        VERDICT_PIN,
        (
            f"shipped {axis}={shipped} is flat on completion against the grid "
            f"({', '.join(c.setting.label for c in cells)}); keep the measured default"
            + _long_transcript_note(axis, cells)
        ),
    )


def _long_transcript_note(axis: str, cells: Sequence[SettingReport]) -> str:
    """Flag when a keep grid never grew past the shipped keep -- the policy was not exercised."""
    if axis != AXIS_KEEP or not cells:
        return ""
    mean_steps = max(c.report.mean_steps for c in cells)
    if mean_steps <= float(DEFAULT_KEEP_LAST_N):
        return (
            f" (warning: max mean_steps={mean_steps:.2f} <= shipped keep={DEFAULT_KEEP_LAST_N}, "
            "so older steps were rarely dropped -- raise max_steps or deepen the task shape)"
        )
    return ""


def format_sweep_table(settings: Sequence[SettingReport], verdicts: Sequence[AxisVerdict]) -> str:
    """Human-readable per-setting table plus the three axis verdicts."""
    lines = [
        f"{'axis':<24} {'setting':<12} {'compl':>6} {'steps':>6} {'prompt':>8} {'ovfl':>4} "
        f"{'d(compl) vs shipped':<28} {'d(prompt) vs shipped':<28}",
        "-" * 130,
    ]
    for cell in settings:
        r = cell.report
        lines.append(
            f"{cell.setting.axis:<24} {cell.setting.label:<12} "
            f"{r.result.objective_score:6.3f} "
            f"{r.mean_steps:6.2f} "
            f"{r.metric_mean(METRIC_PROMPT_TOKENS):8.1f} "
            f"{r.n_context_overflow:4d} "
            f"{_delta_cell(cell.paired.get(METRIC_COMPLETION)):<28} "
            f"{_delta_cell(cell.paired.get(METRIC_PROMPT_TOKENS)):<28}"
        )
    lines.append("")
    lines.append("verdicts:")
    for verdict in verdicts:
        lines.append(
            f"  [{verdict.verdict}] {verdict.axis}={verdict.shipped_value}: {verdict.reason}"
        )
    return "\n".join(lines)


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
    if settings is not None:
        grid = settings
    else:
        grid = grid_for_axes(parse_axes(axes))
    digest = task_set_digest(tasks)
    verification_cfg = verified_data_config(
        data_verified=data_verified, verification_ref=verification_ref
    )
    present_axes = tuple(dict.fromkeys(s.axis for s in grid))

    # Identical (policy, overrides) cells share one episode batch -- the shipped observation_cap
    # cell is both the cap-axis and the head-share-axis baseline.
    cells: list[SettingReport] = []
    cache: dict[tuple[str, tuple[tuple[str, Any], ...]], PolicyReport] = {}
    for setting in grid:
        key = (setting.policy_name, tuple(sorted(setting.overrides.items())))
        report = cache.get(key)
        if report is None:
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
            cache[key] = report
            _LOG.info(
                "[agentic-context-sweep] done %s completion=%.3f prompt-tok=%.0f overflow=%d",
                setting.label,
                report.result.objective_score,
                report.metric_mean(METRIC_PROMPT_TOKENS),
                report.n_context_overflow,
            )
        else:
            _LOG.info("[agentic-context-sweep] reuse %s (identical to a prior cell)", setting.label)
        # Shallow-copy so each cell can carry its own persisted paths without clobbering peers
        # that shared the same episode batch (shipped observation_cap is two axis baselines).
        cells.append(SettingReport(setting=setting, report=copy(report), paired={}))

    pair_against_shipped(cells)
    verdicts = [
        decide_axis_verdict(axis, [c for c in cells if c.setting.axis == axis])
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
            verification_cfg=verification_cfg,
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
