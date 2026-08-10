"""Pair each swept setting against the shipped one, and cut the pin / expose / inapplicable verdict.

The reading is separated from the RUN for the reason every other lane separates them: pairing and
cutting a verdict are pure over already-scored reports, so they are unit-testable without a model,
and a change to how a verdict reads can never change what was measured.
"""

from collections.abc import Sequence

from llb.bench.agentic.context_policy import DEFAULT_KEEP_LAST_N
from llb.bench.agentic_context_report import METRIC_COMPLETION, METRIC_PROMPT_TOKENS, METRICS
from llb.bench.agentic_context_sweep_model import (
    AXES,
    AXIS_KEEP,
    VERDICT_EXPOSE,
    VERDICT_INAPPLICABLE,
    VERDICT_PIN,
    AxisVerdict,
    SettingReport,
    shipped_value,
)
from llb.rag.fusion_evidence.evidence_gate import (
    DEFAULT_CONFIDENCE,
    READING_FLAT,
    READING_SEPARATED,
    apply_evidence_gate,
)
from llb.rag.fusion_evidence.paired import PairedComparison, paired_comparison
from llb.rag.fusion_evidence.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_index_sets


def _pair_axis(
    cells: list[SettingReport],
    baseline: SettingReport,
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> None:
    """Pair one axis's cells against its shipped cell over SHARED bootstrap index sets, in place.

    Shared index sets rather than per-cell draws: two cells compared against the same baseline must
    be resampled the same way, or their intervals are not comparable to each other. A cell scored
    over a different item count is left unpaired instead of being lined up by position.
    """
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


def pair_against_shipped(
    settings: list[SettingReport],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> None:
    """Attach each non-shipped cell's paired deltas against its axis's shipped cell, in place."""
    for axis in AXES:
        cells = [row for row in settings if row.setting.axis == axis]
        baseline = next((row for row in cells if row.setting.is_shipped), None)
        if baseline is not None:
            _pair_axis(cells, baseline, confidence=confidence, resamples=resamples, seed=seed)


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


def _keep_inapplicable_verdict(
    axis: str, cells: Sequence[SettingReport], baseline: SettingReport
) -> AxisVerdict | None:
    """`keep_last_n` is a LONG-transcript policy; say so when this task shape cannot exercise it.

    Every cell overflowing identically at a flat completion means the single fat observation that
    blew the prompt sits inside every kept window -- the axis was not measured, so it is neither
    pinned nor exposed.
    """
    if axis != AXIS_KEEP:
        return None
    overflows = {cell.report.n_context_overflow for cell in cells}
    completions = {round(cell.report.result.objective_score, 6) for cell in cells}
    if len(overflows) != 1 or len(completions) != 1 or baseline.report.n_context_overflow <= 0:
        return None
    keep_labels = tuple(cell.setting.label for cell in cells)
    return AxisVerdict(
        axis,
        shipped_value(axis),
        VERDICT_INAPPLICABLE,
        (
            f"every keep in {keep_labels} completed "
            f"{baseline.report.result.objective_score:.3f} with "
            f"{baseline.report.n_context_overflow} overflows -- the oversized observation "
            "that blows the prompt stays inside the kept window at this max_steps; "
            "keep_last_n is a long-transcript policy this task shape cannot exercise"
        ),
    )


def _completion_expose(
    axis: str, cell: SettingReport, cells: Sequence[SettingReport]
) -> AxisVerdict:
    """One cell beat shipped on completion: expose the knob rather than rewrite the default."""
    completion = cell.paired[METRIC_COMPLETION]
    return AxisVerdict(
        axis,
        shipped_value(axis),
        VERDICT_EXPOSE,
        (
            f"{cell.setting.label} separates on completion vs shipped "
            f"({_delta_cell(completion)}); consider adopting it or keeping the knob "
            "operator-visible" + _long_transcript_note(axis, cells)
        ),
    )


def _cheaper_expose(axis: str, cell: SettingReport, cells: Sequence[SettingReport]) -> AxisVerdict:
    """Same completion, clearly cheaper prompt -- still an expose (cost win), not a silent rewrite."""
    prompt = cell.paired[METRIC_PROMPT_TOKENS]
    return AxisVerdict(
        axis,
        shipped_value(axis),
        VERDICT_EXPOSE,
        (
            f"{cell.setting.label} is flat on completion but separates cheaper on prompt "
            f"tokens ({_delta_cell(prompt)}); expose the knob and consider the cheaper cell"
            + _long_transcript_note(axis, cells)
        ),
    )


def _cell_expose_verdict(
    axis: str, cell: SettingReport, cells: Sequence[SettingReport]
) -> AxisVerdict | None:
    """Whether ONE alternative cell is reason enough to expose the axis."""
    completion = cell.paired.get(METRIC_COMPLETION)
    prompt = cell.paired.get(METRIC_PROMPT_TOKENS)
    completion_reading = _metric_reading(completion)
    if (
        completion_reading == READING_SEPARATED
        and completion is not None
        and completion["delta"]["mean"] > 0
    ):
        return _completion_expose(axis, cell, cells)
    if (
        completion_reading == READING_FLAT
        and prompt is not None
        and _metric_reading(prompt) == READING_SEPARATED
        and prompt["delta"]["mean"] < 0
    ):
        return _cheaper_expose(axis, cell, cells)
    return None


def _is_worse(cell: SettingReport) -> bool:
    """A cell that separates the wrong way is evidence FOR the shipped default."""
    completion = cell.paired.get(METRIC_COMPLETION)
    return (
        _metric_reading(completion) == READING_SEPARATED
        and completion is not None
        and completion["delta"]["mean"] <= 0
    )


def _pin_verdict(axis: str, cells: Sequence[SettingReport], worse: list[str]) -> AxisVerdict:
    """Nothing beat shipped: pin it, saying whether the grid was worse or merely flat."""
    shipped = shipped_value(axis)
    note = _long_transcript_note(axis, cells)
    if worse:
        reason = (
            f"shipped {axis}={shipped}: no alternative improves completion; "
            f"worse cells {', '.join(worse)}; keep the measured default" + note
        )
    else:
        reason = (
            f"shipped {axis}={shipped} is flat on completion against the grid "
            f"({', '.join(cell.setting.label for cell in cells)}); keep the measured default" + note
        )
    return AxisVerdict(axis, shipped, VERDICT_PIN, reason)


def decide_axis_verdict(axis: str, cells: Sequence[SettingReport]) -> AxisVerdict:
    """Cut pin / expose / inapplicable for one axis from its paired cells."""
    baseline = next((cell for cell in cells if cell.setting.is_shipped), None)
    if baseline is None:
        return AxisVerdict(
            axis, shipped_value(axis), VERDICT_EXPOSE, "shipped cell missing from the grid"
        )
    inapplicable = _keep_inapplicable_verdict(axis, cells, baseline)
    if inapplicable is not None:
        return inapplicable
    alternatives = [cell for cell in cells if not cell.setting.is_shipped]
    for cell in alternatives:
        expose = _cell_expose_verdict(axis, cell, cells)
        if expose is not None:
            return expose
    worse = [
        f"{cell.setting.label} ({_delta_cell(cell.paired[METRIC_COMPLETION])})"
        for cell in alternatives
        if _is_worse(cell)
    ]
    return _pin_verdict(axis, cells, worse)


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
