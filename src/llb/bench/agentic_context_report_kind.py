"""Per-task-kind tables and aggregate-safe verdicts for context-policy runs."""

from llb.bench.agentic.context_aggregate import task_kind
from llb.bench.agentic.model import STATUS_CONTEXT_OVERFLOW
from llb.bench.agentic_context_report import BASELINE_POLICY, PolicyReport
from llb.bench.common import mean
from llb.rag.fusion_evidence.evidence_gate import READING_FLAT, READING_SEPARATED, reading_label
from llb.rag.fusion_evidence.paired import PairedComparison, paired_comparison
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED
from llb.rag.fusion_evidence.stats import bootstrap_index_sets

KIND_COUNT = "count"
KIND_LOCATE = "locate"
KIND_OTHER = "other"
KIND_ORDER = (KIND_COUNT, KIND_LOCATE, KIND_OTHER)

PRE_HEADER_COUNT_COMPLETION: dict[str, float] = {
    "full": 0.0,
    "observation_cap": 0.0,
    "keep_last_n": 0.0,
    "compact": 0.0,
}


def kind_indices(report: PolicyReport, kind: str) -> list[int]:
    return [
        index
        for index, row in enumerate(report.rows)
        if task_kind(str(row.get("item_id", ""))) == kind
    ]


def kind_completion(report: PolicyReport, kind: str) -> float | None:
    indexes = kind_indices(report, kind)
    return mean([report.case_success[index] for index in indexes]) if indexes else None


def kind_overflow(report: PolicyReport, kind: str) -> int:
    return sum(
        1
        for index in kind_indices(report, kind)
        if report.rows[index].get("status") == STATUS_CONTEXT_OVERFLOW
    )


def pair_kind_completion(
    reports: list[PolicyReport],
    kind: str,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, PairedComparison]:
    baseline = next((row for row in reports if row.policy == BASELINE_POLICY), None)
    if baseline is None:
        return {}
    indexes = kind_indices(baseline, kind)
    if len(indexes) < 2:
        return {}
    base_vector = [baseline.case_success[index] for index in indexes]
    index_sets = bootstrap_index_sets(len(indexes), resamples, seed)
    return {
        report.policy: paired_comparison(
            [report.case_success[index] for index in indexes],
            base_vector,
            index_sets,
            confidence,
        )
        for report in reports
        if report.policy != BASELINE_POLICY
        and len(report.case_success) == len(baseline.case_success)
    }


def _reference_report(reports: list[PolicyReport]) -> PolicyReport | None:
    return next((row for row in reports if row.policy == BASELINE_POLICY), None) or (
        reports[0] if reports else None
    )


def _kind_header(present: list[str]) -> str:
    header = f"{'policy':<16}" + "".join(f" {kind:>10}" for kind in present)
    header += f" {'overflow-count':>14}"
    return header + f" {'vs-pre-header':>14}" if KIND_COUNT in present else header


def _kind_cells(report: PolicyReport, present: list[str]) -> str:
    cells = []
    for kind in present:
        value = kind_completion(report, kind)
        cells.append(f"{value:>10.3f}" if value is not None else f"{'-':>10}")
    return "".join(f" {cell}" for cell in cells)


def _pre_header_cell(report: PolicyReport, pre_header: dict[str, float]) -> str:
    prior = pre_header.get(report.policy)
    current = kind_completion(report, KIND_COUNT)
    if prior is None or current is None:
        return f" {'-':>14}"
    return f" {current - prior:>+14.3f}"


def _kind_row(
    report: PolicyReport,
    present: list[str],
    pre_header: dict[str, float],
) -> str:
    row = f"{report.policy:<16}" + _kind_cells(report, present)
    if KIND_COUNT not in present:
        return row
    return row + f" {kind_overflow(report, KIND_COUNT):>14d}" + _pre_header_cell(report, pre_header)


def _count_pair_lines(pairs: dict[str, PairedComparison]) -> list[str]:
    lines = ["count-slice paired vs full:"]
    for policy, comparison in pairs.items():
        delta = comparison["delta"]
        reading = READING_SEPARATED if delta["lo"] > 0.0 or delta["hi"] < 0.0 else READING_FLAT
        lines.append(
            f"  {policy:<14} d(completion)={delta['mean']:+.3f} "
            f"[{delta['lo']:+.3f}, {delta['hi']:+.3f}] "
            f"w/l/t={comparison['wins']}/{comparison['losses']}/{comparison['ties']} "
            f"{reading_label(reading)}"
        )
    return lines


def format_kind_table(
    reports: list[PolicyReport],
    *,
    pre_header_count: dict[str, float] | None = None,
) -> str:
    reference = _reference_report(reports)
    if reference is None or not reference.rows:
        return ""
    present = [kind for kind in KIND_ORDER if kind_indices(reference, kind)]
    if not present:
        return ""
    pre_header = pre_header_count or PRE_HEADER_COUNT_COMPLETION
    header = _kind_header(present)
    lines = ["by task kind:", header, "-" * len(header)]
    lines.extend(_kind_row(report, present, pre_header) for report in reports)
    pairs = pair_kind_completion(reports, KIND_COUNT) if KIND_COUNT in present else {}
    if pairs:
        lines.extend(_count_pair_lines(pairs))
    return "\n".join(lines)


def _recovery_bit(name: str, prior: float, current: float) -> str:
    delta = current - prior
    if delta > 0:
        return f"`{name}` count {prior:.3f}->{current:.3f} (recovered {delta:+.3f})"
    return f"`{name}` count {prior:.3f}->{current:.3f} (no recovery; loss is elsewhere or flat)"


def aggregate_safe_verdict(
    reports: list[PolicyReport],
    *,
    pre_header_count: dict[str, float] | None = None,
) -> str:
    pre_header = pre_header_count or PRE_HEADER_COUNT_COMPLETION
    reference = _reference_report(reports)
    if reference is None or not kind_indices(reference, KIND_COUNT):
        return "no count-slice tasks in this set; aggregate-safe trimming not scored"
    measured = []
    for name in ("observation_cap", "compact"):
        report = next((row for row in reports if row.policy == name), None)
        current = kind_completion(report, KIND_COUNT) if report is not None else None
        if current is not None:
            measured.append((name, pre_header.get(name, 0.0), current))
    if not measured:
        return "count slice present but observation_cap/compact were not in this run"
    bits = "; ".join(_recovery_bit(*row) for row in measured)
    if any(current > prior for _name, prior, current in measured):
        return "aggregate-safe trimming recovered count-slice completion: " + bits
    return "aggregate-safe trimming did NOT move the count slice vs pre-header evidence: " + bits
