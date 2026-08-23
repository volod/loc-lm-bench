"""ASCII report rendering for ``compare-retrieval``."""

from collections.abc import Mapping, Sequence
from typing import Any

from llb.core.contracts.rag import RetrievalMetrics
from llb.rag.comparison.models import STITCH_ROW_SUFFIX, ComparisonReport
from llb.rag.embedding_bakeoff.uncertainty import (
    METRIC_MRR,
    METRIC_RECALL,
    METRIC_SERVED_CHARS,
    METRIC_SPAN_COVERAGE,
    METRIC_SPAN_INTACT,
)
from llb.rag.fusion_evidence.paired import PairedComparison, reading_of, regresses
from llb.rag.fusion_evidence.stats import format_interval

# The scored columns, in reading order: was the evidence found, how early, how much of it
# arrived, did one chunk carry it whole -- and, last, what serving it cost. Each column carries its
# own format because the cost is a character count, not a rate. Labels are short because the point
# table repeats them per slice.
POINT_COLUMNS = (
    (METRIC_RECALL, "recall@k", "8.3f"),
    (METRIC_MRR, "mrr", "8.3f"),
    (METRIC_SPAN_COVERAGE, "cover@k", "8.3f"),
    (METRIC_SPAN_INTACT, "intact@k", "8.3f"),
    (METRIC_SERVED_CHARS, "chars@k", "8.0f"),
)
# Paired deltas render two metrics per line, so the four columns above become two blocks: the
# FINDING pair the verdict is decided on, then the INTACTNESS pair a chunking change moves.
PAIRED_BLOCKS = (
    ((METRIC_RECALL, "recall"), (METRIC_MRR, "MRR")),
    ((METRIC_SPAN_COVERAGE, "coverage"), (METRIC_SPAN_INTACT, "intact")),
)
# A slice renders ONE paired block -- the intactness pair. A slice is the resolution at which an
# evidence-delivery change is read (the aggregate already carries all four paired blocks), and
# printing four blocks per slice would bury that reading in a report that repeats them per label.
# The JSON artifact keeps every metric's paired reading per slice either way.
SLICE_PAIRED_BLOCK = PAIRED_BLOCKS[1]
# Column width of one paired metric: interval(24) + space + ledger(8) + gap(3) + reading(21).
_CELL_WIDTH = 57


def format_comparison(report: ComparisonReport) -> str:
    """Render point rows, paired evidence, the verdict, and optional diagnostics."""
    backends = report["backends"]
    lines = [f"[compare-retrieval] n={report['n']} k={report['k']}"]
    if not backends:
        lines.append("  (no backends loaded)")
        return "\n".join(lines)
    width = max(len(label) for label in backends)
    lines.append(_point_header(width, "  "))
    for label in sorted(backends):
        lines.append(_point_row(label, backends[label], width, "  "))
    lines.append(f"  best (recall@k): {report['best_recall']}")
    lines.extend(_paired_lines(report))
    verdict = report["verdict"]
    decision = verdict["decision"].upper()
    named = f" `{verdict['lane']}`" if verdict["lane"] else ""
    lines.append(f"  Verdict: {decision}{named} -- {verdict['reason']}.")
    lines.extend(_diagnostic_lines(report, width))
    return "\n".join(lines)


def _point_header(width: int, indent: str) -> str:
    columns = " ".join(label.rjust(8) for _, label, _ in POINT_COLUMNS)
    return f"{indent}{'backend'.ljust(width)}   {columns}"


def _point_row(label: str, metrics: RetrievalMetrics, width: int, indent: str) -> str:
    """One scored lane; a metric an older artifact never recorded prints as `n/a`."""
    values = " ".join(_point_cell(metrics, metric, spec) for metric, _, spec in POINT_COLUMNS)
    return f"{indent}{label.ljust(width)}   {values}"


def _point_cell(metrics: Mapping[str, object], metric: str, spec: str) -> str:
    value = metrics.get(metric)
    return format(value, spec) if isinstance(value, (int, float)) else "     n/a"


def _paired_lines(report: ComparisonReport) -> list[str]:
    """The aligned delta tables, empty only when no baseline lane was scored."""
    settings = report["uncertainty"]
    baseline = settings["baseline"]
    if baseline is None:
        return []
    backends = report["backends"]
    width = max(len(label) for label in backends)
    lines = [
        f"  paired vs {baseline}: {settings['resamples']} resamples, "
        f"{settings['confidence']:.0%} confidence, seed {settings['seed']}"
    ]
    for block in PAIRED_BLOCKS:
        lines.extend(_paired_block(report, block, width))
    return lines


def _paired_block(
    report: ComparisonReport, block: Sequence[tuple[str, str]], width: int
) -> list[str]:
    """One two-metric delta table over the aggregate lanes."""
    return _paired_table(
        report["backends"], block, width, report["uncertainty"]["confidence"], "  "
    )


def _paired_table(
    lanes: Mapping[str, Any],
    block: Sequence[tuple[str, str]],
    width: int,
    confidence: float,
    indent: str,
) -> list[str]:
    """One two-metric delta table, skipped when no scored lane recorded that metric pair."""
    rows: list[str] = []
    for label in sorted(lanes):
        paired = lanes[label].get("paired_vs_baseline")
        if paired is None or any(metric not in paired["metrics"] for metric, _ in block):
            continue
        cells = "".join(_paired_cells(paired["metrics"][metric], confidence) for metric, _ in block)
        rows.append(f"{indent}{label.ljust(width)}   {cells}".rstrip())
    if not rows:
        return []
    header = "".join(
        f"{name} delta [lo, hi]      w/l/t   reading".ljust(_CELL_WIDTH) for _, name in block
    )
    return [f"{indent}{'backend'.ljust(width)}   {header}".rstrip(), *rows]


def _paired_cells(comparison: PairedComparison, confidence: float) -> str:
    """One metric's delta / ledger / reading, padded to `_CELL_WIDTH` so blocks line up."""
    ledger = f"{comparison['wins']}/{comparison['losses']}/{comparison['ties']}"
    cell = (
        f"{format_interval(comparison['delta']):<24} {ledger:>8}   "
        f"{_reading(comparison, confidence)}"
    )
    return cell.ljust(_CELL_WIDTH)


def _reading(comparison: PairedComparison, confidence: float) -> str:
    """A calibrated reading, or an explicit marker when resampling was disabled.

    `reading_of` is one-sided -- it asks whether the CANDIDATE separated -- so a lane the
    baseline beats by an interval clear of zero would print `flat`, which reads as "nothing
    to see". The intactness columns are where that happens most (a chunker can deliver the
    same recall on visibly less of the span), so a separated loss is named `regressed`.
    """
    if "randomization_p" not in comparison:
        return "unmeasured"
    if regresses(comparison, confidence):
        return "regressed"
    return reading_of(comparison, confidence)


def _diagnostic_lines(report: ComparisonReport, width: int) -> list[str]:
    """Optional numeric-noise, duplicate, and question-type diagnostics."""
    lines: list[str] = []
    floor = report.get("noise_floor")
    if floor is not None:
        from llb.rag.noise_floor.report import format_noise_floor

        lines.extend(format_noise_floor(floor))
    kept = report.get("duplicates_kept", {})
    for label, stats in report.get("duplicates", {}).items():
        from llb.rag.duplicates.collapse import format_duplicate_stats

        line = format_duplicate_stats(stats, kept.get(label))
        lines.append(f"  {label.ljust(width)}   {line}")
    lines.extend(_stitch_lines(report, width))
    lines.extend(_slice_lines(report, width))
    return lines


def _stitch_lines(report: ComparisonReport, width: int) -> list[str]:
    """Per stitched twin: what it merged, what that cost, and the invariance it rests on.

    The note is printed once and matters: merging shortens the returned list, so the first hit can
    only move to an EARLIER position and `mrr` compresses with the block count. The lever is read
    on `intact@k` against `chars@k`, never on the finding columns it cannot move.
    """
    stitching = report.get("stitching", {})
    if not stitching:
        return []
    lines = [
        f"  {STITCH_ROW_SUFFIX} rows merge contiguous retrieved chunks after the top-k cut: "
        "recall@k/cover@k are invariant, mrr compresses with the block count -- "
        "read intact@k against chars@k"
    ]
    for label in sorted(stitching):
        entry = stitching[label]
        census = entry["census"]
        invariant = entry["recall_invariant"] and entry["coverage_invariant"]
        held = "invariance held" if invariant else "INVARIANCE FAILED"
        lines.append(
            f"  {label.ljust(width)}   {census['blocks_per_query']:.2f} blocks/query from "
            f"{census['parts_per_query']:.2f} chunks ({census['merged_per_query']:.2f} merged), "
            f"{census['chars_delta_per_query']:+.1f} chars/query vs `{entry['base']}`, {held}"
        )
    return lines


def _slice_lines(report: ComparisonReport, width: int) -> list[str]:
    """The per-question-type breakdown: one scored block per labeled slice, empties named once.

    Each non-empty slice prints its point rows and then its OWN intactness deltas, drawn on that
    slice's items: a 14-item slice turns on one question, so a point move there is not a reading
    until an interval is beside it. A slice with no labeled item scores nothing, so printing its
    zeros would read as a measured result. The empty focus slices are named on one line instead,
    and the JSON report keeps every slice (with its `n`) either way.
    """
    slices = report.get("slices", {})
    confidence = report["uncertainty"]["confidence"]
    lines: list[str] = []
    for slice_label, slice_report in slices.items():
        if not slice_report["n"]:
            continue
        lines.append(f"  slice {slice_label} (n={slice_report['n']}):")
        for label in sorted(slice_report["backends"]):
            lines.append(_point_row(label, slice_report["backends"][label], width, "    "))
        lines.extend(
            _paired_table(slice_report["backends"], SLICE_PAIRED_BLOCK, width, confidence, "    ")
        )
    empty = [label for label, entry in slices.items() if not entry["n"]]
    if empty:
        lines.append(f"  slices with no labeled item: {', '.join(sorted(empty))}")
    return lines
