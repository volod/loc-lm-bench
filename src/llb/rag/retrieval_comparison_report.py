"""ASCII report rendering for ``compare-retrieval``."""

from llb.rag.compare import ComparisonReport
from llb.rag.fusion_evidence.paired import PairedComparison, reading_of
from llb.rag.fusion_evidence.stats import format_interval


def format_comparison(report: ComparisonReport) -> str:
    """Render point rows, paired evidence, the verdict, and optional diagnostics."""
    backends = report["backends"]
    lines = [f"[compare-retrieval] n={report['n']} k={report['k']}"]
    if not backends:
        lines.append("  (no backends loaded)")
        return "\n".join(lines)
    width = max(len(label) for label in backends)
    lines.append(f"  {'backend'.ljust(width)}   recall@k      mrr")
    for label in sorted(backends):
        metrics = backends[label]
        lines.append(
            f"  {label.ljust(width)}   {metrics['recall_at_k']:8.3f} {metrics['mrr']:8.3f}"
        )
    lines.append(f"  best (recall@k): {report['best_recall']}")
    lines.extend(_paired_lines(report))
    verdict = report["verdict"]
    decision = verdict["decision"].upper()
    named = f" `{verdict['lane']}`" if verdict["lane"] else ""
    lines.append(f"  Verdict: {decision}{named} -- {verdict['reason']}.")
    lines.extend(_diagnostic_lines(report, width))
    return "\n".join(lines)


def _paired_lines(report: ComparisonReport) -> list[str]:
    """The aligned delta table, empty only when no baseline lane was scored."""
    settings = report["uncertainty"]
    baseline = settings["baseline"]
    if baseline is None:
        return []
    backends = report["backends"]
    width = max(len(label) for label in backends)
    lines = [
        f"  paired vs {baseline}: {settings['resamples']} resamples, "
        f"{settings['confidence']:.0%} confidence, seed {settings['seed']}",
        f"  {'backend'.ljust(width)}   recall delta [lo, hi]      w/l/t   reading"
        "              MRR delta [lo, hi]      w/l/t   reading",
    ]
    for label in sorted(backends):
        paired = backends[label].get("paired_vs_baseline")
        if paired is None:
            continue
        recall = paired["metrics"]["recall_at_k"]
        mrr = paired["metrics"]["mrr"]
        recall_ledger = f"{recall['wins']}/{recall['losses']}/{recall['ties']}"
        mrr_ledger = f"{mrr['wins']}/{mrr['losses']}/{mrr['ties']}"
        lines.append(
            f"  {label.ljust(width)}   {format_interval(recall['delta']):<24} "
            f"{recall_ledger:>8}   {_reading(recall, settings['confidence']):<20} "
            f"{format_interval(mrr['delta']):<24} {mrr_ledger:>8}   "
            f"{_reading(mrr, settings['confidence'])}"
        )
    return lines


def _reading(comparison: PairedComparison, confidence: float) -> str:
    """A calibrated reading, or an explicit marker when resampling was disabled."""
    return reading_of(comparison, confidence) if "randomization_p" in comparison else "unmeasured"


def _diagnostic_lines(report: ComparisonReport, width: int) -> list[str]:
    """Optional numeric-noise, duplicate, and question-type diagnostics."""
    lines: list[str] = []
    floor = report.get("noise_floor")
    if floor is not None:
        from llb.rag.noise_floor_report import format_noise_floor

        lines.extend(format_noise_floor(floor))
    for label, stats in report.get("duplicates", {}).items():
        from llb.rag.duplicates import format_duplicate_stats

        lines.append(f"  {label.ljust(width)}   {format_duplicate_stats(stats)}")
    for slice_label, slice_report in report.get("slices", {}).items():
        lines.append(f"  slice {slice_label} (n={slice_report['n']}):")
        for label in sorted(slice_report["backends"]):
            metrics = slice_report["backends"][label]
            lines.append(
                f"    {label.ljust(width)}   {metrics['recall_at_k']:8.3f} {metrics['mrr']:8.3f}"
            )
    return lines
