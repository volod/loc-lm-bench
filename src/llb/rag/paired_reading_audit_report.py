"""Markdown presentation for the vector-backed paired-reading audit."""

from llb.rag.paired_reading_audit import PairedReadingAudit


def format_audit(report: PairedReadingAudit) -> str:
    """Name every artifact verdict and every comparison reading that changed."""
    lines = [
        "# Randomization-calibrated paired-reading audit",
        "",
        f"- vector-backed artifacts re-read: {report['artifacts']}",
        f"- paired comparisons re-read: {report['comparisons']}",
        f"- comparison readings changed: {len(report['reading_changes'])}",
        "",
        "## Verdicts",
        "",
        "| lane | artifact | previous | calibrated | changed? |",
        "| --- | --- | :-: | :-: | :-: |",
    ]
    for verdict in report["verdicts"]:
        changed = "YES" if verdict["previous"] != verdict["calibrated"] else "no"
        lines.append(
            f"| {verdict['lane']} | `{verdict['artifact']}` | {verdict['previous']} "
            f"| {verdict['calibrated']} | {changed} |"
        )
    lines += [
        "",
        "## Changed comparison readings",
        "",
        "| artifact | comparison | previous | calibrated | randomization p |",
        "| --- | --- | :-: | :-: | ---: |",
    ]
    for change in report["reading_changes"]:
        lines.append(
            f"| `{change['artifact']}` | `{change['comparison']}` | {change['previous']} "
            f"| {change['calibrated']} | {change['randomization_p']:.4f} |"
        )
    if not report["reading_changes"]:
        lines.append("| - | - | - | - | - |")
    lines.append("")
    return "\n".join(lines)
