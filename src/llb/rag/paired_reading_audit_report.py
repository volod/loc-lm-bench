"""Markdown presentation for the vector-backed paired-reading audit."""

from llb.rag.paired_reading_audit import PairedReadingAudit


def format_audit(report: PairedReadingAudit) -> str:
    """Name every artifact verdict and every comparison reading that changed."""
    lines = [
        "# Selection-adjusted paired-reading audit",
        "",
        f"- vector-backed artifacts re-read: {report['artifacts']}",
        f"- paired comparisons re-read: {report['comparisons']}",
        f"- comparison readings changed: {len(report['reading_changes'])}",
        f"- legacy aggregate-only grids skipped: {len(report['skipped_artifacts'])}",
        "",
        "## Verdicts",
        "",
        "| lane | artifact | previous | adjusted | selected reading survives? | changed? |",
        "| --- | --- | :-: | :-: | :-: | :-: |",
    ]
    for verdict in report["verdicts"]:
        changed = "YES" if verdict["previous"] != verdict["calibrated"] else "no"
        lines.append(
            f"| {verdict['lane']} | `{verdict['artifact']}` | {verdict['previous']} "
            f"| {verdict['calibrated']} "
            f"| {'YES' if verdict['selection_survives'] else 'no'} | {changed} |"
        )
    lines += [
        "",
        "## Selection-adjusted readings",
        "",
        "| lane | artifact | selected hypothesis | raw p | adjusted p | survives? |",
        "| --- | --- | --- | ---: | ---: | :-: |",
    ]
    for reading in report["selection_readings"]:
        lines.append(
            f"| {reading['lane']} | `{reading['artifact']}` | `{reading['hypothesis']}` "
            f"| {reading['unadjusted_p']:.4f} | {reading['adjusted_p']:.4f} "
            f"| {'YES' if reading['survives'] else 'no'} |"
        )
    if not report["selection_readings"]:
        lines.append("| - | - | - | - | - | - |")
    lines += [
        "",
        "## Legacy artifacts without a joint item ledger",
        "",
        "| lane | artifact | reason |",
        "| --- | --- | --- |",
    ]
    for skipped in report["skipped_artifacts"]:
        lines.append(f"| {skipped['lane']} | `{skipped['artifact']}` | {skipped['reason']} |")
    if not report["skipped_artifacts"]:
        lines.append("| - | - | - |")
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
