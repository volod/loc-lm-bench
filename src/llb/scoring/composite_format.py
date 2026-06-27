"""ASCII formatters for guarded category composite outputs."""

from collections.abc import Sequence

from llb.contracts import JsonObject
from llb.scoring.composite_types import CompositeIssue


def format_composite_issues(issues: Sequence[CompositeIssue], *, limit: int = 12) -> str:
    """ASCII summary of why the composite headline is blocked."""
    if not issues:
        return ""
    lines = ["Category composite headline is blocked:"]
    for issue in issues[:limit]:
        tier = f" {issue.tier}" if issue.tier else ""
        lines.append(f"- {issue.model}{tier}: {issue.reason}")
    if len(issues) > limit:
        lines.append(f"- ... {len(issues) - limit} more")
    return "\n".join(lines)


def format_composite_rows(rows: Sequence[JsonObject]) -> str:
    """Render composite rows as an ASCII table."""
    headers = ["rank", "model", "score", "ci", "avg_reliability", "n_cases", "unresolved"]
    table = [[_row_cell(row, header) for header in headers] for row in rows]
    widths = [
        max(len(header), *(len(row[i]) for row in table)) if table else len(header)
        for i, header in enumerate(headers)
    ]
    out = [
        "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)).rstrip(),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in table:
        out.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))).rstrip())
    return "\n".join(out)


def _row_cell(row: JsonObject, key: str) -> str:
    ci = row.get("score_ci")
    mapping = {
        "rank": str(row.get("rank", "-")),
        "model": str(row["model"]),
        "score": f"{float(row['score']):.3f}",
        "ci": "-" if ci is None else f"[{ci[0]:.2f},{ci[1]:.2f}]",
        "avg_reliability": f"{float(row['avg_reliability']):.3f}",
        "n_cases": str(row["n_cases"]),
        "unresolved": "yes" if row["unresolved"] else "no",
    }
    return mapping[key]
