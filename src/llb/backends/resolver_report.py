"""Focused resolver report implementation."""

from llb.core.contracts import (
    ResolvedModel,
)


def format_resolution(rows: list[ResolvedModel]) -> str:
    """ASCII table: the chosen backend per model + the verdict."""
    headers = ["model", "chosen", "source", "verdict", "note"]

    def fmt(r: ResolvedModel) -> list[str]:
        return [
            r["name"],
            r["chosen_backend"] or "-",
            r["chosen_source"] or "-",
            r["verdict"],
            r["note"] or "ok",
        ]

    table = [fmt(r) for r in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in table)) if table else len(h) for i, h in enumerate(headers)
    ]
    out = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for r in table:
        out.append("  ".join(r[i].ljust(widths[i]) for i in range(len(headers))).rstrip())
    return "\n".join(out)
