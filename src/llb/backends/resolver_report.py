"""Focused resolver report implementation."""

from llb.backends.runtime_floor import RUNTIME_FLOOR_SKIP
from llb.core.contracts.models import ResolvedModel


def runtime_floor_skips(rows: list[ResolvedModel]) -> list[str]:
    """One line per candidate the installed runtime is too old to serve.

    The chosen-backend table cannot show these: a model that falls back to another backend still
    resolves, so the hole -- an entry whose Ollama path this host cannot serve at all -- would be
    invisible exactly when it changes which artifact a measurement was taken on.
    """
    return [
        f"skip {row['name']} / {candidate['backend']}: {candidate['reason']}"
        for row in rows
        for candidate in row["candidates"]
        if candidate.get("skip") == RUNTIME_FLOOR_SKIP
    ]


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
