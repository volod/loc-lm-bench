"""Report sections shared by the bake-off lanes (embedder, reranker).

Every bake-off publishes the same five things beside its own quality columns: the paired delta cells
per row, the minimum-evidence gate summary, how close each row sits to the adoption cut, the
adopt-or-retain (keep-or-swap) sentence, and whether each scored row reproduced its own model card.
Those sections read `paired_vs_baseline` / `verdict` / `skipped` / `card_parity`, none of which is
specific to what was ranked -- so they live here once and each lane's
renderer supplies only the columns that ARE specific to it.

Row and report are typed as plain mappings on purpose: the two lanes have different `TypedDict`s
with the same shared fields, and a section that only reads the shared fields should not have to
know which lane it is rendering.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from llb.rag.embedding_bakeoff.uncertainty import (
    BARS,
    DEFAULT_CONFIDENCE,
    PairedRow,
    bar_stability,
)
from llb.rag.embedding_bakeoff.verdict import DECISION_ADOPT
from llb.rag.fusion_evidence.evidence_gate import evidence_gate_summary
from llb.rag.fusion_evidence.paired import format_randomization_p, gated_readings
from llb.rag.fusion_evidence.stability import boundary_table, format_reading
from llb.rag.fusion_evidence.stats import format_interval

NO_PAIRED_CELL = "-"


def paired_cells(row: Mapping[str, Any], bar: str) -> tuple[str, str, str, str, str]:
    """Delta, w/l/t ledger, sign p, randomization p, and reading for one bar (dashes when absent).

    The reading column is what keeps a `flat` that missed by a mile from printing exactly like one
    that missed by nothing -- the whole point of the borderline annotation.
    """
    paired: PairedRow | None = row.get("paired_vs_baseline")
    if paired is None:
        return (NO_PAIRED_CELL,) * 5
    delta = paired["metrics"][bar]
    stability = bar_stability(paired, bar)
    return (
        format_interval(delta["delta"]),
        f"{delta['wins']}/{delta['losses']}/{delta['ties']}",
        f"{delta['sign_test_p']:.3f}",
        format_randomization_p(delta),
        format_reading(stability, stability["reading"]) if stability else NO_PAIRED_CELL,
    )


def gate_summary(rows: Sequence[Mapping[str, Any]], confidence: float) -> list[str]:
    """How many of the run's per-bar paired readings the minimum-evidence gate relabeled."""
    comparisons = [
        paired["metrics"][bar]
        for row in rows
        if (paired := row.get("paired_vs_baseline")) is not None
        for bar in BARS
    ]
    gated, total = gated_readings(comparisons, confidence)
    return evidence_gate_summary(gated, total, confidence)


def boundary_section(
    rows: Sequence[Mapping[str, Any]],
    baseline: str | None,
    confidence: float = DEFAULT_CONFIDENCE,
    *,
    title: str,
    key_header: str,
    subject: str,
) -> list[str]:
    """How close each row's bar reading sits to the cut that produced it."""
    entries = []
    for row in sorted(rows, key=lambda r: str(r["model"])):
        paired = row.get("paired_vs_baseline")
        if paired is None or row["model"] == baseline:
            continue
        for bar in BARS:
            stability = bar_stability(paired, bar)
            if stability is not None:
                entries.append((f"`{row['model']}` {bar}", stability))
    return boundary_table(
        entries, title=title, key_header=key_header, subject=subject, confidence=confidence
    )


def skipped_section(
    skipped: Sequence[Mapping[str, Any]], *, title: str = "Roster entries not scored"
) -> list[str]:
    """Roster entries that produced no row.

    A report that ranks fewer models than the roster named must SAY so -- otherwise a declined or
    unrunnable candidate reads as a candidate that lost.
    """
    if not skipped:
        return []
    lines = [f"## {title}", "", "| model | family | reason |", "| --- | --- | --- |"]
    lines += [
        f"| `{row['model']}` | {row['family']} | {_one_line(row['detail'])} |" for row in skipped
    ]
    return [*lines, ""]


def card_parity_section(
    rows: Sequence[Mapping[str, Any]], *, title: str = "Model-card parity"
) -> list[str]:
    """Whether each scored row reproduced its own card before it was ranked.

    A row that ran and was never checked is not the same fact as a row that reproduced its card, so
    the status column prints both -- and a reader who wants to know why a lead is trustworthy can
    see which rows were verified against a published number and which were taken on trust.
    """
    checked = [row for row in rows if row.get("card_parity")]
    if not checked:
        return []
    lines = [
        f"## {title}",
        "",
        "| model | status | mode | worst delta | tolerance | card |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in checked:
        parity = row["card_parity"]
        worst = parity.get("max_abs_diff")
        lines.append(
            f"| `{row['model']}` | {parity['status']} | {parity['mode']} "
            f"| {worst:.4f} | {parity['tolerance']:.4f} | {parity['source'] or '-'} |"
            if worst is not None
            else f"| `{row['model']}` | {parity['status']} | {parity['mode']} | - | - "
            f"| {parity['source'] or '-'} |"
        )
    return [*lines, ""]


def _one_line(detail: str) -> str:
    """Collapse a recorded reason to one line: a host error text carries newlines (a CUDA assert
    prints four), and a raw newline inside a Markdown cell ends the table at that row."""
    return " ".join(str(detail).split())


def verdict_lines(
    report: Mapping[str, Any], prefix: str = "", call_word: str = "ADOPT"
) -> list[str]:
    """The decision sentence, or a note that the run carries no paired reading."""
    verdict = report.get("verdict")
    if verdict is None:
        return [
            f"{prefix}No paired uncertainty was computed for this run, so the ranking above is a "
            "point estimate only."
        ]
    call = call_word if verdict["decision"] == DECISION_ADOPT else verdict["decision"].upper()
    named = f" `{verdict['model']}`" if verdict["model"] else ""
    return [f"{prefix}Verdict: {call}{named} -- {verdict['reason']}."]
