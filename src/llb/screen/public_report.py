"""Focused public report implementation."""

import re
from llb.core.contracts.screening import ScreenReport


def screen_score(report: ScreenReport) -> float:
    """One model's headline screen score on its track: the mean of its per-task scores."""
    results = report["results"]
    return sum(r["score"] for r in results) / len(results) if results else 0.0


def select_finalists(reports: list[ScreenReport], top_n: int) -> list[str]:
    """Deterministic per-track finalist policy (public screen): the top-N models by mean screen score,
    computed SEPARATELY per track (logprob vs generation are never cross-ranked) and tie-broken
    by model name so the handoff to Tier-2 is reproducible. Returns the union of per-track picks.
    """
    finalists: list[str] = []
    for track in sorted({r["track"] for r in reports}):
        ranked = sorted(
            (r for r in reports if r["track"] == track),
            key=lambda r: (-screen_score(r), r["model"]),
        )
        finalists += [r["model"] for r in ranked[:top_n]]
    return finalists


def assert_single_track(reports: list[ScreenReport]) -> str:
    """Refuse to rank logprob and generation screens together (they are not comparable)."""
    tracks = {r["track"] for r in reports}
    if len(tracks) > 1:
        raise ValueError(
            f"cannot rank across screen tracks: {sorted(tracks)} "
            "(loglikelihood accuracy is not comparable to generation exact-match)"
        )
    return tracks.pop() if tracks else ""


def format_screen(reports: list[ScreenReport]) -> str:
    """ASCII per-model screen table (one track; coverage shown)."""
    assert_single_track(reports)
    all_tasks = sorted({r["task"] for rep in reports for r in rep["results"]})
    headers = ["model", "backend", *all_tasks, "coverage"]

    def cell(rep: ScreenReport) -> list[str]:
        by_task = {r["task"]: r["score"] for r in rep["results"]}
        cov = f"{len(rep['covered'])}/{len(rep['requested_tasks'])}"
        return [
            rep["model"],
            rep["backend"],
            *[f"{by_task[t]:.3f}" if t in by_task else "-" for t in all_tasks],
            cov + ("" if rep["complete"] else " PARTIAL"),
        ]

    table = [cell(r) for r in reports]
    widths = [
        max(len(h), *(len(r[i]) for r in table)) if table else len(h) for i, h in enumerate(headers)
    ]
    out = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip(),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for r in table:
        out.append("  ".join(r[i].ljust(widths[i]) for i in range(len(headers))).rstrip())
    return "\n".join(out)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "model"
