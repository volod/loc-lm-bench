"""Render a `MissAnalysis` to its Markdown report and machine-readable `analysis.json`, and
persist the analysis bundle (report.md + misses.jsonl + analysis.json) under a timestamped dir.

`latest_analysis` reads the newest persisted payload back for `llb recommend`.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from llb.board.miss_analysis.model import (
    ANALYSIS_FILENAME,
    CLUSTER_DIMENSIONS,
    MISS_ANALYSIS_METHOD,
    MISSES_FILENAME,
    REPORT_FILENAME,
    _TIMESTAMP_FORMAT,
    ClusterRow,
    MissAnalysis,
    _t,
)
from llb.core.contracts import JsonObject


def _fmt_rate(value: float) -> str:
    return f"{value:.0%}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _class_table(analysis: MissAnalysis) -> str:
    n_misses = len(analysis.misses) or 1
    rows = [
        [cls, str(count), _fmt_rate(count / n_misses)]
        for cls, count in analysis.class_counts.items()
        if count
    ]
    return _md_table(["miss class", "n", "share of misses"], rows)


def _cluster_table(rows: list[ClusterRow]) -> str:
    return _md_table(
        ["key", "misses", "cases", "miss rate"],
        [[c.key, str(c.n_misses), str(c.n_cases), _fmt_rate(c.miss_rate)] for c in rows],
    )


def _probe_table(probes: list[JsonObject]) -> str:
    rows = [
        [
            str(p["top_k"]),
            f"{float(p.get('mean_objective', 0.0)):.3f}",
            f"{float(p.get('base_mean_objective', 0.0)):.3f}",
            f"{p.get('recovered_retrieval', 0)}/{p.get('n_retrieval_misses', 0)}",
            str(p.get("run_dir", "?")),
        ]
        for p in sorted(probes, key=lambda p: int(p["top_k"]))
    ]
    return _md_table(
        [
            "probe top_k",
            "mean objective",
            "baseline objective",
            "retrieval misses recovered",
            "run",
        ],
        rows,
    )


def format_report_md(analysis: MissAnalysis) -> str:
    """Render the analysis as the Markdown report written beside `misses.jsonl`."""
    lines = [
        "# loc-lm-bench miss analysis",
        "",
        _t(
            "header_line",
            run_dir=analysis.run_dir,
            model=analysis.model,
            backend=analysis.backend,
            split=analysis.split,
            n_cases=analysis.n_cases,
            threshold=analysis.threshold,
        ),
    ]
    if not analysis.misses:
        lines += ["", _t("no_misses", threshold=analysis.threshold)]
        return "\n".join(lines)
    lines += [
        _t(
            "summary_line",
            n_misses=len(analysis.misses),
            n_cases=analysis.n_cases,
            pct=_fmt_rate(len(analysis.misses) / analysis.n_cases if analysis.n_cases else 0.0),
        ),
        "",
        "## Miss classes",
        "",
        _class_table(analysis),
    ]
    for dimension in CLUSTER_DIMENSIONS:
        rows = analysis.clusters.get(dimension, [])
        if rows:
            lines += ["", f"## Misses by {dimension.replace('_', ' ')}", "", _cluster_table(rows)]
    if analysis.probes:
        lines += [
            "",
            "## Retrieval-depth probes (miss subset only)",
            "",
            _probe_table(analysis.probes),
        ]
    if analysis.recommendations:
        lines += ["", "## Recommendations", ""]
        lines += [f"{rank}. {rec['line']}" for rank, rec in enumerate(analysis.recommendations, 1)]
    return "\n".join(lines)


def analysis_payload(analysis: MissAnalysis) -> JsonObject:
    """Machine-readable summary (`analysis.json`) consumed by `llb recommend`."""
    return {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": analysis.run_dir,
        "model": analysis.model,
        "backend": analysis.backend,
        "split": analysis.split,
        "n_cases": analysis.n_cases,
        "n_misses": len(analysis.misses),
        "threshold": analysis.threshold,
        "rag_config": analysis.rag_config,
        "class_counts": analysis.class_counts,
        "clusters": {
            dimension: [row.as_dict() for row in rows]
            for dimension, rows in analysis.clusters.items()
        },
        "probes": analysis.probes,
        "recommendations": analysis.recommendations,
    }


def write_analysis(analysis: MissAnalysis, out_dir: Path | str) -> dict[str, str]:
    """Persist report.md + misses.jsonl + analysis.json under one analysis directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / REPORT_FILENAME
    report_path.write_text(format_report_md(analysis) + "\n", encoding="utf-8")
    misses_path = out_dir / MISSES_FILENAME
    misses_path.write_text(
        "".join(json.dumps(m.as_dict(), ensure_ascii=False) + "\n" for m in analysis.misses),
        encoding="utf-8",
    )
    analysis_path = out_dir / ANALYSIS_FILENAME
    analysis_path.write_text(
        json.dumps(analysis_payload(analysis), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "misses": str(misses_path),
        "analysis": str(analysis_path),
    }


def analysis_out_dir(data_dir: Path | str) -> Path:
    """A fresh `$DATA_DIR/miss-analysis/<timestamp>/` directory path (not yet created)."""
    stamp = datetime.now(timezone.utc).strftime(_TIMESTAMP_FORMAT)
    return Path(data_dir) / MISS_ANALYSIS_METHOD / stamp


def latest_analysis(data_dir: Path | str) -> JsonObject | None:
    """The newest persisted `analysis.json` under `$DATA_DIR/miss-analysis/`, with its report
    path attached -- or None when no analysis has ever run (recommend then omits the section)."""
    root = Path(data_dir) / MISS_ANALYSIS_METHOD
    if not root.is_dir():
        return None
    for candidate in sorted(root.iterdir(), reverse=True):
        payload_path = candidate / ANALYSIS_FILENAME
        if payload_path.is_file():
            try:
                payload: JsonObject = json.loads(payload_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            payload["report_path"] = str(candidate / REPORT_FILENAME)
            return payload
    return None
