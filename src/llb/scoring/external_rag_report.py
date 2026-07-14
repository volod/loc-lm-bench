"""Markdown report rendering for external-RAG scoring: estimate tables + tuning recommendations.

Turns a scored-row list plus the summary dict into the operator-facing Markdown report. Pure over
its inputs apart from the final atomic write.
"""

from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.scoring.external_rag_sections import (
    _field_set,
    _human_table,
    _priority_table,
    _recommendation_lines,
    _source_audit_section,
    _source_table,
    _split_table,
    _summary_table,
)


def write_report(
    rows: list[dict[str, object]],
    summary: dict[str, object],
    path: Path,
    *,
    answers_path: Path,
    csv_path: Path,
    source_limit: int,
) -> None:
    """Write the Markdown score estimate and recommendation report."""
    lines = [
        "# External RAG score report",
        "",
        "This is an answer-log diagnostic for an external RAG system. It does not launch the "
        "project RAG backend, and it does not certify unverified gold rows as a benchmark "
        "leaderboard.",
        "",
        "## Inputs and outputs",
        "",
        f"- Answer log: `{answers_path}`",
        f"- Detailed CSV: `{csv_path}`",
        f"- System label: `{summary['label']}`",
        "- Scoring: normalized exact match, token F1 objective, and contains.",
        "- Source footer handling: a trailing `Source:` / `Dzherelo:` answer footer is stripped "
        "for scoring; the raw answer remains in the CSV.",
        f"- Source columns: first {source_limit} returned source record(s).",
        "- Answer field(s): "
        + _field_set(summary.get("answer_fields"))
        + "; source field(s): "
        + _field_set(summary.get("sources_fields"))
        + "; error field(s): "
        + _field_set(summary.get("error_fields"))
        + ".",
        "",
        "## Score estimates",
        "",
        _summary_table(summary),
        "",
        "## Human decisions",
        "",
        _human_table(summary),
        "",
        "## Split estimates",
        "",
        _split_table(summary["split_metrics"]),
        "",
        "## Human review workflow",
        "",
        "The interactive command stores human review state in the JSONL answer log. The CSV and "
        "this report are generated only after all rows have `human_score_0_1` and "
        "`human_decision`.",
        "",
        "## Highest-priority rows",
        "",
        _priority_table(rows),
        "",
        *_source_audit_section(summary),
        "## Common returned sources",
        "",
        _source_table(summary["source_title_counts"]),
        "",
        "## Improvement recommendations",
        "",
        *_recommendation_lines(rows, summary),
        "",
        "## Project tuning map",
        "",
        "- Validate the gold data shape before headline use: "
        "`make validate-goldset GOLDSET=<goldset.jsonl> CORPUS=<corpus-dir>`; guide: "
        "`docs/guides/data-prep/goldset-from-scratch.md`.",
        "- Build a local retrieval baseline over the same corpus: "
        "`make build-index CORPUS=<corpus-dir>` then "
        "`make validate-retrieval GOLDSET=<goldset.jsonl> RAG_K=10`.",
        "- Compare Ukrainian embedders on this corpus: "
        "`make compare-embeddings GOLDSET=<goldset.jsonl> RAG_K=20`.",
        "- Test chunking and retrieval-mode knobs locally: "
        "`llb build-index --corpus-root <corpus-dir> --strategy markdown --size 800 "
        "--overlap 120 --mode parent_child`.",
        "- Sweep model and retrieval depth: "
        "`make sweep GOLDSET=<goldset.jsonl> SWEEP_RAG_GRID=top_k=3,5,8` then "
        "`make recommend`.",
        "- Tune RAG parameters on the tuning split and score only the final split winner: "
        "`llb tune --model <model> --backend <backend> --goldset <goldset.jsonl>`.",
        "- Generate and compare prompt packages: "
        "`make prompt-system-prepare PROMPT_SYSTEM_CORPUS=<corpus-dir>`, then run "
        "`make run-eval PROMPT_SYSTEM_ID=<id>`.",
        "- After a local run, classify misses and get evidence-backed actions: "
        "`make analyze-misses RUN_DIR=<run-eval-bundle> PROBE_TOP_K=3,8`.",
        "- External-service artifact manual: "
        "`docs/guides/data-prep/external-ai-service-artifacts.md`.",
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")
