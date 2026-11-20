"""A survey run: every selected candidate on every selected page, one at a time, then a summary.

Layout, under `$DATA_DIR/book-ocr/<run_timestamp>/survey/`:

    host.json                       device, VRAM budget, driver, and the survey cutoff
    pages/<label>/page.{jpg,png}    the smoke page (synthetic, or a page of a real scan)
    <label>/<candidate>/            the candidate's own outputs, `smoke.log`, `reading.txt`
    survey.json                     every result, including not-installed and failed rows
    survey.md                       the same as a table

Both summaries are diagnostic outputs that nothing reads back (the shortlist is written into the
docs from them), so they stay unregistered, as the data-prep policy does for reports.
"""

import json
import logging
import platform
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llb.core.fsutil import atomic_write_text
from llb.prep.book.survey.candidates import EXCLUDE_NEWER, SURVEY_DATE, Candidate
from llb.prep.book.survey.fixture import TRUTH_NAME, SmokePage, read_truth
from llb.prep.book.survey.measure import device_total_mb
from llb.prep.book.survey.results import STATUS_OK, SmokeResult
from llb.prep.book.survey.smoke import run_smoke, score_dict

_LOG = logging.getLogger(__name__)

SURVEY_DIRNAME = "survey"
PAGES_DIRNAME = "pages"
SUMMARY_JSON = "survey.json"
SUMMARY_MD = "survey.md"
HOST_JSON = "host.json"


def host_facts() -> dict[str, Any]:
    from llb.backends.hardware import detect_gpus, detect_ram_mb

    gpus = detect_gpus()
    return {
        "survey_date": SURVEY_DATE,
        "exclude_newer": EXCLUDE_NEWER,
        "platform": platform.platform(),
        "gpus": [asdict(gpu) for gpu in gpus],
        "vram_budget_mb": device_total_mb(),
        "ram_mb": detect_ram_mb(),
    }


def _cell(value: object) -> str:
    return "-" if value is None else str(value)


def _row(result: SmokeResult) -> str:
    score = result.score or {}
    dangerous = f"{score['dangerous_exact']}/{score['dangerous_total']}" if score else "-"
    vram = (
        result.vram_resident_mb
        if result.vram_resident_mb is not None
        else (result.peak_vram_tree_mb or result.peak_vram_rise_mb)
    )
    return (
        f"| {result.candidate} | {result.page} | {result.status} | {_cell(result.seconds)} | "
        f"{_cell(vram)} | {_cell(result.peak_vram_rise_mb)} | {_cell(result.peak_rss_mb)} | "
        f"{_cell(score.get('word_recall'))} | {_cell(score.get('char_agreement'))} | "
        f"{dangerous} |"
    )


def render_markdown(host: dict[str, Any], results: Sequence[SmokeResult]) -> str:
    lines = [
        "# Book recovery tool survey (smoke)",
        "",
        f"Survey date {host['survey_date']}, packages resolved as of {host['exclude_newer']}; "
        f"VRAM budget {host['vram_budget_mb']} MB.",
        "",
        "| candidate | page | status | seconds | VRAM MB | device rise MB | RAM MB | "
        "word recall | char agreement | dangerous exact |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *(_row(result) for result in results),
        "",
    ]
    failures = [result for result in results if result.status != STATUS_OK]
    if failures:
        lines += ["## Not run, failed, or over budget", ""]
        for result in failures:
            reason = (result.error or "").strip().splitlines()
            lines.append(
                f"- {result.candidate} on {result.page}: {result.status} -- "
                f"{reason[-1] if reason else 'no detail'}"
            )
        lines.append("")
    return "\n".join(lines)


def run_survey(
    candidates: Sequence[Candidate], pages: Sequence[SmokePage], survey_dir: Path, data_dir: Path
) -> list[SmokeResult]:
    """Smoke every candidate on every page, writing results as it goes."""
    host = host_facts()
    atomic_write_text(survey_dir / HOST_JSON, json.dumps(host, indent=2) + "\n")
    results: list[SmokeResult] = []
    for page in pages:
        for candidate in candidates:
            _LOG.info("[survey] %s on %s", candidate.id, page.label)
            result = run_smoke(candidate, page, survey_dir / page.label / candidate.id, data_dir)
            _LOG.info("[survey]   -> %s (%s s)", result.status, result.seconds)
            results.append(result)
            _write_summaries(survey_dir, host, results)
    return results


def _write_summaries(
    survey_dir: Path, host: dict[str, Any], results: Sequence[SmokeResult]
) -> None:
    payload = {"host": host, "results": [asdict(r) for r in results]}
    atomic_write_text(
        survey_dir / SUMMARY_JSON, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    atomic_write_text(survey_dir / SUMMARY_MD, render_markdown(host, results))


def rescore_survey(survey_dir: Path) -> list[SmokeResult]:
    """Recompute every synthetic-page score from the saved readings; no candidate re-runs."""
    payload = json.loads((survey_dir / SUMMARY_JSON).read_text(encoding="utf-8"))
    results = [SmokeResult(**row) for row in payload["results"]]
    for result in results:
        truth_path = survey_dir / PAGES_DIRNAME / result.page / TRUTH_NAME
        reading = survey_dir / result.page / result.candidate / (result.reading or "")
        if result.reading and truth_path.exists() and reading.is_file():
            text = reading.read_text(encoding="utf-8", errors="replace")
            result.score = score_dict(read_truth(truth_path), text)
    _write_summaries(survey_dir, payload["host"], results)
    return results
