"""Run one candidate's smoke on one page and record what happened, whatever happened.

Candidates run one at a time (the GPU isolation contract): the device baseline is read before each,
and VRAM is waited back to it afterwards, so one candidate's leftovers never land in the next one's
reading. A candidate that is not installed, fails, times out, silently falls back, or needs more
VRAM than the device has is RECORDED with that status rather than dropped.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any

from llb.prep.book.models import model_problem
from llb.prep.book.survey import envs
from llb.prep.book.survey.candidates import (
    ROUTE_NONE,
    ROUTE_OLLAMA,
    ROUTE_SYSTEM,
    ROUTE_VENV,
    SMOKE_OLLAMA,
    SMOKE_SERVED,
    Candidate,
    Smoke,
)
from llb.prep.book.survey.fixture import SmokePage
from llb.prep.book.survey import served, vllm_server
from llb.prep.book.survey.measure import (
    Usage,
    device_used_reader,
    run_measured,
    wait_for_reclaim,
)
from llb.prep.book.survey.ollama_smoke import read_page
from llb.prep.book.survey.page_content import FixtureTruth
from llb.prep.book.survey.results import (
    OOM,
    STATUS_FAILED,
    STATUS_NOT_INSTALLED,
    STATUS_NOT_RUN,
    STATUS_OK,
    STATUS_OVER_BUDGET,
    STATUS_TIMEOUT,
    SmokeResult,
)
from llb.prep.book.survey.score import score_reading

RUNNERS_DIR = Path(__file__).resolve().parent / "runners"
READING_NAME = "reading.txt"
LOG_NAME = "smoke.log"
POST_OCR_TIMEOUT_S = 120


def _format(argv: tuple[str, ...], values: dict[str, str]) -> list[str]:
    return [part.format(**values) for part in argv]


def _first(out_dir: Path, pattern: str) -> Path | None:
    matches = sorted(path for path in out_dir.glob(pattern) if path.is_file())
    return matches[0] if matches else None


def _post_ocr(image: Path, out_dir: Path) -> str:
    """Read a preprocessed image with Tesseract, so preprocessing is judged by recognition."""
    if shutil.which("tesseract") is None:
        return ""
    base = out_dir / "post_ocr"
    subprocess.run(
        ["tesseract", str(image), str(base), "--psm", "3", "-l", "eng"],
        capture_output=True,
        timeout=POST_OCR_TIMEOUT_S,
        check=False,
    )
    text_path = base.with_suffix(".txt")
    return text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else ""


def _venv_problem(candidate: Candidate, data_dir: Path, values: dict[str, str]) -> str | None:
    """Why the candidate's environment is missing; fills `{bin}` and `{python}` when present."""
    installed = envs.installed_env(data_dir, candidate.env or "")
    if installed is None:
        hint = f"make book-survey-install BOOK_SURVEY_ENVS={candidate.env}"
        return f"environment book-{candidate.env} is not installed ({hint})"
    values.update(bin=str(installed.python.parent), python=str(installed.python))
    return None


def _smoke_problem(candidate: Candidate, smoke: Smoke) -> str | None:
    """Why the smoke's system binary, vLLM server, or Ollama model is unavailable."""
    if candidate.route == ROUTE_SYSTEM and shutil.which(smoke.argv[0]) is None:
        return f"{smoke.argv[0]} is not on PATH (make apt-deps)"
    if smoke.server is not None:
        return vllm_server.docker_problem(smoke.server.image)
    if candidate.route == ROUTE_OLLAMA:
        problem = model_problem(f"ollama:{smoke.model}")
        return None if problem is None else f"{problem} (ollama pull {smoke.model})"
    return None


def _availability(candidate: Candidate, data_dir: Path) -> tuple[str | None, dict[str, str]]:
    """(why it cannot run, placeholder values when it can)."""
    values = {"runners": str(RUNNERS_DIR)}
    if candidate.route == ROUTE_NONE:
        return candidate.blocker or "not installable here", values
    if candidate.route == ROUTE_VENV and (problem := _venv_problem(candidate, data_dir, values)):
        return problem, values
    if candidate.smoke is not None:
        return _smoke_problem(candidate, candidate.smoke), values
    return None, values


def _status_from_log(smoke: Smoke, returncode: int | None, timed_out: bool, log: str) -> str:
    if timed_out:
        return STATUS_TIMEOUT
    if OOM.search(log):
        return STATUS_OVER_BUDGET
    if returncode != 0 or (smoke.fail_on and smoke.fail_on in log):
        return STATUS_FAILED
    return STATUS_OK


def _failure_detail(smoke: Smoke, log: str, usage: Usage) -> str:
    """Why a run failed: the fallback line when one matched, else its exit code and log tail."""
    if smoke.fail_on:
        for line in log.splitlines():
            if smoke.fail_on in line:
                return line.strip()
    ended = "timed out" if usage.timed_out else f"exit code {usage.returncode}"
    return f"{ended}; log tail: {usage.output_tail[-1500:]}"


def _run_process(
    candidate: Candidate,
    smoke: Smoke,
    page: SmokePage,
    out_dir: Path,
    data_dir: Path,
    values: dict[str, str],
    baseline: int | None,
) -> SmokeResult:
    values = {**values, "image": str(page.image), "pdf": str(page.pdf), "out": str(out_dir)}
    bin_dir = Path(values["bin"]) if "bin" in values else None
    env = envs.tool_environment(data_dir, bin_dir)
    env.update({name: value.format(**values) for name, value in smoke.env})
    usage = run_measured(
        _format(smoke.argv, values),
        cwd=out_dir,
        env=env,
        timeout_s=smoke.timeout_s,
        log_path=out_dir / LOG_NAME,
        baseline_mb=baseline,
    )
    log = (out_dir / LOG_NAME).read_text(encoding="utf-8", errors="replace")
    status = _status_from_log(smoke, usage.returncode, usage.timed_out, log)
    text = ""
    if smoke.reading and (found := _first(out_dir, smoke.reading)):
        text = found.read_text(encoding="utf-8", errors="replace")
    if smoke.image_output and (image := _first(out_dir, smoke.image_output)):
        text = _post_ocr(image, out_dir)
    elif smoke.image_output and status == STATUS_OK:
        status = STATUS_FAILED
    if status == STATUS_OK and not text.strip():
        status = STATUS_FAILED
    return SmokeResult(
        candidate=candidate.id,
        page=page.label,
        status=status,
        seconds=usage.seconds,
        peak_rss_mb=usage.peak_rss_mb,
        peak_vram_tree_mb=usage.peak_vram_tree_mb,
        peak_vram_rise_mb=usage.peak_vram_rise_mb,
        reading=text,
        error=None if status == STATUS_OK else _failure_detail(smoke, log, usage),
    )


def _run_ollama(
    candidate: Candidate, smoke: Smoke, page: SmokePage, baseline: int | None
) -> SmokeResult:
    reading = read_page(
        smoke.model or "", smoke.prompt or "", page.image, smoke.timeout_s, baseline
    )
    ok = reading.error is None and bool(reading.text.strip())
    return SmokeResult(
        candidate=candidate.id,
        page=page.label,
        status=STATUS_OK if ok else STATUS_FAILED,
        seconds=reading.seconds,
        peak_rss_mb=reading.ram_resident_mb,
        peak_vram_rise_mb=reading.peak_vram_rise_mb,
        vram_resident_mb=reading.vram_resident_mb,
        reading=reading.text,
        error=reading.error or (None if ok else "empty response"),
    )


def _execute(
    candidate: Candidate,
    smoke: Smoke,
    page: SmokePage,
    out_dir: Path,
    data_dir: Path,
    values: dict[str, str],
    baseline: int | None,
) -> SmokeResult:
    if smoke.kind == SMOKE_OLLAMA:
        return _run_ollama(candidate, smoke, page, baseline)
    if smoke.kind == SMOKE_SERVED:
        return served.run_served_read(candidate, smoke, page, values["server"])
    return _run_process(candidate, smoke, page, out_dir, data_dir, values, baseline)


def score_dict(truth: FixtureTruth, reading: str) -> dict[str, Any]:
    """A reading's smoke score in the summary's shape."""
    scored = score_reading(truth, reading)
    return {
        "word_recall": scored.word_recall,
        "char_agreement": scored.char_agreement,
        "dangerous_exact": scored.dangerous_exact,
        "dangerous_total": len(scored.dangerous),
        "dangerous": scored.dangerous,
    }


def run_smoke(candidate: Candidate, page: SmokePage, out_dir: Path, data_dir: Path) -> SmokeResult:
    """Smoke `candidate` on `page`, writing its outputs under `out_dir`."""
    blocker, values = _availability(candidate, data_dir)
    smoke = candidate.smoke
    if blocker is not None or smoke is None:
        status = STATUS_NOT_RUN if candidate.route == ROUTE_NONE else STATUS_NOT_INSTALLED
        return SmokeResult(candidate.id, page.label, status, error=blocker or "no smoke declared")
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = device_used_reader()
    baseline = reader() if reader is not None else None
    args = (candidate, smoke, page, out_dir, data_dir, values, baseline)
    if smoke.server is not None:
        result = served.run_with_server(*args, execute=_execute)
    else:
        result = _execute(*args)
    result.vram_residual_mb = wait_for_reclaim(baseline)
    if result.reading:
        (out_dir / READING_NAME).write_text(result.reading, encoding="utf-8")
        if page.truth is not None:
            result.score = score_dict(page.truth, result.reading)
        # The text stays in its own file (a real page's text never enters the summary).
        result.reading = READING_NAME
    return result
