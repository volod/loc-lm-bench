"""Tier-1 public screen via lm-evaluation-harness-uk (M3.1, + Belebele wiring from M3.9).

The Tier-1 screen narrows the candidate field on PUBLIC UA benchmarks before the expensive
Tier-2 private eval. It drives lm-eval-harness through its `local-completions` model against an
already-launched OpenAI-compatible endpoint (the same launcher as the eval), so no model is
loaded twice.

Two TRACKS, never cross-ranked (a hard guard, like the Tier-1/Tier-2 split in `aggregate`):

  logprob     vLLM exposes token logprobs, so multiple-choice tasks are scored by
              loglikelihood (Belebele-uk and other MCQ).
  generation  Ollama / llama.cpp only generate text, so only generate-until tasks run
              (SQuAD-uk style QA). MCQ-by-loglikelihood is impossible here.

Comparing a loglikelihood accuracy against a generation exact-match would be meaningless, so
the track is recorded on every report and `assert_single_track` refuses to mix them.

COVERAGE is first-class: the report lists which requested tasks produced a result and which
did not, and `complete` is False whenever any are missing -- so a screen is never silently
partial (a task that errored or was skipped is visible, not dropped).

lm-eval is heavy and external; the actual run is injected (`runner=`), so task selection,
command building, results parsing, and coverage are pure and unit-tested without it.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable

from llb.contracts import ScreenReport, ScreenTaskResult

_LOG = logging.getLogger(__name__)

TRACK_LOGPROB = "logprob"
TRACK_GENERATION = "generation"

# Default UA task ids per track. These are the lm-eval-harness-uk task names; override via the
# CLI for your harness build (the names vary by harness version / fork).
LOGPROB_TASKS = ["belebele_ukr_Cyrl"]  # MCQ -> loglikelihood (Belebele-uk, M3.9)
GENERATION_TASKS = ["squad_uk"]  # generate-until QA

# Primary metric per task, in preference order (lm-eval emits several; we report one).
_METRIC_PREFERENCE = ["acc", "exact_match", "f1", "acc_norm"]

LmEvalRunner = Callable[[list[str]], dict[str, Any]]  # argv -> parsed results JSON


def supports_logprobs(backend: str) -> bool:
    """vLLM's OpenAI endpoint returns token logprobs (loglikelihood scoring); Ollama does not."""
    return backend == "vllm"


def select_tasks(backend: str, extra_tasks: list[str] | None = None) -> tuple[str, list[str]]:
    """The (track, tasks) for a backend: logprob track gets MCQ + generation; generation track
    gets generation-only (MCQ-by-loglikelihood is impossible without logprobs)."""
    if supports_logprobs(backend):
        return TRACK_LOGPROB, list(LOGPROB_TASKS) + list(GENERATION_TASKS) + list(extra_tasks or [])
    return TRACK_GENERATION, list(GENERATION_TASKS) + list(extra_tasks or [])


def build_lm_eval_command(
    model: str,
    base_url: str,
    tasks: list[str],
    output_path: Path | str,
    *,
    limit: int | None = None,
) -> list[str]:
    """The lm-eval `local-completions` argv against an OpenAI-compatible endpoint."""
    model_args = f"base_url={base_url.rstrip('/')}/completions,model={model},num_concurrent=1"
    cmd = [
        "lm_eval",
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--output_path",
        str(output_path),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return cmd


def _primary_metric(metrics: dict[str, Any]) -> tuple[str, float] | None:
    """Pick one headline metric from an lm-eval per-task result (keys look like 'acc,none')."""
    flat = {key.split(",")[0]: val for key, val in metrics.items() if not key.endswith("stderr")}
    for name in _METRIC_PREFERENCE:
        if name in flat and isinstance(flat[name], (int, float)):
            return name, float(flat[name])
    for name, val in flat.items():  # fall back to the first numeric metric
        if isinstance(val, (int, float)):
            return name, float(val)
    return None


def parse_results(
    results_json: dict[str, Any],
    requested_tasks: list[str],
    *,
    model: str,
    backend: str,
    track: str,
) -> ScreenReport:
    """Turn an lm-eval results dict into per-task scores + explicit coverage."""
    results_block = results_json.get("results", {})
    rows: list[ScreenTaskResult] = []
    covered: list[str] = []
    for task in requested_tasks:
        metrics = results_block.get(task)
        if not isinstance(metrics, dict):
            continue
        picked = _primary_metric(metrics)
        if picked is None:
            continue
        rows.append({"task": task, "metric": picked[0], "score": picked[1]})
        covered.append(task)
    missing = [t for t in requested_tasks if t not in covered]
    return {
        "model": model,
        "backend": backend,
        "track": track,
        "requested_tasks": requested_tasks,
        "results": rows,
        "covered": covered,
        "missing": missing,
        "complete": not missing,
    }


def run_screen(
    model: str,
    backend: str,
    base_url: str,
    *,
    extra_tasks: list[str] | None = None,
    output_dir: Path | str | None = None,
    limit: int | None = None,
    runner: LmEvalRunner | None = None,
) -> ScreenReport:
    """Run the Tier-1 screen for one model against its launched endpoint."""
    track, tasks = select_tasks(backend, extra_tasks)
    out = Path(output_dir) if output_dir is not None else Path(".data") / "screen"
    out.mkdir(parents=True, exist_ok=True)
    cmd = build_lm_eval_command(model, base_url, tasks, out / f"{_safe(model)}.json", limit=limit)
    runner = runner or _default_lm_eval_runner
    _LOG.info("[screen-public] %s track=%s tasks=%s", model, track, ",".join(tasks))
    results_json = runner(cmd)
    report = parse_results(results_json, tasks, model=model, backend=backend, track=track)
    if report["missing"]:
        _LOG.warning("[screen-public] PARTIAL: %s missing tasks %s", model, report["missing"])
    return report


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
    return name.replace("/", "_").replace(":", "_")


def _default_lm_eval_runner(cmd: list[str]) -> dict[str, Any]:
    """Run lm-eval as a subprocess and read its results JSON (needs lm-eval-harness installed)."""
    import subprocess

    out_idx = cmd.index("--output_path") + 1
    output_path = Path(cmd[out_idx])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"lm_eval exited {proc.returncode}: {' | '.join(tail)}")
    # lm-eval writes results_*.json under the output path (a dir) or to the file itself.
    candidates = (
        sorted(output_path.glob("**/results*.json")) if output_path.is_dir() else [output_path]
    )
    if not candidates:
        raise RuntimeError(f"lm_eval produced no results JSON under {output_path}")
    data: dict[str, Any] = json.loads(candidates[-1].read_text(encoding="utf-8"))
    return data
