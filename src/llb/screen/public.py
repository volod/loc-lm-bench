"""Tier-1 public screen via lm-evaluation-harness-uk (public screen, + Belebele wiring from verified gold-set ledger).

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
import re
from pathlib import Path
from typing import Any, Callable

from llb.contracts import IsolationOutcome, ScreenReport, ScreenTaskResult

_LOG = logging.getLogger(__name__)

TRACK_LOGPROB = "logprob"
TRACK_GENERATION = "generation"

# Default UA task ids per track, CONFIRMED against the stock lm-eval task registry (v0.4.12):
#  - logprob (multiple_choice -> loglikelihood): the standard UA leaderboard MCQ set.
#  - generation (generate_until): `global_piqa_prompted_ukr_cyrl` is the only stock UA
#    generate-until task; there is no stock UA SQuAD. Override per harness build with `--tasks`.
LOGPROB_TASKS = [
    "belebele_ukr_Cyrl",
    "arc_uk",
    "hellaswag_uk",
    "m_mmlu_uk",
]  # MCQ (verified gold-set ledger)
GENERATION_TASKS = ["global_piqa_prompted_ukr_cyrl"]  # generate-until (Ollama-compatible)

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
        tasks = list(LOGPROB_TASKS) + list(GENERATION_TASKS) + list(extra_tasks or [])
        return TRACK_LOGPROB, list(dict.fromkeys(tasks))
    tasks = list(GENERATION_TASKS) + list(extra_tasks or [])
    return TRACK_GENERATION, list(dict.fromkeys(tasks))


def build_lm_eval_command(
    model: str,
    base_url: str,
    tasks: list[str],
    output_path: Path | str,
    *,
    limit: int | None = None,
    track: str = TRACK_LOGPROB,
) -> list[str]:
    """The lm-eval `local-completions` argv against an OpenAI-compatible endpoint.

    Tokenizer handling is TRACK-aware: loglikelihood (MCQ) tasks need a tokenizer to compute
    context lengths, so the logprob track points lm-eval at the model's HF tokenizer (a vLLM
    `model` is a HF repo id). The generation track sets `tokenizer_backend=None` -- generation
    needs no local tokenizer, and an Ollama tag (e.g. `llama3.2:3b`) is not a valid HF repo id.
    """
    tok = "tokenizer_backend=None" if track == TRACK_GENERATION else f"tokenizer={model}"
    model_args = f"base_url={base_url.rstrip('/')}/completions,model={model},num_concurrent=1,{tok}"
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
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        name = key.split(",", 1)[0]
        if name.endswith("_stderr"):
            continue
        flat.setdefault(name, value)
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
    # lm-eval treats --output_path as a DIRECTORY and writes <path>/<model>/results_<ts>.json,
    # so pass a per-model dir (not a .json file) and let the runner locate the results JSON.
    model_out = out / _safe(model)
    model_out.mkdir(parents=True, exist_ok=True)
    cmd = build_lm_eval_command(model, base_url, tasks, model_out, limit=limit, track=track)
    runner = runner or _default_lm_eval_runner
    _LOG.info("[screen-public] %s track=%s tasks=%s", model, track, ",".join(tasks))
    results_json = runner(cmd)
    report = parse_results(results_json, tasks, model=model, backend=backend, track=track)
    if report["missing"]:
        _LOG.warning("[screen-public] PARTIAL: %s missing tasks %s", model, report["missing"])
    return report


def run_screen_isolated(
    backend: str,
    run_screen_fn: Callable[[], ScreenReport],
    *,
    vram_reader: Callable[[], int] | None = None,
    pid_usage_reader: Callable[[], dict[int, int]] | None = None,
    gpu_sampler: Callable[[], list[Any]] | None = None,
    sleep: Callable[[float], None] | None = None,
    vram_tolerance_mb: int | None = None,
    cooldown_temp_c: int | None = None,
    cooldown_max_s: float | None = None,
) -> tuple[ScreenReport, "IsolationOutcome"]:
    """Run a screen under the SAME isolation contract as a Tier-2 sweep cell (public screen).

    `run_screen_fn` launches the backend, runs lm-eval against it, and kills the backend (its own
    process). This REUSES the shared `executor.isolation.isolate_cell` primitive -- VRAM baseline,
    PID-attributed reclaim gate, capped thermal cooldown -- so the screen and the sweep share one
    isolation contract instead of duplicating it.
    """
    from llb.executor.isolation import (
        DEFAULT_COOLDOWN_MAX_S,
        DEFAULT_COOLDOWN_TEMP_C,
        isolate_cell,
    )
    from llb.executor.vram import DEFAULT_TOLERANCE_MB

    return isolate_cell(
        run_screen_fn,
        backend=backend,
        vram_reader=vram_reader,
        pid_usage_reader=pid_usage_reader,
        gpu_sampler=gpu_sampler,
        sleep=sleep,
        vram_tolerance_mb=DEFAULT_TOLERANCE_MB if vram_tolerance_mb is None else vram_tolerance_mb,
        cooldown_temp_c=DEFAULT_COOLDOWN_TEMP_C if cooldown_temp_c is None else cooldown_temp_c,
        cooldown_max_s=DEFAULT_COOLDOWN_MAX_S if cooldown_max_s is None else cooldown_max_s,
    )


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


def _default_lm_eval_runner(cmd: list[str]) -> dict[str, Any]:
    """Run lm-eval as a subprocess and read its results JSON (needs lm-eval-harness installed)."""
    import subprocess

    out_idx = cmd.index("--output_path") + 1
    output_path = Path(cmd[out_idx])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"lm_eval exited {proc.returncode}: {' | '.join(tail)}")
    # lm-eval writes <output_path>/<model>/results_<ts>.json; accept the file path too.
    if output_path.is_dir():
        candidates = sorted(output_path.glob("**/results*.json"))
    elif output_path.exists():
        candidates = [output_path]
    else:
        candidates = sorted(output_path.parent.glob(f"{output_path.name}/**/results*.json"))
    if not candidates:
        raise RuntimeError(f"lm_eval produced no results JSON under {output_path}")
    data: dict[str, Any] = json.loads(candidates[-1].read_text(encoding="utf-8"))
    return data
