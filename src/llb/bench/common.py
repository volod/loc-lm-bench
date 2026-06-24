"""Shared scaffolding for the Milestone 5 benchmark categories (REUSE, not a new platform).

Each M5 category (security / tooling / agentic / text-analysis) is a TASK FAMILY on the existing
substrate, not a new framework: it drives a model through one injectable `complete` callable
(prompt -> raw text), scores objectively, aggregates into a `ModelResult` under its OWN Tier
(never cross-ranked with the RAG board -- the `aggregate` Tier guard enforces this), and persists
the canonical manifest + per-case scores exactly like `run-eval`. This module factors the parts
every category shares so each category module stays small and focused (AGENTS.md modularity).

Everything here is pure or injectable: `local_complete` / `launcher_complete` build the production
`complete` over an OpenAI-compatible endpoint, but a category run takes any `complete`, so a fake
endpoint proves the whole flow without a GPU.
"""

import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from llb.backends.base import BackendLauncher
from llb.config import RunConfig
from llb.contracts import (
    ChatMessage,
    JsonObject,
    JudgeInputRecord,
    JudgeScore,
    JudgeStatus,
    RunMetrics,
    RunPaths,
)
from llb.scoring.aggregate import (
    BoardRow,
    ModelResult,
    format_board,
    rank_board,
    ranking_policy_note,
)
from llb.scoring.judge import DEFAULT_THRESHOLD, JudgeOutcome, run_judge
from llb.tracking.manifest import RunManifest, persist_run

LLMComplete = Callable[[str], str]  # prompt -> raw completion text
JudgeScorer = Callable[[list[JudgeInputRecord], str], list[JudgeScore]]  # (records, model)->scores
Mirror = Callable[[RunManifest, Path], None]
_R = TypeVar("_R")

_RUN_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S.%fZ"


def new_run_timestamp() -> tuple[str, str]:
    """A fresh (run_id, timestamped-dir-name) pair, matching the run-eval bundle convention."""
    run_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).strftime(_RUN_TIMESTAMP_FORMAT)
    return run_id, f"{now}-{run_id}"


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 for an empty sequence (the category headline convention)."""
    return sum(values) / len(values) if values else 0.0


def run_gated_judge(
    records: list[JudgeInputRecord],
    *,
    judge_model: str | None,
    judge_rho: float | None,
    threshold: float = DEFAULT_THRESHOLD,
    scorer: JudgeScorer | None = None,
    base_url: str | None = None,
) -> JudgeOutcome:
    """Run the calibrated, GATED judge for an M5 category (objective stays the headline).

    A thin reuse of `scoring.judge.run_judge`: the outcome carries per-record scores ONLY when a
    judge is configured AND trusted (`judge_rho >= threshold`); otherwise it is demoted (objective
    ranks alone) and the caller reads `outcome.reason`. `scorer` is injectable -- a fake in tests --
    so the wiring is provable without DeepEval / an endpoint / a GPU; the default scorer is the
    DeepEval judge bound to `base_url`, imported lazily so this stays light in the base install.
    """

    def _default(recs: list[JudgeInputRecord], model: str) -> list[JudgeScore]:
        from llb.scoring.judge import deepeval_scorer

        return deepeval_scorer(recs, model, base_url=base_url)

    return run_judge(records, judge_model, judge_rho, threshold, scorer=scorer or _default)


def local_complete(
    model: str,
    base_url: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> LLMComplete:
    """A `complete` over an already-running OpenAI-compatible endpoint (no launch). Heavy imports
    stay lazy; transport errors map to an empty string via `chat_once`'s normalized result."""
    from llb.backends.openai_client import chat_once, make_client

    client = make_client(base_url)

    def complete(prompt: str) -> str:
        msgs: list[ChatMessage] = [{"role": "user", "content": prompt}]
        return chat_once(
            client, model, msgs, max_tokens=max_tokens, temperature=temperature, timeout=timeout
        ).text

    return complete


def launcher_complete(
    launcher: BackendLauncher,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> LLMComplete:
    """A `complete` over an already-started `BackendLauncher` (its OpenAI-compatible chat)."""

    def complete(prompt: str) -> str:
        msgs: list[ChatMessage] = [{"role": "user", "content": prompt}]
        return launcher.chat(msgs, max_tokens, temperature, timeout).text

    return complete


def drive_with_backend(
    cfg: RunConfig,
    run: Callable[[LLMComplete], _R],
    *,
    base_url: str | None = None,
    vram_reader: Callable[[], int] | None = None,
    pid_usage_reader: Callable[[], dict[int, int]] | None = None,
) -> _R:
    """Build the candidate's `complete` for the chosen endpoint and execute `run(complete)`.

    A running endpoint (`base_url`) or Ollama is called directly; a VRAM-owning backend
    (vllm / llamacpp) is launched and the whole `run` executes under the shared `isolate_cell`
    contract (PID-attributed VRAM-reclaim gate + capped thermal cooldown), so every M5 category
    honors the SAME isolation contract as the RAG sweep.
    """
    if base_url is not None or cfg.backend == "ollama":
        url = base_url or f"{cfg.ollama_host.rstrip('/')}/v1"
        return run(local_complete(cfg.model, url, timeout=cfg.request_timeout_s))

    from llb.executor.isolation import isolate_cell
    from llb.executor.runner import _make_launcher

    launcher = _make_launcher(cfg)

    def work() -> _R:
        with launcher:
            return run(launcher_complete(launcher, timeout=cfg.request_timeout_s))

    result, _outcome = isolate_cell(
        work,
        backend=cfg.backend,
        vram_reader=vram_reader,
        pid_usage_reader=pid_usage_reader,
    )
    return result


def category_result(
    *,
    model: str,
    backend: str,
    tier: str,
    case_objectives: Sequence[float],
    reliability: float = 1.0,
    tokens_per_s: float = 0.0,
    peak_vram_mb: float | None = None,
    judge_score: float | None = None,
    case_judge: Sequence[float] | None = None,
) -> ModelResult:
    """A `ModelResult` stamped with the category's Tier; `case_objectives` feed the bootstrap CI."""
    return ModelResult(
        model=model,
        backend=backend,
        objective_score=mean(case_objectives),
        n_cases=len(case_objectives),
        reliability=reliability,
        tokens_per_s=tokens_per_s,
        peak_vram_mb=peak_vram_mb,
        judge_score=judge_score,
        feasible=True,
        tier=tier,
        case_objectives=list(case_objectives),
        case_judge=list(case_judge or []),
    )


def render_board(
    results: list[ModelResult], *, judge_trusted: bool = False
) -> tuple[list[BoardRow], str]:
    """Rank the category's results under its Tier and render the ASCII board with its policy note."""
    rows = rank_board(results, judge_trusted=judge_trusted)
    table = format_board(rows, policy=ranking_policy_note(results, judge_trusted))
    return rows, table


def persist_category_run(
    *,
    method: str,
    data_dir: Path | str,
    run_name: str,
    config: JsonObject,
    metrics: RunMetrics,
    case_rows: Sequence[Mapping[str, object]],
    judge: JudgeStatus | None = None,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Write one category run bundle under `$DATA_DIR/<method>/<timestamp>/` (manifest + scores),
    atomically published exactly like `run-eval`. `config` carries the category + tier provenance."""
    run_id, run_timestamp = new_run_timestamp()
    out_dir = Path(data_dir) / method / run_timestamp
    staging = out_dir.with_name(f".{out_dir.name}.tmp")
    manifest = RunManifest(
        run_id=run_id,
        run_name=run_name,
        split="final",
        config=config,
        metrics=metrics,
        judge=judge,
        n_cases=len(case_rows),
    )
    return persist_run(manifest, list(case_rows), out_dir, mirror=mirror, staging_dir=staging)
