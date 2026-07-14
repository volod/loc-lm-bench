"""Top-level orchestration: score one model's task-completion rate over the deterministic
tool-world under TIER_AGENTIC, with the opt-in gated trajectory-quality signal alongside.

`run_agentic` wires the episode runner, objective scorer, gated judge, and persistence together;
`load_tasks_file` reads a task set from disk.
"""

import json
from pathlib import Path
from typing import Any

from llb.bench.agentic.episode import _resolve_harness, _run_episodes, _score_episodes
from llb.bench.agentic.model import (
    DEFAULT_MAX_STEPS,
    HARNESS_LOOP,
    AgenticRun,
    AgenticTask,
    Harness,
    _AgenticPersistInput,
    _JudgeConfig,
)
from llb.bench.agentic.persist import _persist_agentic_run
from llb.bench.agentic.trajectory import _run_trajectory_judge
from llb.bench.common import (
    DEFAULT_THRESHOLD,
    JudgeScorer,
    LLMComplete,
    Mirror,
    category_result,
    render_board,
    verified_data_config,
)
from llb.bench.common_backend import ThroughputMeter
from llb.scoring.aggregate import TIER_AGENTIC


def run_agentic(
    tasks: list[AgenticTask],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    max_steps: int = DEFAULT_MAX_STEPS,
    harness_name: str = HARNESS_LOOP,
    harness: "Harness | None" = None,
    prompt_system: str | None = None,
    judge_model: str | None = None,
    judge_rho: float | None = None,
    judge_threshold: float = DEFAULT_THRESHOLD,
    judge_scorer: JudgeScorer | None = None,
    judge_base_url: str | None = None,
    data_dir: Path | str | None = None,
    run_name: str = "agentic",
    persist: bool = True,
    mirror: Mirror | None = None,
    data_verified: bool = False,
    verification_ref: str | None = None,
    meter: ThroughputMeter | None = None,
) -> AgenticRun:
    """Score one model's task-completion rate over the deterministic tool-world under TIER_AGENTIC.

    Objective completion-rate is the headline. When a judge is configured AND trusted
    (`judge_rho >= judge_threshold`), an opt-in trajectory-quality signal is recorded ALONGSIDE
    (per-case + mean + CI) but never folded into the headline; otherwise the judge is demoted and
    completion-rate ranks alone. `judge_scorer` is injectable for tests. A `meter` (populated by the
    endpoint `complete`) supplies the run's real generation tok/s.
    """
    if not tasks:
        raise SystemExit("no agentic tasks provided")
    verification_cfg = verified_data_config(
        data_verified=data_verified, verification_ref=verification_ref
    )
    episodes = _run_episodes(tasks, complete, _resolve_harness(harness_name, harness), max_steps)
    scored = _score_episodes(tasks, episodes)
    judge_config = _JudgeConfig(
        model=judge_model,
        rho=judge_rho,
        threshold=judge_threshold,
        scorer=judge_scorer,
        base_url=judge_base_url,
    )
    quality = _run_trajectory_judge(tasks, episodes, scored.rows, judge_config)
    tokens_per_s = meter.tokens_per_s if meter is not None else 0.0
    result = category_result(
        model=model,
        backend=backend,
        tier=TIER_AGENTIC,
        case_objectives=scored.case_success,
        reliability=scored.reliability,
        tokens_per_s=tokens_per_s,
    )
    board, table = render_board([result])
    paths = (
        _persist_agentic_run(
            _AgenticPersistInput(
                data_dir=data_dir,
                run_name=run_name,
                model=model,
                backend=backend,
                harness_name=harness_name,
                prompt_system=prompt_system,
                n_tasks=len(tasks),
                max_steps=max_steps,
                result=result,
                scored=scored,
                quality=quality,
                judge_config=judge_config,
                verification_cfg=verification_cfg,
                tokens_per_s=tokens_per_s,
                mirror=mirror,
            )
        )
        if persist
        else None
    )
    return AgenticRun(
        result=result,
        episodes=episodes,
        rows=scored.rows,
        board=board,
        table=table,
        completion_ci=scored.completion_ci,
        mean_steps=scored.mean_steps,
        mean_tool_calls=scored.mean_tool_calls,
        paths=paths,
        trajectory_quality=quality.value,
        trajectory_quality_ci=quality.ci,
        judge_trusted=quality.outcome.trusted,
        judge_reason=quality.outcome.reason,
        judge_diagnostics=quality.outcome.diagnostics,
    )


def load_tasks_file(path: Path | str) -> list[AgenticTask]:
    """Load an agentic task set (a JSON array of task records)."""
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array of agentic tasks")
    return [AgenticTask.from_record(r) for r in raw]
