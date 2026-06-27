"""agentic workflows runner -- in-sandbox tool execution under TIER_AGENTIC.

The agentic loop is the text analysis multi-hop controller pattern extended with TOOL CALLS + an in-sandbox
execution step: each step the model emits one tool call (reusing the tooling benchmark `parse_tool_call`), the
deterministic `ToolWorld` EXECUTES it, the observation is fed back, and the loop runs until the
model calls `finish` (or answers in prose) or the step budget is exhausted. Task success is an
OBJECTIVE assertion over the final env-state and/or the final answer; completion-rate is the
headline under `TIER_AGENTIC` and trajectory length + tool-call count are recorded as efficiency.

An OPT-IN, GATED judge adds a TRAJECTORY-QUALITY signal a deterministic check cannot cover (is the
final answer grounded in what the tools actually returned, and does it address the goal?). It is
recorded ALONGSIDE completion-rate -- never folded into the headline -- and only when the judge is
configured AND trusted (calibration `judge_rho >= threshold`, the judge calibration gate gate). The judge `scorer`
is injectable, so the wiring is provable with a FAKE judge (no DeepEval / endpoint / GPU), exactly
like the category expansion summarization faithfulness signal.

LangGraph is the fixed single agent harness for the design; this loop is the pure, langgraph-free
form of that controller -> execute -> controller cycle, so it is unit-tested from a FAKE
tool-calling endpoint with no GPU. The candidate is reached through an injectable `complete`.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from llb.bench.common import (
    DEFAULT_THRESHOLD,
    JudgeScorer,
    LLMComplete,
    Mirror,
    category_result,
    mean,
    persist_category_run,
    render_board,
    run_gated_judge,
    verified_data_config,
)
from llb.bench.tool_world import FINISH, ToolWorld, tool_catalog
from llb.contracts import (
    AgenticCaseRow,
    BoardRow,
    JudgeDiagnostics,
    JudgeInputRecord,
    JudgeScore,
    JudgeStatus,
    RunMetrics,
    RunPaths,
    ToolDef,
)
from llb.scoring.aggregate import TIER_AGENTIC, ModelResult, bootstrap_mean_ci
from llb.scoring.tooling import parse_tool_call

_LOG = logging.getLogger(__name__)

METHOD = "agentic"
DEFAULT_MAX_STEPS = 6

# Named harnesses (agentic harness comparison) -- the comparison axis under TIER_AGENTIC. `loop` is the pure,
# framework-free controller->execute->controller cycle (`run_episode`); `langgraph` compiles that
# same cycle as a LangGraph app; `crewai` drives it through a single-agent CrewAI crew. Holding
# the task set + ToolWorld + objective scoring + gated judge FIXED, the harness is the only
# variable, isolating "how much the agent framework itself moves the score".
HARNESS_LOOP = "loop"
HARNESS_LANGGRAPH = "langgraph"
HARNESS_CREWAI = "crewai"
HARNESS_NAMES = (HARNESS_LOOP, HARNESS_LANGGRAPH, HARNESS_CREWAI)

# The judge "question" for trajectory quality: a fixed UA intent that frames the agent's job, so
# answer-relevancy scores whether the final answer addresses the goal while faithfulness scores
# whether it stays grounded in the tool observations fed back as the retrieval context.
_TRAJECTORY_INTENT = "Виконай завдання інструментами та дай відповідь, підкріплену їх результатами."

STATUS_COMPLETED = "completed"
STATUS_INCOMPLETE = "incomplete"

# Success-assertion kinds (over the final env-state / answer).
ASSERT_FILE_EQUALS = "file_equals"
ASSERT_FILE_CONTAINS = "file_contains"
ASSERT_DB_EQUALS = "db_equals"
ASSERT_ANSWER_CONTAINS = "answer_contains"


@dataclass(frozen=True)
class AgenticTask:
    """One agentic task: a UA goal, an initial env, and the objective success assertions."""

    id: str
    prompt: str
    setup: dict[str, Any] = field(default_factory=dict)
    success: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "AgenticTask":
        success = record.get("success", [])
        if isinstance(success, dict):
            success = [success]
        return cls(
            id=str(record["id"]),
            prompt=str(record["prompt"]),
            setup=dict(record.get("setup", {}) or {}),
            success=[dict(a) for a in success],
        )


@dataclass(slots=True)
class Episode:
    success: bool
    status: str
    n_steps: int
    n_tool_calls: int
    answer: str
    world: ToolWorld
    transcript: list[tuple[str, dict[str, Any], str]]


class Harness(Protocol):
    """A pluggable agentic harness (agentic harness comparison): drive ONE task to a canonical `Episode`.

    Every harness takes the same `(task, complete, catalog, max_steps)` and returns the same
    `Episode` (final answer + tool-call transcript + final env-state), so `check_success`, the
    scorer, and the gated judge are UNCHANGED across harnesses -- the framework is the only
    variable. The pure loop (`run_episode`), the LangGraph app, and the CrewAI crew all conform.
    """

    def __call__(
        self,
        task: AgenticTask,
        complete: LLMComplete,
        catalog: dict[str, ToolDef],
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> Episode: ...


@dataclass(slots=True)
class AgenticRun:
    result: ModelResult
    episodes: list[Episode]
    rows: list[AgenticCaseRow]
    board: list[BoardRow]
    table: str
    completion_ci: tuple[float, float] | None
    mean_steps: float
    mean_tool_calls: float
    paths: RunPaths | None
    trajectory_quality: float | None = None  # mean gated-judge quality (None when not trusted/run)
    trajectory_quality_ci: tuple[float, float] | None = None
    judge_trusted: bool = False
    judge_reason: str = "no judge configured"
    judge_diagnostics: JudgeDiagnostics | None = None  # judge diagnostics zero-valued-judge observability


def _norm(value: Any) -> str:
    return str(value).strip().casefold()


def check_assertion(assertion: dict[str, Any], world: ToolWorld, answer: str) -> bool:
    """Evaluate one success assertion against the final env-state / answer."""
    kind = assertion.get("kind")
    if kind == ASSERT_FILE_EQUALS:
        return _norm(world.files.get(str(assertion.get("path", "")), "")) == _norm(
            assertion.get("value", "")
        )
    if kind == ASSERT_FILE_CONTAINS:
        return _norm(assertion.get("value", "")) in _norm(
            world.files.get(str(assertion.get("path", "")), "")
        )
    if kind == ASSERT_DB_EQUALS:
        return _norm(world.db.get(str(assertion.get("key", "")), "")) == _norm(
            assertion.get("value", "")
        )
    if kind == ASSERT_ANSWER_CONTAINS:
        return _norm(assertion.get("value", "")) in _norm(answer)
    return False


def check_success(task: AgenticTask, world: ToolWorld, answer: str) -> bool:
    """A task succeeds when EVERY planted assertion holds (an empty assertion list never passes)."""
    return bool(task.success) and all(check_assertion(a, world, answer) for a in task.success)


def build_agent_prompt(
    task: AgenticTask,
    catalog: dict[str, ToolDef],
    transcript: list[tuple[str, dict[str, Any], str]],
) -> str:
    """The next-step prompt: available tools, the task, and the running observation transcript."""
    tools_json = json.dumps(list(catalog.values()), ensure_ascii=False, indent=2)
    history = "\n".join(
        f"- {name}({json.dumps(args, ensure_ascii=False)}) -> {obs}"
        for name, args, obs in transcript
    )
    history_block = f"Виконані кроки:\n{history}\n\n" if transcript else ""
    return (
        "Ти агент, що покроково виконує завдання за допомогою інструментів.\n"
        f"Доступні інструменти (JSON-схеми):\n{tools_json}\n\n"
        f"Завдання: {task.prompt}\n\n"
        f"{history_block}"
        'На КОЖНОМУ кроці поверни ЛИШЕ один JSON-виклик {"name": <інструмент>, '
        '"arguments": {<аргументи>}}.\n'
        'Коли завдання виконано, виклич {"name": "finish", "arguments": {"answer": <відповідь>}}.\n'
    )


def run_episode(
    task: AgenticTask,
    complete: LLMComplete,
    *,
    catalog: dict[str, ToolDef] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Episode:
    """Drive one task to completion (or the step budget) in the deterministic sandbox.

    This is the pure `loop` harness: the controller->execute->controller cycle with no agent
    framework. `catalog` is injectable so every harness shares ONE tool catalog; it defaults to
    the canonical `tool_catalog()` (so existing callers are unchanged)."""
    world = ToolWorld.from_setup(task.setup)
    catalog = catalog if catalog is not None else tool_catalog()
    transcript: list[tuple[str, dict[str, Any], str]] = []
    answer = ""
    finished = False
    n_tool_calls = 0
    steps = 0
    for steps in range(1, max_steps + 1):
        raw = complete(build_agent_prompt(task, catalog, transcript))
        call = parse_tool_call(raw)
        if call is None:  # the model answered in prose -> treat as the final answer
            answer = raw.strip()
            finished = True
            break
        if call.name == FINISH:
            answer = str(call.arguments.get("answer", ""))
            finished = True
            break
        observation = world.execute(call.name, call.arguments)
        n_tool_calls += 1
        transcript.append((call.name, call.arguments, observation))
    success = check_success(task, world, answer)
    return Episode(
        success=success,
        status=STATUS_COMPLETED if finished else STATUS_INCOMPLETE,
        n_steps=steps,
        n_tool_calls=n_tool_calls,
        answer=answer,
        world=world,
        transcript=transcript,
    )


def _trajectory_records(
    tasks: list[AgenticTask], episodes: list[Episode]
) -> list[JudgeInputRecord]:
    """One (goal, final answer, [tool observations]) record per episode for the trajectory judge.

    The tool observations become the retrieval context, so faithfulness scores whether the final
    answer stays grounded in what the tools actually returned (a check the env-state assertions
    cannot make), while answer-relevancy scores whether it addresses the goal.
    """
    return [
        {
            "question": f"{_TRAJECTORY_INTENT}\n\nЗавдання: {task.prompt}",
            "answer": episode.answer,
            "contexts": [
                f"{name}({json.dumps(args, ensure_ascii=False)}) -> {obs}"
                for name, args, obs in episode.transcript
            ],
        }
        for task, episode in zip(tasks, episodes)
    ]


def trajectory_quality(score: JudgeScore) -> float:
    """Collapse the judge's two G-Eval signals into one trajectory-quality scalar: the answer is
    GROUNDED in the tool observations (faithfulness) AND addresses the goal (answer_relevancy)."""
    return (float(score["faithfulness"]) + float(score["answer_relevancy"])) / 2.0


def _row(task: AgenticTask, episode: Episode) -> AgenticCaseRow:
    return {
        "item_id": task.id,
        "status": episode.status,
        "success": 1.0 if episode.success else 0.0,
        "objective_score": 1.0 if episode.success else 0.0,
        "n_steps": episode.n_steps,
        "n_tool_calls": episode.n_tool_calls,
        "answer_preview": (episode.answer or "")[:280],
    }


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
) -> AgenticRun:
    """Score one model's task-completion rate over the deterministic tool-world under TIER_AGENTIC.

    Objective completion-rate is the headline. When a judge is configured AND trusted
    (`judge_rho >= judge_threshold`), an opt-in trajectory-quality signal is recorded ALONGSIDE
    (per-case + mean + CI) but never folded into the headline; otherwise the judge is demoted and
    completion-rate ranks alone. `judge_scorer` is injectable for tests.
    """
    if not tasks:
        raise SystemExit("no agentic tasks provided")
    verification_cfg = verified_data_config(
        data_verified=data_verified, verification_ref=verification_ref
    )
    if harness is None:
        from llb.bench.harness import get_harness

        harness = get_harness(harness_name)
    catalog = tool_catalog()
    episodes = [harness(task, complete, catalog, max_steps=max_steps) for task in tasks]
    rows = [_row(task, ep) for task, ep in zip(tasks, episodes)]
    case_success = [1.0 if ep.success else 0.0 for ep in episodes]

    # Opt-in, gated trajectory-quality signal (objective completion-rate stays the headline).
    outcome = run_gated_judge(
        _trajectory_records(tasks, episodes),
        judge_model=judge_model,
        judge_rho=judge_rho,
        threshold=judge_threshold,
        scorer=judge_scorer,
        base_url=judge_base_url,
    )
    quality: float | None = None
    quality_ci: tuple[float, float] | None = None
    if outcome.trusted and outcome.scores:
        per_case = [trajectory_quality(s) for s in outcome.scores]
        for row, value in zip(rows, per_case):
            row["trajectory_quality"] = round(value, 6)
        quality = round(mean(per_case), 6)
        quality_ci = bootstrap_mean_ci(per_case)
    elif judge_model is not None:
        _LOG.info("[agentic] judge demoted (%s); objective completion ranks alone", outcome.reason)

    reliability = sum(1 for ep in episodes if ep.status == STATUS_COMPLETED) / len(episodes)
    mean_steps = mean([ep.n_steps for ep in episodes])
    mean_tool_calls = mean([ep.n_tool_calls for ep in episodes])
    result = category_result(
        model=model,
        backend=backend,
        tier=TIER_AGENTIC,
        case_objectives=case_success,
        reliability=reliability,
    )
    completion_ci = bootstrap_mean_ci(case_success)
    board, table = render_board([result])

    paths: RunPaths | None = None
    if persist and data_dir is not None:
        metrics: RunMetrics = {
            "objective_score": result.objective_score,  # completion rate
            "reliability": reliability,
            "tokens_per_s": 0.0,
        }
        config = {
            "model": model,
            "backend": backend,
            "tier": TIER_AGENTIC,
            "category": "agentic",
            "harness": harness_name,  # loop | langgraph | crewai (board comparison axis)
            "prompt_system": prompt_system,  # RAG prompt-system comparison prompt-system id (board comparison axis)
            "n_tasks": len(tasks),
            "max_steps": max_steps,
            "completion_rate": result.objective_score,
            "mean_trajectory_steps": round(mean_steps, 4),
            "mean_tool_calls": round(mean_tool_calls, 4),
            "completion_rate_ci": list(completion_ci) if completion_ci else None,
            "judge_trusted": outcome.trusted,
            "trajectory_quality": quality,  # gated diagnostic, NOT the headline
            "trajectory_quality_ci": list(quality_ci) if quality_ci else None,
            "judge_diagnostics": outcome.diagnostics,  # judge diagnostics zero-valued-judge observability
            **verification_cfg,
        }
        judge_status: JudgeStatus | None = None
        if judge_model is not None:
            judge_status = {
                "calibration_rho": judge_rho,
                "threshold": judge_threshold,
                "trusted": outcome.trusted,
                "model": judge_model,
                "metrics": ["trajectory_quality"],
                "diagnostics": outcome.diagnostics,
            }
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=run_name,
            config=config,
            metrics=metrics,
            case_rows=rows,
            judge=judge_status,
            mirror=mirror,
        )
        _LOG.info(
            "[agentic] %s completion=%.3f mean-steps=%.2f mean-tool-calls=%.2f quality=%s -> %s",
            model,
            result.objective_score,
            mean_steps,
            mean_tool_calls,
            f"{quality:.3f}" if quality is not None else "n/a",
            paths["manifest"],
        )
    return AgenticRun(
        result=result,
        episodes=episodes,
        rows=rows,
        board=board,
        table=table,
        completion_ci=completion_ci,
        mean_steps=mean_steps,
        mean_tool_calls=mean_tool_calls,
        paths=paths,
        trajectory_quality=quality,
        trajectory_quality_ci=quality_ci,
        judge_trusted=outcome.trusted,
        judge_reason=outcome.reason,
        judge_diagnostics=outcome.diagnostics,
    )


def load_tasks_file(path: Path | str) -> list[AgenticTask]:
    """Load an agentic task set (a JSON array of task records)."""
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array of agentic tasks")
    return [AgenticTask.from_record(r) for r in raw]
