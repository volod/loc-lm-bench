"""M5.3 agentic workflows runner -- in-sandbox tool execution under TIER_AGENTIC.

The agentic loop is the M5.0 multi-hop controller pattern extended with TOOL CALLS + an in-sandbox
execution step: each step the model emits one tool call (reusing the M5.2 `parse_tool_call`), the
deterministic `ToolWorld` EXECUTES it, the observation is fed back, and the loop runs until the
model calls `finish` (or answers in prose) or the step budget is exhausted. Task success is an
OBJECTIVE assertion over the final env-state and/or the final answer; the gated judge (opt-in) is
reserved for trajectory quality a deterministic check cannot cover. Completion-rate is the headline
under `TIER_AGENTIC`; trajectory length + tool-call count are recorded as efficiency.

LangGraph is the fixed single agent harness for the design; this loop is the pure, langgraph-free
form of that controller -> execute -> controller cycle, so it is unit-tested from a FAKE
tool-calling endpoint with no GPU. The candidate is reached through an injectable `complete`.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llb.bench.common import (
    LLMComplete,
    Mirror,
    category_result,
    mean,
    persist_category_run,
    render_board,
)
from llb.bench.tool_world import FINISH, ToolWorld, tool_catalog
from llb.contracts import AgenticCaseRow, BoardRow, RunMetrics, RunPaths, ToolDef
from llb.scoring.aggregate import TIER_AGENTIC, ModelResult, bootstrap_mean_ci
from llb.scoring.tooling import parse_tool_call

_LOG = logging.getLogger(__name__)

METHOD = "agentic"
DEFAULT_MAX_STEPS = 6

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
    task: AgenticTask, complete: LLMComplete, *, max_steps: int = DEFAULT_MAX_STEPS
) -> Episode:
    """Drive one task to completion (or the step budget) in the deterministic sandbox."""
    world = ToolWorld.from_setup(task.setup)
    catalog = tool_catalog()
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


def _row(task: AgenticTask, episode: Episode) -> AgenticCaseRow:
    return {
        "item_id": task.id,
        "status": episode.status,
        "success": 1.0 if episode.success else 0.0,
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
    data_dir: Path | str | None = None,
    run_name: str = "m5-agentic",
    persist: bool = True,
    mirror: Mirror | None = None,
) -> AgenticRun:
    """Score one model's task-completion rate over the deterministic tool-world under TIER_AGENTIC."""
    if not tasks:
        raise SystemExit("no agentic tasks provided")
    episodes = [run_episode(task, complete, max_steps=max_steps) for task in tasks]
    rows = [_row(task, ep) for task, ep in zip(tasks, episodes)]
    case_success = [1.0 if ep.success else 0.0 for ep in episodes]

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
            "n_tasks": len(tasks),
            "max_steps": max_steps,
            "completion_rate": result.objective_score,
            "mean_trajectory_steps": round(mean_steps, 4),
            "mean_tool_calls": round(mean_tool_calls, 4),
            "completion_rate_ci": list(completion_ci) if completion_ci else None,
        }
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=run_name,
            config=config,
            metrics=metrics,
            case_rows=rows,
            mirror=mirror,
        )
        _LOG.info(
            "[agentic] %s completion=%.3f mean-steps=%.2f mean-tool-calls=%.2f -> %s",
            model,
            result.objective_score,
            mean_steps,
            mean_tool_calls,
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
    )


def load_tasks_file(path: Path | str) -> list[AgenticTask]:
    """Load an agentic task set (a JSON array of task records)."""
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array of agentic tasks")
    return [AgenticTask.from_record(r) for r in raw]
