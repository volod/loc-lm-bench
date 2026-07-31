"""Shared agentic types + constants: the task/episode/run dataclasses, the `Harness` protocol,
and the named harness / status / assertion-kind constants used across the agentic package.

No behavior lives here -- this is the vocabulary the episode runner, trajectory judge, persistence,
and orchestration submodules all import from.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from llb.bench.agentic.context import ContextPolicy, ContextTelemetry
from llb.bench.agentic.context_budget import ContextBudget
from llb.bench.common import JudgeScorer, LLMComplete, Mirror
from llb.bench.tool_world import ToolWorld
from llb.eval.common import CONTEXT_OVERFLOW
from llb.core.contracts.benchmarks import AgenticCaseRow, ToolDef
from llb.core.contracts.results import BoardRow
from llb.core.contracts.judging import JudgeDiagnostics
from llb.core.contracts.runs import RunPaths
from llb.scoring.leaderboard import ModelResult
from llb.scoring.judge.model import JudgeOutcome

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

STATUS_COMPLETED = "completed"
STATUS_INCOMPLETE = "incomplete"
# A step's prompt did not fit the resolved window, so it was NEVER SENT and the episode ended
# there. Reused verbatim from the shared eval taxonomy (the context-ablation lane raises the same
# status for the same reason) so one vocabulary covers "the context did not fit" everywhere.
STATUS_CONTEXT_OVERFLOW = CONTEXT_OVERFLOW

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
    # Context accounting (agent context-management policies). Defaulted so every harness that
    # builds an `Episode` without the policy-aware loop keeps working and simply reports nothing.
    telemetry: ContextTelemetry = field(default_factory=ContextTelemetry)
    # Whether this harness APPLIED the requested context policy to the prompts it sent. False
    # means the framework owned the transcript (CrewAI) and the comparison must not treat the
    # row as our `full` / `observation_cap` / ... assembly -- see harness comparison docs.
    context_policy_supported: bool = True
    # Loop-policy diagnostics. Defaults keep episodes built by framework adapters compatible.
    n_model_calls: int = 0
    n_malformed_calls: int = 0
    n_repair_attempts: int = 0
    n_repeated_noops: int = 0
    elapsed_s: float = 0.0


class Harness(Protocol):
    """A pluggable agentic harness (agentic harness comparison): drive ONE task to a canonical `Episode`.

    Every harness takes the same `(task, complete, catalog, max_steps, policy, budget)` and returns
    the same `Episode` (final answer + tool-call transcript + final env-state + prompt telemetry),
    so `check_success`, the scorer, and the gated judge are UNCHANGED across harnesses -- the
    framework is the only variable. `loop` and `langgraph` apply `policy`/`budget` to every step
    prompt; a harness that cannot (CrewAI owns its own transcript) still accepts the kwargs, runs
    framework-native, and sets `Episode.context_policy_supported=False` so the comparison never
    silently labels that row as our `full` policy.
    """

    def __call__(
        self,
        task: AgenticTask,
        complete: LLMComplete,
        catalog: dict[str, ToolDef],
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        policy: ContextPolicy | None = None,
        budget: ContextBudget | None = None,
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
    judge_diagnostics: JudgeDiagnostics | None = (
        None  # judge diagnostics zero-valued-judge observability
    )


@dataclass(slots=True)
class _ScoredAgenticEpisodes:
    rows: list[AgenticCaseRow]
    case_success: list[float]
    reliability: float
    completion_ci: tuple[float, float] | None
    mean_steps: float
    mean_tool_calls: float


@dataclass(slots=True)
class _TrajectoryQualityResult:
    outcome: JudgeOutcome
    value: float | None
    ci: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class _JudgeConfig:
    model: str | None
    rho: float | None
    threshold: float
    scorer: JudgeScorer | None
    base_url: str | None


@dataclass(frozen=True, slots=True)
class _AgenticPersistInput:
    data_dir: Path | str | None
    run_name: str
    model: str
    backend: str
    harness_name: str
    prompt_system: str | None
    n_tasks: int
    max_steps: int
    result: ModelResult
    scored: _ScoredAgenticEpisodes
    quality: _TrajectoryQualityResult
    judge_config: _JudgeConfig
    verification_cfg: dict[str, object]
    tokens_per_s: float
    mirror: Mirror | None
    budget_provenance: dict[str, object] | None = None
    context_policy: str = "full"
    context_policy_supported: bool = True
    mean_max_prompt_tokens: float = 0.0
