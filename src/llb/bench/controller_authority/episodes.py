"""Execute one seeded observation-versus-controller authority comparison."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import cast

from llb.bench.agentic.context_policy import ContextPolicy
from llb.bench.agentic.context_budget import ContextBudget
from llb.bench.agentic.controller_channel import (
    CHANNEL_CONTROLLER,
    CHANNEL_OBSERVATION,
    ControllerChannel,
)
from llb.bench.agentic.episode import run_episode
from llb.bench.agentic.model import HARNESS_LOOP, AgenticTask, Episode, Harness
from llb.bench.agentic.run import run_agentic
from llb.bench.agentic.loop_policy import (
    MALFORMED_ANSWER,
    REPEATED_NOOP,
    REPEAT_FEEDBACK_GEMMA_AUTHORITY,
    LoopPolicy,
)
from llb.bench.controller_authority.run import PLACEMENTS
from llb.bench.controller_authority.model import ChannelCell, ChannelSeedRun
from llb.bench.controller_authority.report import persist_channel_cell
from llb.bench.common import LLMChat, LLMComplete, Mirror
from llb.bench.common_backend import ThroughputMeter
from llb.core.contracts.benchmarks import ToolDef
from llb.core.contracts.common import ChatMessage

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class _PlacementContext:
    """Everything one placement's episodes are driven with, and the proof they leave behind.

    A record rather than a closure over the seed function's locals: the harness callback needs nine
    settings and must accumulate snapshots, and threading that through captured variables is what
    made this step unreadable.
    """

    seed: int
    placement: str
    backend: str
    chat: LLMChat
    n_tasks: int
    serializer_transforms: dict[str, dict[str, list[dict[str, str]]]] | None
    snapshots: dict[str, list[ChatMessage]] = field(default_factory=dict)
    completed: int = 0


def _record_snapshot(context: _PlacementContext, task_id: str, messages: list[ChatMessage]) -> None:
    """Keep the FIRST multi-message transcript per task: that is the placement's own evidence."""
    if len(messages) > 1 and task_id not in context.snapshots:
        context.snapshots[task_id] = [
            {"role": message["role"], "content": message["content"]} for message in messages
        ]


def _run_placement_episode(
    task: AgenticTask,
    complete: LLMComplete,
    catalog: dict[str, ToolDef],
    *,
    context: _PlacementContext,
    max_steps: int,
    policy: ContextPolicy | None,
    budget: ContextBudget | None,
) -> Episode:
    """One task under one placement, logged on both sides and snapshotted as it is sent."""
    context.completed += 1
    _LOG.info(
        "[controller-channel] seed=%d placement=%s task=%d/%d id=%s",
        context.seed,
        context.placement,
        context.completed,
        context.n_tasks,
        task.id,
    )
    episode = run_episode(
        task,
        complete,
        catalog=catalog,
        max_steps=max_steps,
        policy=policy,
        budget=budget,
        loop_policy=LoopPolicy(
            malformed_call=MALFORMED_ANSWER,
            repeated_call=REPEATED_NOOP,
            repeat_feedback=REPEAT_FEEDBACK_GEMMA_AUTHORITY,
        ),
        chat=context.chat,
        feedback_channel=cast(ControllerChannel, context.placement),
        feedback_backend=context.backend,
        feedback_serialization=context.serializer_transforms,
        snapshot=lambda messages: _record_snapshot(context, task.id, messages),
    )
    _LOG.info(
        "[controller-channel] seed=%d placement=%s id=%s success=%s "
        "steps=%d repeats=%d redirected=%s wall=%.1fs",
        context.seed,
        context.placement,
        task.id,
        episode.success,
        episode.n_steps,
        episode.n_repeated_noops,
        episode.repeat_feedback_redirected,
        episode.elapsed_s,
    )
    return episode


def _unused_complete(_prompt: str) -> str:
    """The completion seam is unreachable here: a typed controller chat handles every model call."""
    raise AssertionError("typed controller chat must handle every model call")


def _cell_tokens_per_s(meter: ThroughputMeter | None, tokens: int, seconds: float) -> float:
    """The rate this placement generated at, measured as the delta across its own episodes."""
    if meter is None:
        return 0.0
    generated = meter.completion_tokens - tokens
    elapsed = meter.generation_s - seconds
    return round(generated / elapsed, 2) if elapsed > 0 else 0.0


def _run_placement(
    tasks: list[AgenticTask],
    context: _PlacementContext,
    *,
    model: str,
    budget: ContextBudget,
    max_steps: int,
    design: dict[str, object],
    data_dir: Path | str | None,
    meter: ThroughputMeter | None,
    mirror: Mirror | None,
) -> ChannelCell:
    """Run every task under one placement and persist the cell it produced."""
    tokens_before = meter.completion_tokens if meter is not None else 0
    seconds_before = meter.generation_s if meter is not None else 0.0

    def harness(
        task: AgenticTask,
        complete: LLMComplete,
        catalog: dict[str, ToolDef],
        *,
        max_steps: int = max_steps,
        policy: ContextPolicy | None = None,
        budget: ContextBudget | None = None,
    ) -> Episode:
        """Adapter to the harness protocol; the step itself is `_run_placement_episode`."""
        return _run_placement_episode(
            task,
            complete,
            catalog,
            context=context,
            max_steps=max_steps,
            policy=policy,
            budget=budget,
        )

    run = run_agentic(
        tasks,
        model=model,
        backend=context.backend,
        complete=_unused_complete,
        max_steps=max_steps,
        harness_name=HARNESS_LOOP,
        harness=cast(Harness, harness),
        policy=ContextPolicy(),
        budget=budget,
        persist=False,
        meter=meter,
    )
    tokens_per_s = _cell_tokens_per_s(meter, tokens_before, seconds_before)
    cell = ChannelCell(
        placement=context.placement,
        rows=run.rows,
        snapshots=context.snapshots,
        tokens_per_s=tokens_per_s,
    )
    if data_dir is None:
        return cell
    paths = persist_channel_cell(
        design,
        cell,
        seed=context.seed,
        model=model,
        backend=context.backend,
        data_dir=data_dir,
        mirror=mirror,
    )
    return ChannelCell(
        placement=context.placement,
        rows=run.rows,
        snapshots=context.snapshots,
        manifest=paths["manifest"],
        tokens_per_s=tokens_per_s,
    )


def run_channel_authority_seed(
    tasks: list[AgenticTask],
    *,
    seed: int,
    model: str,
    backend: str,
    chat: LLMChat,
    budget: ContextBudget,
    max_steps: int,
    design: dict[str, object],
    data_dir: Path | str | None = None,
    meter: ThroughputMeter | None = None,
    mirror: Mirror | None = None,
) -> ChannelSeedRun:
    """Run both immutable placements over fresh episodes and persist their source cells."""
    placements = cast(list[str], design.get("placements", list(PLACEMENTS)))
    serializer_transforms = cast(
        dict[str, dict[str, list[dict[str, str]]]] | None,
        design.get("serializer_transforms"),
    )
    cells: dict[str, ChannelCell] = {}
    for placement in placements:
        context = _PlacementContext(
            seed=seed,
            placement=placement,
            backend=backend,
            chat=chat,
            n_tasks=len(tasks),
            serializer_transforms=serializer_transforms,
        )
        cells[placement] = _run_placement(
            tasks,
            context,
            model=model,
            budget=budget,
            max_steps=max_steps,
            design=design,
            data_dir=data_dir,
            meter=meter,
            mirror=mirror,
        )
    return ChannelSeedRun(seed=seed, model=model, backend=backend, cells=cells)


__all__ = ["run_channel_authority_seed", "CHANNEL_OBSERVATION", "CHANNEL_CONTROLLER"]
