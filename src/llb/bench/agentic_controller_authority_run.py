"""Execute one seeded observation-versus-controller authority comparison."""

import logging
from pathlib import Path
from typing import cast

from llb.bench.agentic.context import ContextPolicy
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
from llb.bench.agentic_controller_authority import (
    PLACEMENTS,
    ChannelCell,
    ChannelSeedRun,
)
from llb.bench.agentic_controller_authority_report import persist_channel_cell
from llb.bench.common import LLMChat, LLMComplete, Mirror
from llb.bench.common_backend import ThroughputMeter
from llb.core.contracts.benchmarks import ToolDef
from llb.core.contracts.common import ChatMessage

_LOG = logging.getLogger(__name__)


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
    cells: dict[str, ChannelCell] = {}
    placements = cast(list[str], design.get("placements", list(PLACEMENTS)))
    serializer_transforms = cast(
        dict[str, dict[str, list[dict[str, str]]]] | None,
        design.get("serializer_transforms"),
    )
    for placement in placements:
        meter_tokens = meter.completion_tokens if meter is not None else 0
        meter_seconds = meter.generation_s if meter is not None else 0.0
        snapshots: dict[str, list[ChatMessage]] = {}
        task_number = 0

        def harness(
            task: AgenticTask,
            _complete: LLMComplete,
            catalog: dict[str, ToolDef],
            *,
            max_steps: int = max_steps,
            policy: ContextPolicy | None = None,
            budget: ContextBudget | None = None,
        ) -> Episode:
            nonlocal task_number
            task_number += 1
            _LOG.info(
                "[controller-channel] seed=%d placement=%s task=%d/%d id=%s",
                seed,
                placement,
                task_number,
                len(tasks),
                task.id,
            )

            def record(messages: list[ChatMessage]) -> None:
                if len(messages) > 1 and task.id not in snapshots:
                    snapshots[task.id] = [
                        {"role": message["role"], "content": message["content"]}
                        for message in messages
                    ]

            episode = run_episode(
                task,
                _complete,
                catalog=catalog,
                max_steps=max_steps,
                policy=policy,
                budget=budget,
                loop_policy=LoopPolicy(
                    malformed_call=MALFORMED_ANSWER,
                    repeated_call=REPEATED_NOOP,
                    repeat_feedback=REPEAT_FEEDBACK_GEMMA_AUTHORITY,
                ),
                chat=chat,
                feedback_channel=cast(ControllerChannel, placement),
                feedback_backend=backend,
                feedback_serialization=serializer_transforms,
                snapshot=record,
            )
            _LOG.info(
                "[controller-channel] seed=%d placement=%s id=%s success=%s "
                "steps=%d repeats=%d redirected=%s wall=%.1fs",
                seed,
                placement,
                task.id,
                episode.success,
                episode.n_steps,
                episode.n_repeated_noops,
                episode.repeat_feedback_redirected,
                episode.elapsed_s,
            )
            return episode

        def unused_complete(_prompt: str) -> str:
            raise AssertionError("typed controller chat must handle every model call")

        run = run_agentic(
            tasks,
            model=model,
            backend=backend,
            complete=unused_complete,
            max_steps=max_steps,
            harness_name=HARNESS_LOOP,
            harness=cast(Harness, harness),
            policy=ContextPolicy(),
            budget=budget,
            persist=False,
            meter=meter,
        )
        generated_tokens = (meter.completion_tokens - meter_tokens) if meter is not None else 0
        generation_s = (meter.generation_s - meter_seconds) if meter is not None else 0.0
        cell_tokens_per_s = round(generated_tokens / generation_s, 2) if generation_s > 0 else 0.0
        cell = ChannelCell(
            placement=placement,
            rows=run.rows,
            snapshots=snapshots,
            tokens_per_s=cell_tokens_per_s,
        )
        if data_dir is not None:
            paths = persist_channel_cell(
                design,
                cell,
                seed=seed,
                model=model,
                backend=backend,
                data_dir=data_dir,
                mirror=mirror,
            )
            cell = ChannelCell(
                placement=placement,
                rows=run.rows,
                snapshots=snapshots,
                manifest=paths["manifest"],
                tokens_per_s=cell_tokens_per_s,
            )
        cells[placement] = cell
    return ChannelSeedRun(seed=seed, model=model, backend=backend, cells=cells)


__all__ = ["run_channel_authority_seed", "CHANNEL_OBSERVATION", "CHANNEL_CONTROLLER"]
