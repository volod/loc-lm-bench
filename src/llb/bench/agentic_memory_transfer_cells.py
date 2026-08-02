"""Shared execution and row contract for compact-memory transfer cells."""

from pathlib import Path

from llb.bench.agentic.context_budget import fixed_budget
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_compact_vs_cap import run_compact_vs_cap
from llb.bench.agentic_compact_vs_cap_report import CompactVsCapRun
from llb.bench.agentic_context_report import (
    METRIC_COMPACTION_PROMPT_TOKENS,
    METRIC_TOTAL_MODEL_INPUT_TOKENS,
)
from llb.bench.agentic_memory_transcript import build_memory_dependent_tasks
from llb.bench.common import LLMComplete


def run_transfer_cell(
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    data_dir: Path,
    n_tasks: int,
    depth: int,
    pad_chars: int,
    compact_share: float,
    max_prompt_chars: int,
    max_steps_margin: int,
    observation_cap_chars: int,
    observation_head_share: float,
    minimum_compaction_rate: float,
    cell_id: str | None = None,
    require_cap_fits: bool = False,
) -> dict[str, object]:
    """Run one paired cell and retain its exact geometry and source manifests."""
    tasks = [
        AgenticTask.from_record(row)
        for row in build_memory_dependent_tasks(
            n_tasks=n_tasks,
            depth=depth,
            pad_chars=pad_chars,
        )
    ]
    run = run_compact_vs_cap(
        tasks,
        model=model,
        backend=backend,
        complete=complete,
        max_steps=depth + max_steps_margin,
        budget=fixed_budget(max_prompt_chars),
        observation_cap_chars=observation_cap_chars,
        observation_head_share=observation_head_share,
        compact_share=compact_share,
        min_compaction_rate=minimum_compaction_rate,
        data_dir=data_dir,
    )
    return transfer_cell_row(
        run,
        depth=depth,
        compact_share=compact_share,
        max_prompt_chars=max_prompt_chars,
        cell_id=cell_id,
        require_cap_fits=require_cap_fits,
    )


def transfer_cell_row(
    run: CompactVsCapRun,
    *,
    depth: int,
    compact_share: float,
    max_prompt_chars: int,
    cell_id: str | None = None,
    require_cap_fits: bool = False,
) -> dict[str, object]:
    """Project a focused run onto the self-contained cross-study cell schema."""
    return {
        "cell_id": cell_id or f"depth={depth},compact_share={compact_share:g}",
        "depth": depth,
        "compact_share": compact_share,
        "max_prompt_chars": max_prompt_chars,
        "require_cap_fits": require_cap_fits,
        "task_set_digest": run.task_set_digest,
        "n_tasks": len(run.cap.episodes),
        "cap_completion": run.cap.result.objective_score,
        "compact_completion": run.compact.result.objective_score,
        "cap_mean_total_model_input_tokens": run.cap.metric_mean(METRIC_TOTAL_MODEL_INPUT_TOKENS),
        "compact_mean_total_model_input_tokens": run.compact.metric_mean(
            METRIC_TOTAL_MODEL_INPUT_TOKENS
        ),
        # The compact cost splits into the controller prompts the fold step decides and the
        # summarizer call it pays for; only the second one reads the trigger continuously.
        "compact_mean_compaction_prompt_tokens": run.compact.metric_mean(
            METRIC_COMPACTION_PROMPT_TOKENS
        ),
        "compact_mean_controller_prompt_tokens": (
            run.compact.metric_mean(METRIC_TOTAL_MODEL_INPUT_TOKENS)
            - run.compact.metric_mean(METRIC_COMPACTION_PROMPT_TOKENS)
        ),
        "cap_context_overflows": run.cap.n_context_overflow,
        "compact_context_overflows": run.compact.n_context_overflow,
        "compaction_activation_rate": run.compaction_activation_rate,
        "n_compactions": run.n_compactions,
        "paired": run.paired,
        "verdict": run.verdict,
        "verdict_reason": run.reason,
        "manifests": {
            report.policy: str(report.paths["manifest"]) if report.paths is not None else None
            for report in (run.cap, run.compact)
        },
    }
