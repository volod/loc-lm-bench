"""Active-compaction evidence: compare compact directly against observation_cap.

The broad context-policy lane pairs every policy against ``full``. This focused lane answers the
different operational question left by the short-transcript tie: once compaction ACTUALLY fires,
does its extra summarizer call buy completion or model-input savings beyond live observation trim?
"""

from dataclasses import replace
import math
from pathlib import Path
from typing import Any

from llb.bench.agentic.context_policy import (
    DEFAULT_COMPACT_SHARE,
    DEFAULT_OBSERVATION_CAP_CHARS,
    DEFAULT_SUMMARY_INPUT_CAP,
    OBSERVATION_HEAD_SHARE,
    POLICY_COMPACT,
    POLICY_OBSERVATION_CAP,
    ContextPolicy,
)
from llb.bench.agentic.context_budget import ContextBudget, unbounded_budget
from llb.bench.agentic.model import DEFAULT_MAX_STEPS, AgenticTask
from llb.bench.context_policy.run import run_policy, task_set_digest
from llb.bench.context_policy.report import PolicyReport
from llb.bench.context_policy.report_persist import policy_config, policy_metrics
from llb.bench.context_policy.compact_vs_cap_report import (
    METHOD,
    PAIRED_METRICS,
    CompactVsCapRun,
    decide_verdict,
    format_table,
)
from llb.bench.common import LLMComplete, Mirror, persist_category_run, verified_data_config
from llb.bench.common_backend import ThroughputMeter
from llb.rag.fusion_evidence.evidence_gate import DEFAULT_CONFIDENCE
from llb.rag.fusion_evidence.paired import PairedComparison, paired_comparison
from llb.rag.fusion_evidence.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_index_sets


def run_compact_vs_cap(
    tasks: list[AgenticTask],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    max_steps: int = DEFAULT_MAX_STEPS,
    budget: ContextBudget | None = None,
    observation_cap_chars: int = DEFAULT_OBSERVATION_CAP_CHARS,
    observation_head_share: float = OBSERVATION_HEAD_SHARE,
    compact_share: float = DEFAULT_COMPACT_SHARE,
    summary_input_cap: str = DEFAULT_SUMMARY_INPUT_CAP,
    min_compaction_rate: float = 0.0,
    data_dir: Path | str | None = None,
    persist: bool = True,
    mirror: Mirror | None = None,
    data_verified: bool = False,
    verification_ref: str | None = None,
    meter: ThroughputMeter | None = None,
) -> CompactVsCapRun:
    """Run both policies, pair compact against cap, and require observable compaction."""
    if not tasks:
        raise SystemExit("no agentic tasks provided")
    if not 0.0 <= min_compaction_rate <= 1.0:
        raise ValueError(f"min_compaction_rate must be in [0, 1], got {min_compaction_rate}")
    budget = budget if budget is not None else unbounded_budget()
    common: dict[str, Any] = {
        "observation_cap_chars": observation_cap_chars,
        "observation_head_share": observation_head_share,
        "compact_share": compact_share,
        # Inert on the cap arm (nothing summarizes there), carried anyway so the persisted
        # policy_settings name the summarize-input bound the compact arm actually ran under.
        "summary_input_cap": summary_input_cap,
    }
    cap = run_policy(
        tasks,
        ContextPolicy(name=POLICY_OBSERVATION_CAP, **common),
        model=model,
        backend=backend,
        complete=complete,
        max_steps=max_steps,
        budget=budget,
    )
    compact = run_policy(
        tasks,
        ContextPolicy(name=POLICY_COMPACT, **common),
        model=model,
        backend=backend,
        complete=complete,
        max_steps=max_steps,
        budget=budget,
    )
    tokens_per_s = meter.tokens_per_s if meter is not None else 0.0
    cap.result = replace(cap.result, tokens_per_s=tokens_per_s)
    compact.result = replace(compact.result, tokens_per_s=tokens_per_s)

    index_sets = bootstrap_index_sets(len(tasks), DEFAULT_RESAMPLES, DEFAULT_SEED)
    paired = {
        metric: paired_comparison(
            compact.vector(metric),
            cap.vector(metric),
            index_sets,
            DEFAULT_CONFIDENCE,
        )
        for metric in PAIRED_METRICS
    }
    n_compactions = int(sum(compact.vector("n_compactions")))
    n_compacted_episodes = sum(
        1 for episode in compact.episodes if episode.telemetry.n_compactions > 0
    )
    min_compacted_episodes = max(1, math.ceil(len(tasks) * min_compaction_rate))
    activation_rate = n_compacted_episodes / len(tasks)
    verdict, reason = decide_verdict(
        cap,
        compact,
        paired,
        min_compacted_episodes=min_compacted_episodes,
    )
    table = format_table(cap, compact, paired, verdict=verdict, reason=reason)
    digest = task_set_digest(tasks)
    if persist and data_dir is not None:
        _persist(
            cap,
            compact,
            paired=paired,
            verdict=verdict,
            reason=reason,
            table=table,
            data_dir=data_dir,
            model=model,
            backend=backend,
            digest=digest,
            max_steps=max_steps,
            budget=budget,
            policy_settings=common,
            min_compaction_rate=min_compaction_rate,
            data_verified=data_verified,
            verification_ref=verification_ref,
            mirror=mirror,
        )
    return CompactVsCapRun(
        model=model,
        backend=backend,
        cap=cap,
        compact=compact,
        paired=paired,
        verdict=verdict,
        reason=reason,
        table=table,
        task_set_digest=digest,
        max_prompt_chars=budget.max_prompt_chars,
        n_compactions=n_compactions,
        n_compacted_episodes=n_compacted_episodes,
        min_compacted_episodes=min_compacted_episodes,
        compaction_activation_rate=activation_rate,
    )


def _persist(
    cap: PolicyReport,
    compact: PolicyReport,
    *,
    paired: dict[str, PairedComparison],
    verdict: str,
    reason: str,
    table: str,
    data_dir: Path | str,
    model: str,
    backend: str,
    digest: str,
    max_steps: int,
    budget: ContextBudget,
    policy_settings: dict[str, Any],
    min_compaction_rate: float,
    data_verified: bool,
    verification_ref: str | None,
    mirror: Mirror | None,
) -> None:
    verification = verified_data_config(
        data_verified=data_verified,
        verification_ref=verification_ref,
    )
    for report in (cap, compact):
        config = {
            **policy_config(
                report,
                model=model,
                backend=backend,
                task_digest=digest,
                policies=[POLICY_OBSERVATION_CAP, POLICY_COMPACT],
                max_steps=max_steps,
                max_prompt_chars=budget.max_prompt_chars,
                policy_settings=policy_settings,
                budget_provenance=budget.provenance(),
            ),
            **verification,
            "category": METHOD,
            "paired_compact_vs_observation_cap": paired if report is compact else None,
            "verdict": verdict,
            "verdict_reason": reason,
        }
        report.paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=f"agentic-compact-vs-cap-{report.policy}",
            config=config,
            metrics=policy_metrics(report),
            case_rows=report.rows,
            mirror=mirror,
        )
    persist_category_run(
        method=METHOD,
        data_dir=data_dir,
        run_name="agentic-compact-vs-cap-summary",
        config={
            "model": model,
            "backend": backend,
            "category": METHOD,
            "task_set_digest": digest,
            "max_steps": max_steps,
            "max_prompt_chars": budget.max_prompt_chars,
            "policy_settings": policy_settings,
            "paired_compact_vs_observation_cap": paired,
            "verdict": verdict,
            "verdict_reason": reason,
            "n_compactions": int(sum(compact.vector("n_compactions"))),
            "n_compacted_episodes": sum(
                1 for episode in compact.episodes if episode.telemetry.n_compactions > 0
            ),
            "min_compaction_rate": min_compaction_rate,
            "min_compacted_episodes": max(
                1, math.ceil(len(compact.episodes) * min_compaction_rate)
            ),
            "compaction_activation_rate": (
                sum(1 for episode in compact.episodes if episode.telemetry.n_compactions > 0)
                / len(compact.episodes)
            ),
            "table": table,
            **budget.provenance(),
            **verification,
        },
        metrics={"objective_score": 0.0, "reliability": 1.0, "tokens_per_s": 0.0},
        case_rows=[],
        mirror=mirror,
    )
