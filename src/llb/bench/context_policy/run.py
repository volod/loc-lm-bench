"""Context-policy comparison for the AGENT LOOP -- rank `full` / `observation_cap` /
`keep_last_n` / `compact` for ONE fixed model over one task set.

The agent loop rebuilds its prompt from the whole transcript on every step, so context management
was never a decision anyone made -- it was the absence of one. This lane makes it a POLICY ROW and
measures it, exactly as `bench-chain-context` does for question chains and as the agentic harness
comparison does for the framework: the model, the task set, the tool world, the success checks, and
the step budget are FIXED and only the policy varies.

Every policy runs a fresh episode over the identical task set through the pure `loop` harness
(`run_episode`) -- context management is a property of the loop, not of a framework -- and is
persisted as its OWN bundle under `$DATA_DIR/agentic-context/` tagged with the policy, mirroring
the per-harness and per-chain-policy bundles. The per-step prompt guard rides underneath all four:
`llb.backends.context_budget` resolves the model's usable window once per run, and a step
whose prompt does not fit ends the episode as `context_overflow` instead of being sent.
"""

import hashlib
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from llb.bench.agentic.context_policy import CONTEXT_POLICIES, ContextPolicy
from llb.backends.context_budget import ContextBudget, unbounded_budget
from llb.bench.agentic.batch import _score_episodes
from llb.bench.agentic.episode import run_episode
from llb.bench.agentic.model import (
    DEFAULT_MAX_STEPS,
    STATUS_CONTEXT_OVERFLOW,
    AgenticTask,
    Episode,
)
from llb.bench.context_policy.report import (
    AgenticContextRun,
    PolicyReport,
    pair_against_baseline,
)
from llb.bench.context_policy.report_kind import aggregate_safe_verdict, format_kind_table
from llb.bench.context_policy.report_persist import known_policies, persist_policy_bundles
from llb.bench.context_policy.report_recommendation import (
    build_recommendation,
    format_policy_table,
)
from llb.bench.common import (
    LLMComplete,
    Mirror,
    category_result,
    render_board,
    verified_data_config,
)
from llb.bench.common_backend import ThroughputMeter
from llb.bench.tool_world import tool_catalog
from llb.scoring.aggregate import TIER_AGENTIC

_LOG = logging.getLogger(__name__)


def task_set_digest(tasks: list[AgenticTask]) -> str:
    """Order-sensitive digest of the task-set content, so two policies provably ran the same set."""
    payload = json.dumps([asdict(task) for task in tasks], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_policy(
    tasks: list[AgenticTask],
    policy: ContextPolicy,
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    max_steps: int = DEFAULT_MAX_STEPS,
    budget: ContextBudget | None = None,
    preserve_memory_markers: bool = True,
) -> PolicyReport:
    """Run one fresh episode per task under one context policy and score the batch."""
    budget = budget if budget is not None else unbounded_budget()
    catalog = tool_catalog()
    episodes: list[Episode] = []
    for index, task in enumerate(tasks, start=1):
        _LOG.info(
            "[agentic-context] policy=%s task=%d/%d id=%s",
            policy.name,
            index,
            len(tasks),
            task.id,
        )
        episodes.append(
            run_episode(
                task,
                complete,
                catalog=catalog,
                max_steps=max_steps,
                policy=policy,
                budget=budget,
                preserve_memory_markers=preserve_memory_markers,
            )
        )
    scored = _score_episodes(tasks, episodes)
    result = category_result(
        model=policy.name,  # the policy IS the ranked row label (the model is fixed)
        backend=backend,
        tier=TIER_AGENTIC,
        case_objectives=scored.case_success,
        reliability=scored.reliability,
    )
    return PolicyReport(
        policy=policy.name,
        result=result,
        rows=scored.rows,
        episodes=episodes,
        case_success=scored.case_success,
        reliability=scored.reliability,
        completion_ci=scored.completion_ci,
        mean_steps=scored.mean_steps,
        mean_tool_calls=scored.mean_tool_calls,
        n_context_overflow=sum(1 for e in episodes if e.status == STATUS_CONTEXT_OVERFLOW),
    )


def run_agentic_context(
    tasks: list[AgenticTask],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    policies: list[str] | None = None,
    policy_overrides: dict[str, Any] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    budget: ContextBudget | None = None,
    data_dir: Path | str | None = None,
    run_name: str = "agentic-context",
    persist: bool = True,
    mirror: Mirror | None = None,
    data_verified: bool = False,
    verification_ref: str | None = None,
    meter: ThroughputMeter | None = None,
) -> AgenticContextRun:
    """Rank context-management policies for one fixed model over one agentic task set.

    Each policy is walked over the identical task set, scored on the same objective completion
    checks, paired against the `full` baseline, and persisted as its own tagged bundle; the
    returned board ranks all policies together under `TIER_AGENTIC`.
    """
    if not tasks:
        raise SystemExit("no agentic tasks provided")
    policies = known_policies(policies or list(CONTEXT_POLICIES))
    budget = budget if budget is not None else unbounded_budget()
    overrides = policy_overrides or {}
    verification_cfg = verified_data_config(
        data_verified=data_verified, verification_ref=verification_ref
    )
    digest = task_set_digest(tasks)

    reports = [
        run_policy(
            tasks,
            ContextPolicy(name=name, **overrides),
            model=model,
            backend=backend,
            complete=complete,
            max_steps=max_steps,
            budget=budget,
        )
        for name in policies
    ]
    # The meter accumulates ACROSS policies (one endpoint drives them all), so it is only
    # meaningful once every policy has run -- reading it per policy would credit the first row
    # with 0.0 and each later row with a running average of the ones before it.
    _stamp_throughput(reports, meter)
    pair_against_baseline(reports)
    board, board_table = render_board([r.result for r in reports])
    kind_table = format_kind_table(reports)
    safe_verdict = aggregate_safe_verdict(reports)
    table = f"{board_table}\n\n{format_policy_table(reports)}"
    if kind_table:
        table = f"{table}\n\n{kind_table}"
    if safe_verdict:
        table = f"{table}\n\n{safe_verdict}"
    recommendation = build_recommendation(model, reports)

    if persist and data_dir is not None:
        persist_policy_bundles(
            reports,
            data_dir=data_dir,
            run_name=run_name,
            model=model,
            backend=backend,
            digest=digest,
            policies=policies,
            max_steps=max_steps,
            max_prompt_chars=budget.max_prompt_chars,
            overrides=overrides,
            verification_cfg=verification_cfg,
            mirror=mirror,
            budget_provenance=budget.provenance(),
        )
    return AgenticContextRun(
        model=model,
        backend=backend,
        reports=reports,
        board=board,
        table=table,
        recommendation=recommendation,
        task_set_digest=digest,
        max_prompt_chars=budget.max_prompt_chars,
        kind_table=kind_table,
        aggregate_safe_verdict=safe_verdict,
    )


def _stamp_throughput(reports: list[PolicyReport], meter: ThroughputMeter | None) -> None:
    """Put the run's measured generation tok/s on every policy row, after all policies have run."""
    tokens_per_s = meter.tokens_per_s if meter is not None else 0.0
    for report in reports:
        report.result = replace(report.result, tokens_per_s=tokens_per_s)
