"""Context-policy benchmark -- rank context-management policies for ONE fixed model.

A chain-of-questions item is a 2-4 step ordered walk where each step's answer depends on the
prior steps. This benchmark holds the model, the verified chain set, and the scoring fixed and
varies ONLY the context-management POLICY -- the row label -- exactly as the agentic
harness comparison holds everything fixed and varies the harness. Four policies:

  - ``fresh``   -- fresh retrieval per step, NO prior-step carryover (the naive baseline);
  - ``history`` -- fresh retrieval PLUS the accumulated full (question, answer) history;
  - ``summary`` -- fresh retrieval PLUS a running model-written summary of the prior steps;
  - ``roles``   -- staged role/system-prompt sequence (librarian -> analyst -> answerer) built
    from prompt-system role templates, PLUS the accumulated history.

Retrieval is reached through an injectable ``Retriever`` (``retrieve(query, k)``) and the model
through the same injectable ``complete`` (prompt -> raw text) every category uses, so the exact
context assembled per policy per step is provable over a FAKE endpoint with no GPU. Each step's
answer is scored objectively against its reference answer (``scoring.correctness`` token-F1);
per-step and final-answer correctness carry bootstrap CIs and the policies are ranked under
``TIER_CHAIN_CONTEXT``.
"""

import logging
from pathlib import Path
from typing import Any

from llb.bench.common import (
    Mirror,
    persist_category_run,
    render_board,
    verified_data_config,
)
from llb.goldset.chains import ChainItem
from llb.bench.chain_context_report import (
    METHOD,
    ChainContextRun,
    build_recommendation,
    chain_set_digest,
    policy_config,
    policy_metrics,
)
from llb.bench.chain_context_policy import (
    DEFAULT_K,
    POLICY_FRESH,
    POLICY_HISTORY,
    POLICY_ROLES,
    POLICY_SUMMARY,
    PolicyReport,
    Retriever,
    run_policy,
)

_LOG = logging.getLogger(__name__)

CONTEXT_POLICIES: tuple[str, ...] = (POLICY_FRESH, POLICY_HISTORY, POLICY_SUMMARY, POLICY_ROLES)

# Prompt-system template ids for the role sequence and the default instruction. These are the
# reviewable prompt-system content recorded in provenance as `prompt_system_ids`.


# --- context assembly (pure, unit-tested) -------------------------------------------------


# --- per-policy run ------------------------------------------------------------------------


# --- run + persistence ---------------------------------------------------------------------


def run_chain_context(
    chains: list[ChainItem],
    *,
    model: str,
    backend: str,
    retriever: Retriever,
    complete: Any,
    policies: list[str] | None = None,
    k: int = DEFAULT_K,
    data_dir: Path | str | None = None,
    run_name: str = "chain-context",
    persist: bool = True,
    mirror: Mirror | None = None,
    data_verified: bool = False,
    verification_ref: str | None = None,
) -> ChainContextRun:
    """Rank context-management policies for one fixed model over a verified chain set.

    Each policy is walked step by step, scored against the step references, and persisted as its
    OWN run bundle under ``$DATA_DIR/chain-context/`` tagged with the policy (mirroring the
    per-harness agentic bundles); the returned board ranks all policies together under
    ``TIER_CHAIN_CONTEXT``. ``data_verified`` follows the category-suite verification-gate rules.
    """
    if not chains:
        raise SystemExit("no chains provided")
    policies = policies or list(CONTEXT_POLICIES)
    unknown = [p for p in policies if p not in CONTEXT_POLICIES]
    if unknown:
        raise SystemExit(f"unknown context policies: {unknown}; choose from {CONTEXT_POLICIES}")
    verification_cfg = verified_data_config(
        data_verified=data_verified, verification_ref=verification_ref
    )
    digest = chain_set_digest(chains)

    reports = [
        run_policy(
            chains,
            policy,
            model=model,
            backend=backend,
            retriever=retriever,
            complete=complete,
            k=k,
        )
        for policy in policies
    ]
    board, table = render_board([r.result for r in reports])
    recommendation = build_recommendation(model, reports, board)

    if persist and data_dir is not None:
        for report in reports:
            config = {
                **policy_config(report, model, backend, digest, policies),
                **verification_cfg,
            }
            report.paths = persist_category_run(
                method=METHOD,
                data_dir=data_dir,
                run_name=f"{run_name}-{report.policy}",
                config=config,
                metrics=policy_metrics(report),
                case_rows=report.step_rows,
                mirror=mirror,
            )
        _log_run(model, reports)
    return ChainContextRun(
        model=model,
        backend=backend,
        reports=reports,
        board=board,
        table=table,
        recommendation=recommendation,
        chain_digest=digest,
    )


def _log_run(model: str, reports: list[PolicyReport]) -> None:
    for report in reports:
        manifest = report.paths["manifest"] if report.paths else "(not persisted)"
        _LOG.info(
            "[chain-context] %s policy=%s final=%.3f per-step=%.3f reliability=%.3f -> %s",
            model,
            report.policy,
            sum(report.final_objectives) / len(report.final_objectives),
            sum(report.step_objectives) / len(report.step_objectives),
            report.reliability,
            manifest,
        )


def load_chains_file(path: Path | str) -> list[ChainItem]:
    """Load a verified chain set (JSONL) for the benchmark."""
    from llb.goldset.chains import load_chains

    return load_chains(path)
