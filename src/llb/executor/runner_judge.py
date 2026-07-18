"""Judge scoring for the runner: scorer-policy seam, gated local judge, and worksheet.

`run_eval` calls `_judge_cases` (trusted per-case score) and `_build_judge_metadata`;
`_write_calibration_worksheet` backs the `worksheet=` path. `JudgeScorer` types the injectable
scoring seam.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from llb.core.config import RunConfig
from llb.core.contracts.judging import JudgeInputRecord, JudgeScore, JudgeStatus
from llb.executor.cases import CaseBatch
from llb.scoring.judge.model import judge_is_trusted, run_judge
from llb.scoring.policy import (
    BudgetExceeded,
    HUMAN_LANE_REASON,
    ScorerPolicyRequest,
    resolve_scorer,
    scorer_dir,
)
from llb.scoring.policy.lanes import ScorerLane

JudgeScorer = Callable[[list[JudgeInputRecord], str], list[JudgeScore]]
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeCaseResult:
    """Trusted mean score (if any) plus policy metadata for the manifest."""

    mean_score: float | None
    policy_metadata: dict[str, object]


def _judge_records(batch: CaseBatch) -> list[JudgeInputRecord]:
    """The (question, answer, retrieved-contexts) record per case the judge scores."""
    return [
        {
            "question": item.question,
            "answer": answer,
            "contexts": [str(chunk.get("text", "")) for chunk in retrieved],
        }
        for (item, answer), (retrieved, _spans) in zip(batch.answers, batch.retrieval_pairs)
    ]


def _judge_value(score: JudgeScore) -> float:
    """One scalar judge rating per case: the mean of faithfulness + answer-relevancy."""
    return (score["faithfulness"] + score["answer_relevancy"]) / 2.0


def _policy_request(
    config: RunConfig,
    scorer: JudgeScorer | None,
    staging_dir: Path | None,
) -> ScorerPolicyRequest:
    return ScorerPolicyRequest(
        lane=config.scorer_policy,
        judge_model=config.judge_model,
        judge_base_url=config.judge_base_url,
        egress_consent=config.scorer_egress_consent,
        max_usd=config.frontier_max_usd,
        max_calls=config.frontier_max_calls,
        run_dir=staging_dir,
        local_scorer=scorer,
    )


def _configured_judge_scorer(
    config: RunConfig,
    scorer: JudgeScorer | None,
    staging_dir: Path | None = None,
) -> tuple[JudgeScorer, dict[str, object]]:
    """Bind the scorer-policy lane; return (scorer, policy metadata)."""
    lane: ScorerLane = config.scorer_policy
    if lane == "human":
        return _noop_scorer, {"scorer_policy": "human", "provider": "human"}
    if lane == "local" and config.judge_model is None:
        return _noop_scorer, {"scorer_policy": "local", "provider": "none"}
    if scorer is not None:
        return scorer, {
            "scorer_policy": lane,
            "provider": "injected",
            "model": config.judge_model,
        }
    resolved = resolve_scorer(_policy_request(config, scorer, staging_dir))
    return resolved.scorer, dict(resolved.metadata or {})


def _noop_scorer(records: list[JudgeInputRecord], model: str) -> list[JudgeScore]:
    del model
    return [{"faithfulness": 0.0, "answer_relevancy": 0.0} for _ in records]


def _judge_cases(
    config: RunConfig,
    batch: CaseBatch,
    judge_rho: float | None,
    scorer: JudgeScorer | None,
    staging_dir: Path | None = None,
) -> JudgeCaseResult:
    """Score answers with the scorer-policy seam and the calibration gate."""
    if config.scorer_policy == "human":
        _LOG.info("[run-eval] scorer_policy=human; %s", HUMAN_LANE_REASON)
        return JudgeCaseResult(None, {"scorer_policy": "human", "provider": "human"})
    if config.scorer_policy == "local" and config.judge_model is None:
        return JudgeCaseResult(None, {"scorer_policy": "local", "provider": "none"})
    score_fn, policy_meta = _configured_judge_scorer(config, scorer, staging_dir)
    try:
        outcome = run_judge(
            _judge_records(batch),
            config.judge_model,
            judge_rho,
            config.judge_threshold,
            scorer=score_fn,
        )
    except BudgetExceeded as exc:
        _write_budget_abort(staging_dir, exc)
        raise SystemExit(
            f"[run-eval] frontier scorer budget exceeded ({exc.reason}); "
            f"staging preserved under {staging_dir} -- resume with --resume"
        ) from exc
    if not outcome.trusted or not outcome.scores:
        _LOG.info("[run-eval] judge demoted (%s); objective ranks alone", outcome.reason)
        return JudgeCaseResult(None, policy_meta)
    per_case = [_judge_value(s) for s in outcome.scores]
    for row, value in zip(batch.rows, per_case):
        row["judge_score"] = round(value, 4)
    mean = sum(per_case) / len(per_case) if per_case else None
    return JudgeCaseResult(mean, policy_meta)


def _judge_ratings(
    config: RunConfig,
    batch: CaseBatch,
    scorer: JudgeScorer | None,
    staging_dir: Path | None = None,
) -> list[float] | None:
    """Run the judge UNGATED and return one rating per case (calibration worksheet path)."""
    if config.scorer_policy == "human":
        return None
    if config.judge_model is None:
        return None
    score_fn, _ = _configured_judge_scorer(config, scorer, staging_dir)
    scores = score_fn(_judge_records(batch), config.judge_model)
    return [_judge_value(s) for s in scores]


def _build_judge_metadata(
    config: RunConfig,
    judge_rho: float | None,
    policy_metadata: dict[str, object] | None = None,
) -> JudgeStatus:
    judge_metadata: JudgeStatus = {
        "calibration_rho": judge_rho,
        "threshold": config.judge_threshold,
        "trusted": judge_is_trusted(judge_rho, config.judge_threshold),
        "scorer_policy": config.scorer_policy,
    }
    meta = policy_metadata or {}
    if config.scorer_policy == "human":
        judge_metadata["provider"] = "human"
        return judge_metadata
    if config.judge_model is None:
        return judge_metadata
    if config.scorer_policy == "frontier":
        judge_metadata["provider"] = str(meta.get("provider", "litellm-frontier"))
        judge_metadata["model"] = config.judge_model
        judge_metadata["prompt_language"] = "uk"
        judge_metadata["metrics"] = ["faithfulness", "answer_relevancy"]
        budget = meta.get("budget")
        if isinstance(budget, dict):
            judge_metadata["budget"] = budget
        return judge_metadata
    from llb.scoring.judge.endpoint import judge_experiment_metadata

    experiment_metadata = judge_experiment_metadata(config.judge_model, config.judge_base_url)
    judge_metadata["provider"] = experiment_metadata["provider"]
    judge_metadata["model"] = experiment_metadata["model"]
    judge_metadata["base_url"] = experiment_metadata["base_url"]
    judge_metadata["prompt_language"] = experiment_metadata["prompt_language"]
    judge_metadata["metrics"] = experiment_metadata["metrics"]
    return judge_metadata


def _write_calibration_worksheet(
    config: RunConfig,
    batch: CaseBatch,
    worksheet: Path | str,
    judge_scorer: JudgeScorer | None,
    staging_dir: Path | None = None,
) -> int:
    from llb.judge.calibration_worksheet import write_filled_worksheet

    judge_ratings: list[float] | None = None
    if config.judge_model is not None and config.scorer_policy != "human":
        try:
            judge_ratings = _judge_ratings(config, batch, judge_scorer, staging_dir)
        except (Exception, SystemExit) as exc:
            _LOG.warning(
                "[run-eval] judge unavailable for the worksheet (%s); judge_rating left blank "
                "-- pick the judge (OQ2) and install its backend to calibrate.",
                exc,
            )
    return write_filled_worksheet(batch.answers, Path(worksheet), judge_ratings=judge_ratings)


def _write_budget_abort(staging_dir: Path | None, exc: BudgetExceeded) -> None:
    if staging_dir is None:
        return
    root = scorer_dir(staging_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "aborted",
        "resumable": True,
        "reason": exc.reason,
        "calls": exc.calls,
        "cost_usd": round(exc.cost_usd, 6),
    }
    (root / "abort.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _publish_scorer_artifacts(staging_dir: Path, run_dir: Path) -> None:
    """No-op when persist already moved staging; copy scorer/ if it was written beside staging.

    ``persist_run`` renames the staging directory to ``run_dir``, so the scorer artifacts
    written under staging land at ``run_dir/scorer/`` automatically. This helper only fills
    the gap when a test or alternate persist path left artifacts behind.
    """
    source = scorer_dir(staging_dir)
    dest = scorer_dir(run_dir)
    if dest.is_dir() or not source.is_dir():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        target = dest / path.name
        if not target.exists():
            target.write_bytes(path.read_bytes())
