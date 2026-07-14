"""security benchmark security / robustness runner -- objective ASR + refusal-appropriateness under TIER_SECURITY.

Drives a candidate over a set of planted security cases (`scoring.security`), detects attack
success objectively per family, and aggregates one `ModelResult` whose headline is the DEFENSE
rate (1 - ASR; higher better, so it ranks under `rank_board` like any quality) on its OWN Tier --
never cross-ranked with the RAG board. ASR and refusal-appropriateness each carry a bootstrap CI.

The candidate is reached through an injectable `complete` (prompt -> raw text), so a FAKE endpoint
proves the whole flow with no GPU. The gated judge for borderline unsafe-content quality is opt-in
and out of this objective core (documented residual).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.bench.common import (
    DEFAULT_THRESHOLD,
    JudgeScorer,
    LLMComplete,
    Mirror,
    persist_category_run,
    render_board,
    verified_data_config,
)
from llb.bench.common_backend import ThroughputMeter
from llb.core.contracts import (
    BoardRow,
    RunPaths,
    SecurityCaseRow,
)
from llb.scoring import security
from llb.scoring.security_cases import load_security_cases
from llb.scoring.leaderboard import ModelResult
from llb.bench.security_scoring import (
    METHOD,
    _SecurityJudgeConfig,
    _run_refusal_judge,
    _score_cases,
)
from llb.bench.security_persistence import (
    _judge_config,
    _judge_status,
    _log_persisted_run,
    _score_config,
    _security_metrics,
    _security_result,
)


# The judge "intent" for the opt-in unsafe-content quality signal: given a harmful ask, a SAFE
# response refuses, explains the risk, and withholds harmful detail. answer-relevancy to this
# intent is high for a clean refusal and low for compliance, so it REFINES the objective
# `refuse` detection on borderline cases (a hedge / partial refusal) without a safety classifier.


@dataclass(slots=True)
class SecurityRun:
    """Outcome of one scored security run."""

    result: ModelResult
    score: security.SecurityScore
    rows: list[SecurityCaseRow]
    board: list[BoardRow]
    table: str
    asr_ci: tuple[float, float] | None
    refusal_ci: tuple[float, float] | None
    paths: RunPaths | None
    refusal_quality: float | None = None  # mean gated-judge quality (None when not trusted/run)
    refusal_quality_ci: tuple[float, float] | None = None
    judge_trusted: bool = False
    judge_reason: str = "no judge configured"


def run_security(
    cases: list[security.SecurityCase],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    judge_model: str | None = None,
    judge_rho: float | None = None,
    judge_threshold: float = DEFAULT_THRESHOLD,
    judge_scorer: JudgeScorer | None = None,
    judge_base_url: str | None = None,
    data_dir: Path | str | None = None,
    run_name: str = "security",
    persist: bool = True,
    mirror: Mirror | None = None,
    data_verified: bool = False,
    verification_ref: str | None = None,
    meter: ThroughputMeter | None = None,
) -> SecurityRun:
    """Score one model's robustness over the planted cases and return its board under TIER_SECURITY.

    Objective defense rate is the headline. When a judge is configured AND trusted
    (`judge_rho >= judge_threshold`), an opt-in unsafe-content REFUSAL-QUALITY signal is recorded
    ALONGSIDE (per harmful-ask case + mean + CI) but never folded into the headline; otherwise the
    judge is demoted and the objective ASR ranks alone. `judge_scorer` is injectable for tests.
    A `meter` (populated by the endpoint `complete`) supplies the run's real generation tok/s.
    """
    if not cases:
        raise SystemExit("no security cases provided")
    verification_cfg = verified_data_config(
        data_verified=data_verified, verification_ref=verification_ref
    )
    scored = _score_cases(cases, complete)
    tokens_per_s = meter.tokens_per_s if meter is not None else 0.0
    judge_cfg = _SecurityJudgeConfig(
        model=judge_model,
        rho=judge_rho,
        threshold=judge_threshold,
        scorer=judge_scorer,
        base_url=judge_base_url,
    )
    judge = _run_refusal_judge(cases, scored.outputs, scored.rows, judge_cfg)
    result = _security_result(model, backend, scored, tokens_per_s)
    board, table = render_board([result])

    paths: RunPaths | None = None
    if persist and data_dir is not None:
        config = {
            **_score_config(model, backend, scored.score, scored.asr_ci, scored.refusal_ci),
            **_judge_config(judge),
            **verification_cfg,
        }
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=run_name,
            config=config,
            metrics=_security_metrics(result, scored.reliability, tokens_per_s),
            case_rows=scored.rows,
            judge=_judge_status(judge_cfg, judge),
            mirror=mirror,
        )
        _log_persisted_run(model, scored.score, judge, paths, tokens_per_s)
    return SecurityRun(
        result=result,
        score=scored.score,
        rows=scored.rows,
        board=board,
        table=table,
        asr_ci=scored.asr_ci,
        refusal_ci=scored.refusal_ci,
        paths=paths,
        refusal_quality=judge.quality,
        refusal_quality_ci=judge.quality_ci,
        judge_trusted=judge.outcome.trusted,
        judge_reason=judge.outcome.reason,
    )


def load_cases_file(path: Path | str) -> list[security.SecurityCase]:
    """Load a committed security-case set (a JSON array of case records)."""
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array of security cases")
    return load_security_cases(raw)
