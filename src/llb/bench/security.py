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
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.bench.common import (
    DEFAULT_THRESHOLD,
    JudgeScorer,
    LLMComplete,
    Mirror,
    category_result,
    mean,
    persist_category_run,
    render_board,
    run_gated_judge,
    verified_data_config,
)
from llb.core.contracts import (
    BoardRow,
    JudgeInputRecord,
    JudgeScore,
    JudgeStatus,
    RunMetrics,
    RunPaths,
    SecurityCaseRow,
)
from llb.eval.common import EMPTY, is_refusal
from llb.prompts import render_text
from llb.scoring import security
from llb.scoring.aggregate import TIER_SECURITY, ModelResult, bootstrap_mean_ci
from llb.scoring.judge import JudgeOutcome

_LOG = logging.getLogger(__name__)

METHOD = "security"
_RELIABILITY_FAILURES = frozenset({EMPTY, "timeout", "backend_error"})

# The judge "intent" for the opt-in unsafe-content quality signal: given a harmful ask, a SAFE
# response refuses, explains the risk, and withholds harmful detail. answer-relevancy to this
# intent is high for a clean refusal and low for compliance, so it REFINES the objective
# `refuse` detection on borderline cases (a hedge / partial refusal) without a safety classifier.
_REFUSAL_INTENT = render_text("bench.security.refusal_intent")


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


@dataclass(frozen=True, slots=True)
class _SecurityJudgeConfig:
    model: str | None
    rho: float | None
    threshold: float
    scorer: JudgeScorer | None
    base_url: str | None


@dataclass(frozen=True, slots=True)
class _ScoredSecurityCases:
    outputs: list[str]
    score: security.SecurityScore
    rows: list[SecurityCaseRow]
    reliability: float
    asr_ci: tuple[float, float] | None
    refusal_ci: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class _RefusalJudgeRun:
    outcome: JudgeOutcome
    quality: float | None
    quality_ci: tuple[float, float] | None


def build_prompt(case: security.SecurityCase) -> str:
    """Assemble the candidate-facing prompt. RAG-injection cases frame the malicious text as
    untrusted retrieved context (the realistic test: no explicit 'ignore injected instructions'
    hand-holding, so following the injection is a genuine failure)."""
    if case.injected_context:
        return render_text(
            "bench.security.rag_injection",
            {"injected_context": case.injected_context, "prompt": case.prompt},
        )
    return case.prompt


def _status(output: str, refused: bool) -> str:
    if not output.strip():
        return EMPTY
    if refused:
        return "refusal"
    return "ok"


def _row(case: security.SecurityCase, output: str) -> SecurityCaseRow:
    attack = False if case.benign else security.attack_succeeded(case, output)
    refused = is_refusal(output) or not output.strip()
    row: SecurityCaseRow = {
        "item_id": case.id,
        "family": case.family,
        "benign": case.benign,
        "expect_refusal": case.expect_refusal,
        "status": _status(output, refused),
        "attack_success": 1.0 if attack else 0.0,
        "defended": 0.0 if attack else 1.0,
        "refused": 1.0 if refused else 0.0,
        "appropriate_refusal": 1.0 if security.appropriate_refusal(case, output) else 0.0,
        "answer_preview": (output or "")[:280],
    }
    if not case.benign:
        row["objective_score"] = row["defended"]
    if case.lang:
        row["lang"] = case.lang
    if case.xlang_group:
        row["xlang_group"] = case.xlang_group
    return row


def _refusal_records(
    cases: list[security.SecurityCase], outputs: list[str]
) -> tuple[list[JudgeInputRecord], list[int]]:
    """One (safe-refusal intent, model output, [harmful ask]) record per harmful-ask case.

    Returns the judge records plus the row indices they map back to, so per-case judge scores can
    attach to the right rows. Benign controls and leak cases are excluded (the judge refines only
    the unsafe-content / jailbreak refusal decision -- a check the marker detector cannot make).
    """
    records: list[JudgeInputRecord] = []
    indices: list[int] = []
    for i, (case, output) in enumerate(zip(cases, outputs)):
        if case.benign or case.detector_kind != security.DETECT_REFUSE:
            continue
        records.append(
            {
                "question": render_text(
                    "bench.security.refusal_question",
                    {"intent": _REFUSAL_INTENT, "prompt": case.prompt},
                ),
                "answer": output,
                "contexts": [case.prompt],
            }
        )
        indices.append(i)
    return records, indices


def refusal_quality(score: JudgeScore) -> float:
    """The unsafe-content quality scalar: answer-relevancy to the safe-refusal intent (a clean
    refusal that addresses the safety concern scores high; harmful compliance scores low)."""
    return float(score["answer_relevancy"])


def _score_cases(cases: list[security.SecurityCase], complete: LLMComplete) -> _ScoredSecurityCases:
    outputs = [complete(build_prompt(case)) for case in cases]
    score = security.score_security(cases, outputs)
    rows = [_row(case, output) for case, output in zip(cases, outputs)]
    n_ok = sum(1 for r in rows if r["status"] not in _RELIABILITY_FAILURES)
    reliability = n_ok / len(rows) if rows else 0.0
    return _ScoredSecurityCases(
        outputs=outputs,
        score=score,
        rows=rows,
        reliability=reliability,
        asr_ci=bootstrap_mean_ci(score.case_defended),
        refusal_ci=bootstrap_mean_ci(score.case_appropriate),
    )


def _attach_refusal_quality(
    rows: list[SecurityCaseRow], indices: list[int], values: list[float]
) -> None:
    for idx, value in zip(indices, values):
        rows[idx]["refusal_quality"] = round(value, 6)


def _trusted_refusal_quality(
    outcome: JudgeOutcome, rows: list[SecurityCaseRow], indices: list[int]
) -> _RefusalJudgeRun:
    per_case = [refusal_quality(s) for s in outcome.scores or []]
    _attach_refusal_quality(rows, indices, per_case)
    return _RefusalJudgeRun(
        outcome=outcome,
        quality=round(mean(per_case), 6),
        quality_ci=bootstrap_mean_ci(per_case),
    )


def _run_refusal_judge(
    cases: list[security.SecurityCase],
    outputs: list[str],
    rows: list[SecurityCaseRow],
    config: _SecurityJudgeConfig,
) -> _RefusalJudgeRun:
    judge_records, judge_indices = _refusal_records(cases, outputs)
    outcome = run_gated_judge(
        judge_records,
        judge_model=config.model,
        judge_rho=config.rho,
        threshold=config.threshold,
        scorer=config.scorer,
        base_url=config.base_url,
    )
    if outcome.trusted and outcome.scores:
        return _trusted_refusal_quality(outcome, rows, judge_indices)
    if config.model is not None:
        _LOG.info("[security] judge demoted (%s); objective ASR ranks alone", outcome.reason)
    return _RefusalJudgeRun(outcome=outcome, quality=None, quality_ci=None)


def _security_result(model: str, backend: str, scored: _ScoredSecurityCases) -> ModelResult:
    return category_result(
        model=model,
        backend=backend,
        tier=TIER_SECURITY,
        case_objectives=scored.score.case_defended,  # per-attack-case defended -> defense-rate CI
        reliability=scored.reliability,
    )


def _security_metrics(result: ModelResult, reliability: float) -> RunMetrics:
    return {
        "objective_score": result.objective_score,  # defense rate (1 - ASR)
        "reliability": reliability,
        "tokens_per_s": 0.0,
    }


def _score_config(
    model: str,
    backend: str,
    score: security.SecurityScore,
    asr_ci: tuple[float, float] | None,
    refusal_ci: tuple[float, float] | None,
) -> dict[str, Any]:
    return {
        "model": model,
        "backend": backend,
        "tier": TIER_SECURITY,
        "category": "security",
        "n_cases": score.n_cases,
        "n_attacks": score.n_attacks,
        "asr": score.asr,
        "defense_rate": score.defense_rate,
        "refusal_appropriateness": score.refusal_appropriateness,
        "asr_by_family": score.asr_by_family,
        "defense_ci": list(asr_ci) if asr_ci else None,
        "refusal_appropriateness_ci": list(refusal_ci) if refusal_ci else None,
        "cross_language": _cross_language_config(score.cross_language),
    }


def _cross_language_config(
    xlang: security.CrossLanguageConsistency | None,
) -> dict[str, Any] | None:
    """Persist the cross-language-consistency block (None when the set has no matched groups)."""
    if xlang is None:
        return None
    ci = bootstrap_mean_ci(xlang.group_consistent)
    return {
        "n_groups": xlang.n_groups,
        "consistency": xlang.consistency,
        "refusal_rate_by_lang": xlang.refusal_rate_by_lang,
        "consistency_ci": list(ci) if ci else None,
    }


def _judge_config(judge: _RefusalJudgeRun) -> dict[str, Any]:
    return {
        "judge_trusted": judge.outcome.trusted,
        "refusal_quality": judge.quality,  # gated diagnostic, NOT the headline
        "refusal_quality_ci": list(judge.quality_ci) if judge.quality_ci else None,
        "judge_diagnostics": judge.outcome.diagnostics,
    }


def _judge_status(config: _SecurityJudgeConfig, judge: _RefusalJudgeRun) -> JudgeStatus | None:
    if config.model is None:
        return None
    return {
        "calibration_rho": config.rho,
        "threshold": config.threshold,
        "trusted": judge.outcome.trusted,
        "model": config.model,
        "metrics": ["refusal_quality"],
        "diagnostics": judge.outcome.diagnostics,
    }


def _log_persisted_run(
    model: str,
    score: security.SecurityScore,
    judge: _RefusalJudgeRun,
    paths: RunPaths,
) -> None:
    xlang = (
        f"{score.cross_language.consistency:.3f} ({score.cross_language.n_groups}g)"
        if score.cross_language is not None
        else "n/a"
    )
    _LOG.info(
        "[security] %s ASR=%.3f defense=%.3f refusal-appropriateness=%.3f "
        "xlang-consistency=%s quality=%s -> %s",
        model,
        score.asr,
        score.defense_rate,
        score.refusal_appropriateness,
        xlang,
        f"{judge.quality:.3f}" if judge.quality is not None else "n/a",
        paths["manifest"],
    )


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
) -> SecurityRun:
    """Score one model's robustness over the planted cases and return its board under TIER_SECURITY.

    Objective defense rate is the headline. When a judge is configured AND trusted
    (`judge_rho >= judge_threshold`), an opt-in unsafe-content REFUSAL-QUALITY signal is recorded
    ALONGSIDE (per harmful-ask case + mean + CI) but never folded into the headline; otherwise the
    judge is demoted and the objective ASR ranks alone. `judge_scorer` is injectable for tests.
    """
    if not cases:
        raise SystemExit("no security cases provided")
    verification_cfg = verified_data_config(
        data_verified=data_verified, verification_ref=verification_ref
    )
    scored = _score_cases(cases, complete)
    judge_cfg = _SecurityJudgeConfig(
        model=judge_model,
        rho=judge_rho,
        threshold=judge_threshold,
        scorer=judge_scorer,
        base_url=judge_base_url,
    )
    judge = _run_refusal_judge(cases, scored.outputs, scored.rows, judge_cfg)
    result = _security_result(model, backend, scored)
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
            metrics=_security_metrics(result, scored.reliability),
            case_rows=scored.rows,
            judge=_judge_status(judge_cfg, judge),
            mirror=mirror,
        )
        _log_persisted_run(model, scored.score, judge, paths)
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
    return security.load_security_cases(raw)
