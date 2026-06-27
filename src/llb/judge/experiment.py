"""Recorded Ukrainian judge sanity experiment for local OpenAI-compatible models."""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llb.contracts import JudgeDiagnostics, JudgeInputRecord, JudgeScore
from llb.paths import resolve_data_dir
from llb.scoring.judge import (
    UA_ANSWER_RELEVANCY_STEPS,
    UA_FAITHFULNESS_STEPS,
    UkrainianGEvalTemplate,
    deepeval_scorer,
    judge_experiment_metadata,
)
from llb.scoring.judge_diag import summarize_judge_diagnostics

METHOD = "judge-experiment"
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S.%fZ"

EXPERIMENT_CASES: list[JudgeInputRecord] = [
    {
        "question": "Яка столиця України?",
        "answer": "Київ є столицею України.",
        "contexts": ["Київ є столицею України."],
    },
    {
        "question": "Яка столиця України?",
        "answer": "Львів є столицею України.",
        "contexts": ["Київ є столицею України."],
    },
    {
        "question": "Яка столиця України?",
        "answer": "Дніпро впадає в Чорне море.",
        "contexts": ["Дніпро впадає в Чорне море."],
    },
]

JudgeScorer = Callable[[list[JudgeInputRecord], str], list[JudgeScore]]

# A single grounded-TRUE case: a healthy strict-JSON judge scores it well above zero, so a zero or
# malformed result is an unambiguous judge fault (not a candidate fault).
SMOKE_CASE: JudgeInputRecord = EXPERIMENT_CASES[0]


@dataclass(slots=True)
class JudgeSmokeResult:
    """Outcome of the M7.2 strict-JSON judge smoke precheck."""

    ok: bool
    reason: str
    score: JudgeScore | None
    diagnostics: JudgeDiagnostics


def _well_formed(score: JudgeScore) -> bool:
    """True only when both judge signals are finite floats in [0, 1] (a strict-JSON score)."""
    for key in ("faithfulness", "answer_relevancy"):
        value = score.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return False
        if not 0.0 <= float(value) <= 1.0:
            return False
    return True


def judge_smoke_check(
    judge_model: str,
    *,
    base_url: str | None = None,
    scorer: JudgeScorer | None = None,
) -> JudgeSmokeResult:
    """Strict-JSON judge smoke precheck: run ONE grounded-true case and verify a well-formed,
    non-zero score before a long judged run, so a local judge that cannot emit strict JSON (or
    whose endpoint is unreachable) is caught up front and named, not discovered mid-run.

    `scorer` is injectable so the precheck is provable from a fake judge with no DeepEval / endpoint;
    the default scorer is the real DeepEval judge bound to `base_url`."""
    records = [SMOKE_CASE]
    reasons: list[str | None] = []
    if scorer is None:
        scores = deepeval_scorer(records, judge_model, base_url=base_url, diagnostics_out=reasons)
    else:
        scores = scorer(records, judge_model)
    diagnostics = summarize_judge_diagnostics(records, scores, reasons or None)
    score = scores[0] if scores else None
    if score is None:
        return JudgeSmokeResult(False, "judge returned no score", None, diagnostics)
    if not _well_formed(score):
        return JudgeSmokeResult(
            False, "judge score is not a well-formed strict-JSON score", score, diagnostics
        )
    if diagnostics["n_zero"]:
        return JudgeSmokeResult(
            False,
            f"grounded-true smoke case scored zero (reasons={diagnostics['reasons']})",
            score,
            diagnostics,
        )
    return JudgeSmokeResult(True, "ok", score, diagnostics)


def run_judge_experiment(
    judge_model: str,
    *,
    base_url: str | None = None,
    data_dir: Path | str | None = None,
    scorer: JudgeScorer | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run fixed UA sanity cases and persist the non-secret experiment record."""
    score_fn: JudgeScorer
    if scorer is None:

        def score_fn(records: list[JudgeInputRecord], model: str) -> list[JudgeScore]:
            return deepeval_scorer(records, model, base_url=base_url)

    else:
        score_fn = scorer
    scores = score_fn(EXPERIMENT_CASES, judge_model)
    if len(scores) != len(EXPERIMENT_CASES):
        raise ValueError(
            f"judge returned {len(scores)} scores for {len(EXPERIMENT_CASES)} experiment cases"
        )

    timestamp = datetime.now(timezone.utc).strftime(_TIMESTAMP_FORMAT)
    out_dir = resolve_data_dir(data_dir) / METHOD / timestamp
    out_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "timestamp": timestamp,
        "judge": judge_experiment_metadata(judge_model, base_url),
        "prompts": {
            "faithfulness_steps": UA_FAITHFULNESS_STEPS,
            "answer_relevancy_steps": UA_ANSWER_RELEVANCY_STEPS,
            "result_template": UkrainianGEvalTemplate.generate_evaluation_results(
                "<кроки оцінювання>",
                "<тестовий приклад>",
                "<параметри>",
            ),
        },
        "cases": [
            {"input": record, "scores": score} for record, score in zip(EXPERIMENT_CASES, scores)
        ],
    }
    out_path = out_dir / "result.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report, out_path
