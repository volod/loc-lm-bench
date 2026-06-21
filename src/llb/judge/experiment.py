"""Recorded Ukrainian judge sanity experiment for local OpenAI-compatible models."""

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llb.contracts import JudgeInputRecord, JudgeScore
from llb.paths import resolve_data_dir
from llb.scoring.judge import (
    UA_ANSWER_RELEVANCY_STEPS,
    UA_FAITHFULNESS_STEPS,
    UkrainianGEvalTemplate,
    deepeval_scorer,
    judge_experiment_metadata,
)

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
