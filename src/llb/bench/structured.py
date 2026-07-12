"""structured-output runner -- JSON-schema conformance + field accuracy under TIER_STRUCTURED.

Drives a candidate over instruction + target-schema cases, validates each output with Pydantic
(`scoring.structured`), and scores conformance + field accuracy. The headline is field accuracy
(non-conformant outputs score 0); the conformance rate is recorded alongside. Driven by an
injectable `complete`, so a FAKE endpoint proves the flow without a GPU.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.bench.common import (
    LLMComplete,
    Mirror,
    ThroughputMeter,
    category_result,
    persist_category_run,
    render_board,
    verified_data_config,
)
from llb.core.contracts import BoardRow, RunMetrics, RunPaths, StructuredCaseRow
from llb.prompts import render_text
from llb.scoring import structured
from llb.scoring.aggregate import TIER_STRUCTURED, ModelResult, bootstrap_mean_ci

_LOG = logging.getLogger(__name__)

METHOD = "structured"


@dataclass(slots=True)
class StructuredRun:
    result: ModelResult
    score: structured.StructuredScore
    rows: list[StructuredCaseRow]
    board: list[BoardRow]
    table: str
    accuracy_ci: tuple[float, float] | None
    paths: RunPaths | None


def structured_prompt(case: structured.StructuredCase) -> str:
    """Ask for a JSON object matching the target schema."""
    schema_json = json.dumps(case.schema, ensure_ascii=False, indent=2)
    return render_text(
        "bench.structured.structured",
        {"instruction": case.instruction, "schema_json": schema_json},
    )


def _row(score: structured.StructuredCaseScore) -> StructuredCaseRow:
    return {
        "item_id": score.item_id,
        "conformant": score.conformant,
        "field_accuracy": score.field_accuracy,
        "score": score.score,
        "objective_score": score.score,
    }


def run_structured(
    cases: list[structured.StructuredCase],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    data_dir: Path | str | None = None,
    run_name: str = "structured",
    persist: bool = True,
    mirror: Mirror | None = None,
    data_verified: bool = False,
    verification_ref: str | None = None,
    meter: ThroughputMeter | None = None,
) -> StructuredRun:
    """Score one model's structured-output conformance + field accuracy under TIER_STRUCTURED.

    A `meter` (populated by the endpoint `complete`) supplies the run's real generation tok/s.
    """
    if not cases:
        raise SystemExit("no structured-output cases provided")
    verification_cfg = verified_data_config(
        data_verified=data_verified, verification_ref=verification_ref
    )
    outputs = [complete(structured_prompt(c)) for c in cases]
    score = structured.score_structured(cases, outputs)
    rows = [_row(s) for s in score.cases]

    tokens_per_s = meter.tokens_per_s if meter is not None else 0.0
    result = category_result(
        model=model,
        backend=backend,
        tier=TIER_STRUCTURED,
        case_objectives=score.case_score,
        reliability=score.conformance_rate,
        tokens_per_s=tokens_per_s,
    )
    accuracy_ci = bootstrap_mean_ci(score.case_score)
    board, table = render_board([result])

    paths: RunPaths | None = None
    if persist and data_dir is not None:
        metrics: RunMetrics = {
            "objective_score": result.objective_score,  # field accuracy
            "reliability": score.conformance_rate,
            "tokens_per_s": tokens_per_s,
        }
        config: dict[str, Any] = {
            "model": model,
            "backend": backend,
            "tier": TIER_STRUCTURED,
            "category": "structured",
            "n_cases": score.n_cases,
            "field_accuracy": score.field_accuracy,
            "conformance_rate": score.conformance_rate,
            "field_accuracy_ci": list(accuracy_ci) if accuracy_ci else None,
            **verification_cfg,
        }
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=run_name,
            config=config,
            metrics=metrics,
            case_rows=rows,
            mirror=mirror,
        )
        _LOG.info(
            "[structured] %s field-accuracy=%.3f conformance=%.3f -> %s",
            model,
            score.field_accuracy,
            score.conformance_rate,
            paths["manifest"],
        )
    return StructuredRun(
        result=result,
        score=score,
        rows=rows,
        board=board,
        table=table,
        accuracy_ci=accuracy_ci,
        paths=paths,
    )


def load_cases_file(path: Path | str) -> list[structured.StructuredCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array of structured-output cases")
    return [structured.StructuredCase.from_record(r) for r in raw]
