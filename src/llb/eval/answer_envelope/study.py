"""Roster conformance study for the declared answer contract (typed-rag-answer-envelope).

Envelope conformance is a property of the MODEL, not of the harness: the same prompt, the same
context, and the same validator produce different verdicts on different weights, which is the whole
reason the format is adopted per model with evidence rather than switched on by construction. This
study reads finished envelope bundles -- one per model over ONE item set -- and reports what each
model did with the contract.

Two separations are the point of the report and are enforced by its shape:

  - FORMAT from REASONING. Conformance, the two failure rates, and the repair rate are reported
    beside correctness, never blended into it. A model whose repair reprompt fixes its formatting
    gains conformance, not objective score, and the table shows both so nobody can read one as the
    other.
  - TRUNCATION from NON-CONFORMANCE. A completion cut off at the run's token cap is not JSON
    either, and would otherwise be counted as a model that cannot emit the shape. The share of
    non-conformant cases that reached the cap is reported, so a conformance number produced by too
    small a completion budget is visible rather than believed.
  - FIRST ATTEMPT from FINAL. `repair_rate` is the share of cases whose first completion failed the
    contract, so first-attempt conformance is `1 - repair_rate` and the repair's contribution is
    the gap up to final conformance. A large gap means a model that CAN emit the shape but does not
    do it unprompted -- a different (and cheaper) problem than one that cannot.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llb.core.fsutil import atomic_write_text
from llb.eval import common as eval_common
from llb.eval.paired_cases import CaseRows, recorded_lane_rows, shared_item_ids
from llb.scoring.verbosity import mean

# A bundle with no envelope column cannot answer this study at all -- it measured a different lane.
REQUIRED_COLUMNS = ("envelope_status", "repaired", "n_claims")
# Correctness columns reported BESIDE conformance, never folded into it.
CORRECTNESS_COLUMNS = ("objective_score", "ranking_score", "contains")

REPORT_NAME = "report.md"
JSON_NAME = "report.json"


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{run_dir}: manifest.json is not a JSON object")
    return payload


def _check_envelope_bundle(run_dir: Path, config: Mapping[str, Any], rows: Sequence[Any]) -> None:
    """Refuse a bundle that did not run the envelope lane, before any number is read from it."""
    recorded = str(config.get("answer_format", "free_text"))
    if recorded != "envelope":
        raise ValueError(f"{run_dir}: bundle ran answer_format={recorded!r}, not the envelope lane")
    missing = sorted({column for column in REQUIRED_COLUMNS for row in rows if column not in row})
    if missing:
        raise ValueError(f"{run_dir}: bundle carries no envelope columns: {missing}")


def _rate(rows: Sequence[Mapping[str, Any]], status: str) -> float:
    return sum(1 for row in rows if str(row["envelope_status"]) == status) / len(rows)


def _truncation_suspect_rate(rows: Sequence[Mapping[str, Any]], max_tokens: int) -> float:
    """Share of the NON-CONFORMANT cases whose completion reached the run's token cap.

    A truncated completion is not JSON, so it lands in the same bucket as prose. A high rate here
    says the completion budget, not the model, produced the conformance number -- the envelope
    needs a bigger budget than a one-line free-text answer, and this is what catches a run that did
    not get one. Zero non-conformant cases means nothing to suspect.
    """
    failed = [row for row in rows if str(row["envelope_status"]) != eval_common.OK]
    if not failed or max_tokens <= 0:
        return 0.0
    return sum(1 for row in failed if int(row.get("completion_tokens", 0)) >= max_tokens) / len(
        failed
    )


def _model_summary(run_dir: Path) -> tuple[str, CaseRows, dict[str, Any]]:
    """One bundle read as (model, its per-case rows, conformance + correctness numbers)."""
    manifest = _read_manifest(run_dir)
    rows = recorded_lane_rows([run_dir])
    if not rows:
        raise ValueError(f"{run_dir}: bundle scored no cases")
    config = manifest.get("config") or {}
    _check_envelope_bundle(run_dir, config, rows)
    repair_rate = sum(1 for row in rows if bool(row.get("repaired"))) / len(rows)
    conformance = _rate(rows, eval_common.OK)
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "n": len(rows),
        "conformance": conformance,
        "first_attempt_conformance": 1.0 - repair_rate,
        "repair_rate": repair_rate,
        "repair_gain": conformance - (1.0 - repair_rate),
        "schema_invalid_rate": _rate(rows, eval_common.SCHEMA_INVALID),
        "malformed_rate": _rate(rows, eval_common.MALFORMED),
        "mean_claims": mean([float(row.get("n_claims", 0)) for row in rows]),
        "abstention_rate": sum(1 for row in rows if bool(row.get("envelope_abstained")))
        / len(rows),
        "max_tokens": int(config.get("max_tokens", 0) or 0),
        "truncation_suspect_rate": _truncation_suspect_rate(
            rows, int(config.get("max_tokens", 0) or 0)
        ),
    }
    summary.update(
        {
            column: mean([float(row.get(column, 0.0)) for row in rows])
            for column in CORRECTNESS_COLUMNS
        }
    )
    model = str(config.get("model") or manifest.get("run_name") or run_dir.name)
    return model, rows, summary


def analyze(run_dirs: Sequence[Path | str]) -> dict[str, Any]:
    """Compare what each roster model did with the answer contract over ONE shared item set."""
    if len(run_dirs) < 2:
        raise ValueError(
            "the conformance study needs at least two envelope bundles (one per model)"
        )
    models: dict[str, dict[str, Any]] = {}
    lanes: dict[str, CaseRows] = {}
    for run_dir in run_dirs:
        model, rows, summary = _model_summary(Path(run_dir))
        if model in models:
            raise ValueError(
                f"the conformance study needs one bundle per model; duplicate: {model}"
            )
        models[model] = summary
        lanes[model] = rows
    # Conformance is only comparable over the SAME questions; a lane that scored a different set
    # fails here rather than being silently intersected away.
    item_ids = shared_item_ids(lanes)
    return {
        "n": len(item_ids),
        "item_ids": item_ids,
        "models": models,
        "conformance_order": sorted(
            models, key=lambda model: (-models[model]["conformance"], model)
        ),
        "objective_order": sorted(
            models, key=lambda model: (-models[model]["objective_score"], model)
        ),
    }


def render(report: Mapping[str, Any]) -> str:
    """The maintained ASCII evidence artifact: conformance first, correctness beside it."""
    lines = [
        "# Answer-envelope conformance by model",
        "",
        f"- fixed item set: {report['n']} items, one envelope bundle per model",
        "- format and reasoning are separate columns: a repair gain is a FORMATTING gain",
        "",
        "| model | conformance | first attempt | repaired | schema_invalid | malformed "
        "| truncated | mean claims | abstained | objective | policy | found |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in report["conformance_order"]:
        m = report["models"][model]
        lines.append(
            f"| `{model}` | {m['conformance']:.3f} | {m['first_attempt_conformance']:.3f} "
            f"| {m['repair_rate']:.3f} | {m['schema_invalid_rate']:.3f} "
            f"| {m['malformed_rate']:.3f} | {m['truncation_suspect_rate']:.3f} "
            f"| {m['mean_claims']:.2f} "
            f"| {m['abstention_rate']:.3f} | {m['objective_score']:.3f} "
            f"| {m['ranking_score']:.3f} | {m['contains']:.3f} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- `conformance` is the share of cases whose FINAL completion satisfied the contract;",
        "  `first attempt` is `1 - repaired`, and the gap between them is what the one bounded",
        "  repair reprompt bought -- formatting, never reasoning.",
        "- `schema_invalid` (JSON of the wrong shape) and `malformed` (not JSON at all) are kept",
        "  apart because they call for different fixes.",
        "- `truncated` is the share of the NON-conformant cases that reached the run's completion",
        "  token cap: a high value means the budget produced the number, not the model.",
        "- `objective` / `policy` / `found` are the ordinary correctness columns, unchanged by the",
        "  format: the envelope's own `answer` string is scored exactly as free text would be.",
        "",
        f"Conformance order: {', '.join(report['conformance_order'])}.",
        f"Objective order: {', '.join(report['objective_order'])}.",
    ]
    return "\n".join(lines) + "\n"


def write(report: Mapping[str, Any], out_dir: Path) -> dict[str, str]:
    """Persist the study artifacts under `$DATA_DIR/answer-envelope/<run>/`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / REPORT_NAME
    json_path = out_dir / JSON_NAME
    atomic_write_text(report_path, render(report))
    atomic_write_text(json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {"report": str(report_path), "json": str(json_path)}
