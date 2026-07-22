"""Markdown and JSONL persistence for the query robustness probe."""

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.eval.query_robustness import RobustnessResult


def render_report(result: RobustnessResult, metadata: Mapping[str, object]) -> str:
    typo_rate = metadata["typo_rate"]
    if not isinstance(typo_rate, int | float):
        raise TypeError("typo_rate metadata must be numeric")
    lines = [
        "# Ukrainian query robustness benchmark",
        "",
        f"- model: `{metadata['model']}`",
        f"- backend: `{metadata['backend']}`",
        f"- split: `{metadata['split']}`",
        f"- seed: {metadata['seed']}",
        f"- keyboard/homoglyph rate: {typo_rate:.3f}",
        f"- clean baseline: `{metadata['clean_run_dir']}`",
        f"- clean objective: {result.clean_objective:.4f}",
        f"- clean recall@k: {result.clean_recall:.4f}",
        "",
        "Variant rows are probe-only and live in `robustness.jsonl`; they never enter the clean",
        "run's `scores.jsonl` or correctness aggregates. Generation delta is measured only on",
        "items where both the clean and noisy lane retrieved gold evidence.",
        "Mitigation lanes are isolated: `normalize` inverts only attributable noise, while",
        "`normalize,typos` adds corpus-vocabulary correction under the Ukrainian morphology",
        "guard, so vocabulary-correction risk is read apart from normalization recovery.",
        "Recovery columns are measured against the `off` lane of the same noise class.",
        "",
        "| Class | Mitigation | N | Errors | Objective | Obj delta | Recall | Recall delta | Shared hits | Generation delta | Obj recovery | Recall recovery |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane in result.lanes:
        lines.append(
            f"| {lane.variant_class} | `{lane.mitigation}` | {lane.n} | "
            f"{lane.errors} | {lane.objective_score:.4f} | {lane.objective_delta:+.4f} | "
            f"{lane.recall_at_k:.4f} | {lane.recall_delta:+.4f} | {lane.shared_hit_n} | "
            f"{lane.generation_delta_on_shared_hits:+.4f} | {lane.objective_recovery:+.4f} | "
            f"{lane.recall_recovery:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def write_robustness_artifacts(
    result: RobustnessResult,
    out_dir: Path,
    metadata: Mapping[str, object],
) -> dict[str, str]:
    """Atomically publish only the probe report and rows under the method run directory."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        raise FileExistsError(f"query robustness artifacts already exist in {out_dir}")
    staging = Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=f".{out_dir.name}.tmp-"))
    try:
        report_path = staging / "report.md"
        rows_path = staging / "robustness.jsonl"
        atomic_write_text(report_path, render_report(result, metadata))
        atomic_write_text(
            rows_path,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result.rows),
        )
        staging.replace(out_dir)
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "report": str(out_dir / "report.md"),
        "robustness": str(out_dir / "robustness.jsonl"),
    }
