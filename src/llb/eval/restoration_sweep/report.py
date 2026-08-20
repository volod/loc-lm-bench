"""Markdown and JSONL publication for the restoration constraint sweep."""

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.eval.restoration_sweep.lanes import LaneReading, SweepResult
from llb.eval.restoration_sweep.verdict import (
    CONSTANT_KNOBS,
    ConstantVerdict,
    recommended_policy,
)
from llb.rag.fusion_evidence.paired import PairedComparison, format_randomization_p
from llb.rag.fusion_evidence.stats import format_interval
from llb.rag.query_prep.restore_policy import DEFAULT_RESTORATION_POLICY

_SETTING_HEADER = (
    "| Setting | Class | N | Recall@k | MRR | Corrections | Judged | Wrong | Wrong share "
    "| Opportunities | Restored | Restoration recall |"
)
_SETTING_RULE = (
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)


def _metric_row(label: str, reading: LaneReading) -> str:
    counts = reading.counts
    return (
        f"| {label} | {reading.variant_class} | {reading.n} | {reading.recall_at_k:.4f} "
        f"| {reading.mrr:.4f} | {counts.corrections} | {counts.labeled} | {counts.wrong} "
        f"| {counts.wrong_share:.4f} | {counts.opportunities} | {counts.restored} "
        f"| {counts.restoration_recall:.4f} |"
    )


def _lane_rows(result: SweepResult, lane: str, label: str) -> list[str]:
    return [_metric_row(label, reading) for reading in result.lane_readings(lane)]


def _header(result: SweepResult, metadata: Mapping[str, object]) -> list[str]:
    typo_rate = metadata["typo_rate"]
    if not isinstance(typo_rate, int | float):
        raise TypeError("typo_rate metadata must be numeric")
    return [
        "# Restoration constraint threshold sweep",
        "",
        f"- goldset: `{metadata['goldset']}`",
        f"- split: `{metadata['split']}` (n={len(result.item_ids)})",
        f"- embedding model: `{metadata['embedding_model']}`",
        f"- retrieval depth: k={result.top_k}",
        f"- seed: {metadata['seed']}",
        f"- keyboard/homoglyph rate: {typo_rate:.3f}",
        f"- noise classes: {', '.join(f'`{name}`' for name in result.variant_classes)}",
        f"- lane: `{metadata['lane']}` (morphology guard "
        f"{'on' if metadata.get('typo_guard') else 'off'}, dense-lane casing "
        f"{'on' if metadata.get('query_prep_dense_case') else 'off'})",
        f"- settings: {len(result.policies)}",
        f"- default setting: `{DEFAULT_RESTORATION_POLICY.label}`",
        "",
        "RETRIEVAL ONLY: no answer is generated, so a setting costs a store pass rather than a",
        "model run. The constants decide which corpus surface a noisy query token is rewritten to,",
        "which is a retrieval move; what a model does with the evidence is measured by",
        "`bench-query-robustness` under the setting this sweep pins.",
        "",
        "`Corrections` counts every vocabulary correction the setting made; `Judged` is the subset",
        "whose clean source token the noisy/clean alignment could identify, and `Wrong share` is",
        "the share of those that produced a token the user did not type. `Opportunities` counts",
        "noised tokens whose clean form the corpus does contain -- the corrections the constraints",
        "could have made -- and `Restoration recall` is the share of them they did make.",
    ]


def _reference_section(result: SweepResult) -> list[str]:
    lanes = tuple(dict.fromkeys(reading.lane for reading in result.references))
    lines = [
        "",
        "## Reference lanes",
        "",
        "None of these consults the restoration constraints, so they are measured once and bound",
        "every setting below: `clean` is the unperturbed question, `off` the noisy question with",
        "no query prep, `normalize` safe normalization alone.",
        "",
        _SETTING_HEADER,
        _SETTING_RULE,
    ]
    for lane in lanes:
        lines.extend(_lane_rows(result, lane, f"`{lane}`"))
    return lines


def _settings_section(result: SweepResult) -> list[str]:
    lines = [
        "",
        "## Settings",
        "",
        _SETTING_HEADER,
        _SETTING_RULE,
    ]
    for policy in result.policies:
        lines.extend(_lane_rows(result, policy.label, f"`{policy.label}`"))
    return lines


def _paired_row(label: str, comparison: PairedComparison, reading: str) -> str:
    return (
        f"| `{label}` | {format_interval(comparison['delta'], 4)} | {reading} "
        f"| {format_randomization_p(comparison)} "
        f"| {comparison['wins']}/{comparison['losses']}/{comparison['ties']} |"
    )


def _verdict_section(verdicts: Sequence[ConstantVerdict]) -> list[str]:
    lines = [
        "",
        "## Paired recall against the default setting",
        "",
        "Pooled over the noise classes, paired per item. A positive delta means the alternative",
        "retrieved gold evidence on items the default missed.",
        "",
        "| Setting | Recall delta | Reading | rand p | wins/losses/ties |",
        "| --- | ---: | :-: | ---: | :-: |",
    ]
    for verdict in verdicts:
        for alternative in verdict.alternatives:
            lines.append(_paired_row(alternative.label, alternative.recall, alternative.reading))
    lines.extend(
        [
            "",
            "## Verdict per constant",
            "",
            "| Constant | Default | Verdict | Knob | Rationale |",
            "| --- | ---: | :-: | --- | --- |",
        ]
    )
    for verdict in verdicts:
        lines.append(
            f"| `{verdict.constant}` | {verdict.default_value} | **{verdict.verdict}** "
            f"| `{CONSTANT_KNOBS[verdict.constant]}` | {verdict.rationale} |"
        )
    recommended = recommended_policy(verdicts)
    lines.extend(
        [
            "",
            f"Recommended setting: `{recommended.label}`"
            + (
                " (unchanged from the shipped default)."
                if recommended == DEFAULT_RESTORATION_POLICY
                else " -- an `adopt` verdict moved it off the shipped default."
            ),
        ]
    )
    return lines


def render_report(
    result: SweepResult,
    verdicts: Sequence[ConstantVerdict],
    metadata: Mapping[str, object],
) -> str:
    """The whole sweep as one operator-readable report."""
    lines = _header(result, metadata)
    lines.extend(_reference_section(result))
    lines.extend(_settings_section(result))
    lines.extend(_verdict_section(verdicts))
    return "\n".join(lines) + "\n"


def write_sweep_artifacts(
    result: SweepResult,
    verdicts: Sequence[ConstantVerdict],
    out_dir: Path,
    metadata: Mapping[str, object],
) -> dict[str, str]:
    """Atomically publish the report, the per-setting metrics, and every labeled correction."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        raise FileExistsError(f"restoration sweep artifacts already exist in {out_dir}")
    staging = Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=f".{out_dir.name}.tmp-"))
    try:
        atomic_write_text(staging / "report.md", render_report(result, verdicts, metadata))
        atomic_write_text(
            staging / "settings.jsonl",
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result.metric_rows()),
        )
        atomic_write_text(
            staging / "edit_audit.jsonl",
            "".join(
                json.dumps(record.as_row(), ensure_ascii=False) + "\n" for record in result.edits
            ),
        )
        atomic_write_text(
            staging / "metadata.json",
            json.dumps(dict(metadata), ensure_ascii=False, indent=2, default=str) + "\n",
        )
        staging.replace(out_dir)
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "report": str(out_dir / "report.md"),
        "settings": str(out_dir / "settings.jsonl"),
        "edit_audit": str(out_dir / "edit_audit.jsonl"),
        "metadata": str(out_dir / "metadata.json"),
    }
