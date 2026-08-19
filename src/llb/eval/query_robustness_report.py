"""Markdown and JSONL persistence for the query robustness probe."""

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.eval.query_robustness import RobustnessResult
from llb.eval.query_robustness_languages import LANGUAGE_VARIANT_CLASSES
from llb.rag.fusion_evidence.stability import format_reading
from llb.rag.fusion_evidence.stats import format_interval
from llb.rag.fusion_evidence.paired import PairedComparison
from llb.rag.query_prep.restore_policy import DEFAULT_RESTORATION_POLICY, RestorationPolicy


def render_report(result: RobustnessResult, metadata: Mapping[str, object]) -> str:
    typo_rate = metadata["typo_rate"]
    if not isinstance(typo_rate, int | float):
        raise TypeError("typo_rate metadata must be numeric")
    classes = result.variant_classes or tuple(
        dict.fromkeys(lane.variant_class for lane in result.lanes)
    )
    lines = [
        "# Ukrainian RAG query robustness benchmark",
        "",
        f"- model: `{metadata['model']}`",
        f"- backend: `{metadata['backend']}`",
        f"- split: `{metadata['split']}`",
        f"- seed: {metadata['seed']}",
        f"- keyboard/homoglyph rate: {typo_rate:.3f}",
        f"- dense-lane casing: {'on' if metadata.get('query_prep_dense_case') else 'off'}",
        f"- restoration constraints: {_restoration_label(metadata)}",
        f"- noise classes: {', '.join(f'`{name}`' for name in classes)}",
        f"- clean baseline: `{metadata['clean_run_dir']}`",
        f"- clean objective: {result.clean_objective:.4f}",
        f"- clean recall@k: {result.clean_recall:.4f}",
        f"- clean MRR: {result.clean_mrr:.4f}",
        f"- paired bootstrap: {result.resamples} resamples at {result.confidence * 100:g}%",
        "",
        "Variant rows are probe-only and live in `robustness.jsonl`; they never enter the clean",
        "run's `scores.jsonl` or correctness aggregates. Generation delta is measured only on",
        "items where both the clean and variant lane retrieved gold evidence.",
    ]
    lines.extend(_method_notes(classes, metadata))
    lines.extend(
        [
            "Recovery columns are measured against the `off` lane of the same noise class.",
            "",
            "| Class | Mitigation | N | Errors | Objective | Obj delta | Recall | Recall delta | MRR | MRR delta | Shared hits | Generation delta | Obj recovery | Recall recovery | MRR recovery |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lane in result.lanes:
        lines.append(
            f"| {lane.variant_class} | `{lane.mitigation}` | {lane.n} | "
            f"{lane.errors} | {lane.objective_score:.4f} | {lane.objective_delta:+.4f} | "
            f"{lane.recall_at_k:.4f} | {lane.recall_delta:+.4f} | {lane.mrr:.4f} | "
            f"{lane.mrr_delta:+.4f} | {lane.shared_hit_n} | "
            f"{lane.generation_delta_on_shared_hits:+.4f} | {lane.objective_recovery:+.4f} | "
            f"{lane.recall_recovery:+.4f} | {lane.mrr_recovery:+.4f} |"
        )
    lines.extend(_uncertainty_section(result))
    lines.extend(_affected_section(result))
    return "\n".join(lines) + "\n"


def _restoration_label(metadata: Mapping[str, object]) -> str:
    """The run's restoration constants, so two sweep settings' reports read against each other."""
    constants = metadata.get("restoration_constants")
    if not isinstance(constants, Mapping):
        return f"`{DEFAULT_RESTORATION_POLICY.label}`"
    policy = RestorationPolicy(
        surface_max_distance=int(constants["query_prep_surface_max_distance"]),
        ambiguous_token_max_chars=int(constants["query_prep_ambiguous_max_chars"]),
        rank_order=str(constants["query_prep_restore_rank"]),
    )
    return f"`{policy.label}`"


def _method_notes(classes: tuple[str, ...], metadata: Mapping[str, object]) -> list[str]:
    notes: list[str] = []
    if any(name not in LANGUAGE_VARIANT_CLASSES for name in classes):
        notes.extend(
            [
                "Character-noise classes isolate one mechanism each. `normalize` only inverts",
                "attributable noise; `normalize,typos` adds guarded corpus-vocabulary correction.",
                "The combined `apostrophe_mixed_script` class runs only when explicitly requested.",
            ]
        )
    if metadata.get("language_fixture"):
        excluded = metadata.get("language_baseline_excluded_ids", [])
        if not isinstance(excluded, list | tuple):
            raise TypeError("language_baseline_excluded_ids metadata must be a sequence")
        excluded_text = ", ".join(f"`{str(item_id)}`" for item_id in excluded) or "none"
        notes.extend(
            [
                "Language questions come from the committed fixture at "
                f"`{metadata['language_fixture']}` and remain "
                f"`{metadata['language_fixture_status']}`.",
                "`translate_to_uk` is a benchmark-only exact paired retrieval upper bound: it",
                "replaces the drafted query with its source Ukrainian question for retrieval,",
                "while generation still receives the Russian or code-switched question. It is",
                "not a shipped translator.",
                f"Non-Ukrainian questions excluded from the paired baseline: {excluded_text}.",
            ]
        )
    return notes


def _uncertainty_section(result: RobustnessResult) -> list[str]:
    """Every signed delta on the scale that decides its three-state paired reading."""
    lines = [
        "",
        "## Paired uncertainty by noise class",
        "",
        "Each delta is paired by item. `rand p` is the calibrated sign-flip probability in the",
        "observed direction and drives the reading; `p_positive` is the diagnostic share of",
        "bootstrap resamples where the first named lane is above its reference. A reading is",
        "`borderline` when 90%, the reporting level, and 97.5% do not agree. Directional claims",
        "also require enough differing items for the exact sign test to reach the stated level.",
        "`*_delta` compares the named mitigation lane with clean; `*_recovery` compares it with",
        "the unmitigated lane of the same noise class.",
        "",
        "| Class | Mitigation | Scope | Comparison | Delta | Reading | rand p | p_positive "
        "| Settled? |",
        "| --- | --- | --- | --- | ---: | :-: | ---: | ---: | :-: |",
    ]
    for lane in result.lanes:
        lines.extend(
            _uncertainty_rows(lane.variant_class, lane.mitigation, "all", lane.comparisons)
        )
    return lines


def _uncertainty_rows(
    variant_class: str,
    mitigation: str,
    scope: str,
    comparisons: Mapping[str, PairedComparison],
) -> list[str]:
    lines: list[str] = []
    for name, comparison in comparisons.items():
        delta = comparison["delta"]
        stability = comparison.get("stability")
        reading = (
            format_reading(stability, stability["reading"])
            if stability is not None
            else "unannotated"
        )
        p_positive = f"{stability['p_positive']:.3f}" if stability is not None else "-"
        randomization_p = (
            f"{comparison['randomization_p']:.4f}" if "randomization_p" in comparison else "-"
        )
        settled = ("no" if stability["borderline"] else "yes") if stability is not None else "-"
        lines.append(
            f"| {variant_class} | `{mitigation}` | {scope} | `{name}` "
            f"| {format_interval(delta)} | {reading} | {randomization_p} "
            f"| {p_positive} | {settled} |"
        )
    return lines


def _affected_section(result: RobustnessResult) -> list[str]:
    """Repeat every lane over the items its class actually perturbed, when some were untouched."""
    diluted = [lane for lane in result.lanes if lane.changed.n < lane.n]
    if not diluted:
        return []
    untouched = {
        lane.variant_class: lane.n - lane.changed.n for lane in diluted if lane.mitigation == "off"
    }
    lines = [
        "",
        "## Affected items only",
        "",
        "A single-mechanism class cannot perturb a question that carries none of its trigger",
        "characters, and those untouched items pull every pooled delta above toward zero. The rows",
        "below repeat each lane over the perturbed items only, against the SAME items' clean",
        "baseline. Untouched items per class: "
        + ", ".join(f"`{name}` {count}" for name, count in sorted(untouched.items()))
        + ".",
        "",
        "| Class | Mitigation | Changed N | Objective | Obj delta | Recall | Recall delta | MRR | MRR delta | Obj recovery | Recall recovery | MRR recovery |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane in result.lanes:
        changed = lane.changed
        if not changed.n:
            # a class that perturbed nothing measured nothing; zeros here would read as a result
            lines.append(f"| {lane.variant_class} | `{lane.mitigation}` | 0 |" + " - |" * 9)
            continue
        lines.append(
            f"| {lane.variant_class} | `{lane.mitigation}` | {changed.n} | "
            f"{changed.objective_score:.4f} | {changed.objective_delta:+.4f} | "
            f"{changed.recall_at_k:.4f} | {changed.recall_delta:+.4f} | "
            f"{changed.mrr:.4f} | {changed.mrr_delta:+.4f} | "
            f"{changed.objective_recovery:+.4f} | {changed.recall_recovery:+.4f} | "
            f"{changed.mrr_recovery:+.4f} |"
        )
    lines.extend(
        [
            "",
            "### Affected-subset paired uncertainty",
            "",
            "These are the same paired readings restricted to questions the generator changed.",
            "",
            "| Class | Mitigation | Scope | Comparison | Delta | Reading | rand p | p_positive "
            "| Settled? |",
            "| --- | --- | --- | --- | ---: | :-: | ---: | ---: | :-: |",
        ]
    )
    for lane in result.lanes:
        if not lane.changed.n:
            continue
        lines.extend(
            _uncertainty_rows(
                lane.variant_class,
                lane.mitigation,
                "changed",
                lane.changed.comparisons,
            )
        )
    return lines


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
