"""The maintained ASCII evidence artifact for the answer-validation comparison.

The report's job is to make the two readings a validator can be wrong about impossible to confuse:
what it STOPPED and what it REFUSED. So the catch and false-rejection columns sit beside each
other per axiom class, the objective delta states the item set it was read on, and the abstention
rate and answered count sit beside the objective -- a lane that improves the mean by answering
fewer questions has to look like one.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from llb.eval.answer_validation.labelling import SIGNAL_CONTAINS
from llb.eval.answer_validation.verdict import DECISION_ADOPT
from llb.rag.fusion_evidence.paired import format_randomization_p

# How much of a refused answer the table quotes. Enough to adjudicate a short factoid answer by
# eye; the bundle keeps the full text.
REFUSAL_PREVIEW_CHARS = 90

LANE_HEADER = (
    "| lane | n | answered | abstained | ontology_violation | objective | found | "
    "completion tokens | latency s | schema repair | semantic repair |"
)
LANE_RULE = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
CLASS_HEADER = (
    "| axiom class | rejected | caught | wrongly refused | under `contains` alone | catch rate "
    "| false-rejection rate | net delta | p | decision |"
)
CLASS_RULE = "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |"


def format_report(report: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> str:
    """Render the whole comparison: lanes, paired readings, per-class verdicts, and the reading."""
    lines = ["# Answer validation: off vs pydantic vs pydantic+ontology", ""]
    lines += _metadata_lines(metadata or {}, report)
    lines += ["", "## Lanes", "", LANE_HEADER, LANE_RULE]
    lines += [_lane_row(label, summary) for label, summary in report["lanes"].items()]
    lines += ["", "## Paired readings on the commonly-answered items", ""]
    lines += _reading_lines(report)
    lines += ["", "## Per-axiom-class adopt-or-reject", ""]
    lines += _class_lines(report["axiom_classes"])
    lines += ["", "## Every refused answer", ""]
    lines += _refusal_lines(report.get("refusals", []), report.get("relabelled", []))
    lines += ["", "## Reading", ""] + _reading_notes(report)
    return "\n".join(lines) + "\n"


def _metadata_lines(metadata: Mapping[str, Any], report: Mapping[str, Any]) -> list[str]:
    rows = [f"- {key}: {value}" for key, value in sorted(metadata.items())]
    rows.append(
        f"- items scored: {report['n_items']}; commonly answered by every lane: "
        f"{report['n_commonly_answered']}"
    )
    settings = report["settings"]
    rows.append(
        f"- paired bootstrap: {settings['resamples']} resamples, "
        f"{settings['confidence']:.2f} confidence, seed {settings['seed']}"
    )
    return rows


def _lane_row(label: str, summary: Mapping[str, Any]) -> str:
    return (
        f"| `{label}` | {summary['n']} | {summary['n_answered']} "
        f"| {summary['abstention_rate']:.3f} | {summary['ontology_violation_rate']:.3f} "
        f"| {summary['objective_score']:.3f} | {summary['contains']:.3f} "
        f"| {summary['completion_tokens']:.1f} | {summary['latency_s']:.2f} "
        f"| {summary['envelope_repair_rate']:.3f} | {summary['validation_repair_rate']:.3f} |"
    )


def _reading_lines(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "| lane | objective delta [lo, hi] | p | found delta | added tokens/answer "
        "| added s/answer | decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for reading in report["readings"]:
        delta = reading["objective_delta"]["delta"]
        found = reading["contains_delta"]["delta"]
        lines.append(
            f"| `{reading['lane']}` | {delta['mean']:+.3f} [{delta['lo']:+.3f}, "
            f"{delta['hi']:+.3f}] | {format_randomization_p(reading['objective_delta'])} "
            f"| {found['mean']:+.3f} | {reading['added_completion_tokens']:+.1f} "
            f"| {reading['added_latency_s']:+.3f} | {reading['decision']} |"
        )
    lines += [""] + [f"- {reading['reason']}" for reading in report["readings"]]
    return lines


def _class_lines(verdicts: Sequence[Mapping[str, Any]]) -> list[str]:
    if not verdicts:
        return [
            "No axiom class refused an answer on this item set, so no class is adopted: absence "
            "of a rejection is absence of evidence, not a clean bill."
        ]
    lines = [CLASS_HEADER, CLASS_RULE]
    for verdict in verdicts:
        net = verdict["net"]["delta"]
        lines.append(
            f"| `{verdict['axiom_class']}` | {verdict['n_rejected']} | {verdict['n_catches']} "
            f"| {verdict['n_false_rejections']} "
            f"| {verdict['n_catches_contains']} caught / "
            f"{verdict['n_false_rejections_contains']} refused "
            f"| {verdict['catch_rate']:.3f} "
            f"| {verdict['false_rejection_rate']:.3f} | {net['mean']:+.3f} "
            f"| {format_randomization_p(verdict['net'])} | {verdict['decision']} |"
        )
    lines += [""] + [f"- {verdict['reason']}" for verdict in verdicts]
    lines.append(
        "- ``caught` / `wrongly refused`` are the inflection-tolerant reading and decide the "
        "verdict; ``under `contains` alone`` is what the shipped surface-token proxy gives on the "
        "same rejections, so the two readings of this run are comparable rather than one "
        "replacing the other."
    )
    return lines


def _refusal_lines(
    refused: Sequence[Mapping[str, Any]], relabelled: Sequence[str] = ()
) -> list[str]:
    """List every rejection, not just the counts: a proxy label is a claim a reader may check."""
    if not refused:
        return ["The gate refused nothing on this item set."]
    lines = [
        "| item | axioms | labelled | by | `contains` said | semantic | objective | repaired "
        "| tokens | reference | answer |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in refused:
        signals = [name for name in row["label_signals"] if name != SIGNAL_CONTAINS]
        lines.append(
            f"| `{row['item_id']}` | {', '.join(row['axiom_ids'])} | {row['labelled']} "
            f"| {', '.join(signals) if signals else '-'} | {row['shipped_label']} "
            f"| {_semantic(row['semantic'])} | {row['objective_score']:.2f} "
            f"| {'yes' if row['repaired'] else 'no'} | {row['completion_tokens']} "
            f"| {_quote(row.get('reference', ''))} | {_quote(row['answer_preview'])} |"
        )
    lines += [
        "",
        "`labelled` is the inflection-tolerant reading; `by` names the added signal that fired "
        "(blank when plain `contains` already said the answer was right), and "
        "``contains` said` is the reading the shipped surface-token proxy alone gives, so the two "
        "sit side by side rather than one silently replacing the other.",
        "`semantic` is the `--score-semantic` cosine where the run recorded one. It is REPORTED "
        "and never thresholded: no cut on this corpus separates a paraphrase from a different "
        "answer, and a label nobody can reproduce without the pinned embedder is a worse proxy "
        "than a deterministic one.",
        "None of it is a verdict a reader owes agreement with -- the reference and the answer are "
        "in the row so a refusal can be adjudicated by eye.",
    ]
    if relabelled:
        lines += [
            "",
            f"Re-labelled against the surface-token reading: {', '.join(relabelled)}.",
        ]
    return lines


def _quote(text: str) -> str:
    """One cell of free text: capped, and stripped of what would break the table."""
    return text[:REFUSAL_PREVIEW_CHARS].replace("|", "/").replace("\n", " ")


def _semantic(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _reading_notes(report: Mapping[str, Any]) -> list[str]:
    adopted = [v["axiom_class"] for v in report["axiom_classes"] if v["decision"] == DECISION_ADOPT]
    return [
        "- `objective` and `found` are means over EVERY case a lane scored; the paired deltas "
        "above are read only on the items every lane ended `ok` on, so a gate cannot buy a higher "
        "mean by declining the hard items.",
        "- `catch rate` and `false-rejection rate` are both per case of the gated lane, so they "
        "are directly comparable: a class whose false-rejection rate meets or exceeds its catch "
        "rate is refusing as much correct work as it stops.",
        "- a rejection counts as a FALSE rejection when a correctness signal scores the refused "
        "answer correct: `contains` (the found-rate signal rather than the token-F1 objective, so "
        "a verbose but correct answer is not priced as a wrong one), the same containment over "
        "LEMMAS, or the terse reading in which every content lemma the answer states is in the "
        "reference. The first has no morphology and Ukrainian references are inflected, which is "
        "the failure the other two exist to cover.",
        "- `added tokens/answer` and `added s/answer` are the gate's cost, the repair round trip "
        "included, on the same commonly-answered items.",
        f"- axiom classes adopted by this run: {', '.join(adopted) if adopted else 'none'}.",
    ]
