"""Machine-readable and Markdown report rendering."""

import json
from typing import Any


def build_report(
    *,
    model: str,
    backend: str,
    source: dict[str, object],
    summary: dict[str, object],
    fit: dict[str, object],
    n_events: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark": "knowledge-cutoff",
        "model": model,
        "backend": backend,
        "probe": "position-balanced-mcq",
        "n_events": n_events,
        "source": source,
        "summary": summary,
        "decay_fit": fit,
        "interpretation": {
            "primary_estimate": fit.get("effective_cutoff"),
            "definition": (
                "Month where the fitted monotone accuracy curve reaches the midpoint between "
                "its learned ceiling and the four-choice chance floor."
            ),
            "limitations": [
                "This is an effective benchmark estimate, not proof of the training-data cutoff.",
                "Four-choice questions permit chance success; the fit includes a 0.25 floor.",
                "Event selection, benchmark contamination, fine-tuning, and quantization can shift it.",
                "English question comprehension can confound results for language-specialized models.",
                "Use the monthly observations and controls when the fitted curve looks implausible.",
            ],
        },
    }


def _format_rate(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    fit = report["decay_fit"]
    controls = summary["controls"]
    source = report["source"]
    lines = [
        "# Knowledge Cutoff Report",
        "",
        f"- Model: `{report['model']}`",
        f"- Backend: `{report['backend']}`",
        f"- Evaluated events: {report['n_events']}",
        f"- Effective cutoff (Optuna decay fit): `{fit.get('effective_cutoff') or 'unavailable'}`",
        f"- Fit status: `{fit['status']}`",
        f"- Eligible MCQ accuracy: {_format_rate(summary['eligible_accuracy'])}",
        f"- Parse rate: {_format_rate(summary['parse_rate'])}",
        f"- Threshold last-above month: `{summary.get('last_above') or 'none'}`",
        f"- First sustained-below month: `{summary.get('first_sustained_below') or 'none'}`",
        "",
        "## Monthly Evidence",
        "",
        "| Month | N | Correct | Incorrect | Abstain | Accuracy | Chance-adjusted | Fitted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    predicted = {
        point["month"]: point["predicted_accuracy"] for point in fit.get("fitted_curve", [])
    }
    for point in summary["curve"]:
        lines.append(
            "| {month} | {n} | {correct} | {incorrect} | {abstain} | {accuracy:.3f} | "
            "{adjusted:.3f} | {fitted} |".format(
                **point,
                adjusted=point["chance_adjusted_accuracy"],
                fitted=_format_rate(predicted.get(point["month"])),
            )
        )
    lines.extend(
        [
            "",
            "## Controls",
            "",
            f"- Living-person accuracy: {_format_rate(controls['living_person_accuracy'])} "
            f"(N={controls['living_person_n']})",
            f"- Fake-event rejection rate: {_format_rate(controls['fake_event_rejection_rate'])} "
            f"(N={controls['fake_event_n']})",
            f"- Fake-event confabulation rate: "
            f"{_format_rate(controls['fake_event_confabulation_rate'])}",
            "",
            "## Provenance and Interpretation",
            "",
            f"Dataset source: `{source['identity']}` at revision `{source['resolved_revision']}` "
            f"({source['license']}).",
            "",
            str(report["interpretation"]["definition"]),
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["interpretation"]["limitations"])
    lines.extend(
        [
            "",
            "Method and dataset choice were inspired by Apoorv Saxena's `knowledge-cutoff` "
            "project. This implementation uses loc-lm-bench's local backend, Optuna, canonical "
            "artifact, and MLflow conventions; it does not reuse the upstream application code.",
            "",
        ]
    )
    return "\n".join(lines)


def report_artifacts(report: dict[str, object]) -> dict[str, str]:
    return {
        "report.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        "report.md": render_markdown(report),
    }
