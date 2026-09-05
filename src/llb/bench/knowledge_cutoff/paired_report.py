"""Paired English/Ukrainian cutoff statistics and report rendering."""

import json
import random
from collections import defaultdict
from typing import Any

BOOTSTRAP_SAMPLES = 2000
CONFIDENCE_LEVEL = 0.95


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = round(probability * (len(ordered) - 1))
    return ordered[index]


def paired_statistics(
    english: list[dict[str, Any]], ukrainian: list[dict[str, Any]], *, seed: int
) -> dict[str, object]:
    uk_by_id = {str(row["item_id"]): row for row in ukrainian}
    pairs: list[tuple[str, str, float]] = []
    months: dict[str, list[float]] = defaultdict(list)
    for row in english:
        item_id = str(row["item_id"])
        other = uk_by_id.get(item_id)
        if other is None:
            raise ValueError(f"Ukrainian lane is missing {item_id}")
        if row["choice_order"] != other["choice_order"] or row["expected"] != other["expected"]:
            raise ValueError(f"{item_id}: language lanes used different source-choice mappings")
        if not row["counts_for_curve"]:
            continue
        delta = float(other["objective_score"]) - float(row["objective_score"])
        pairs.append((item_id, str(row["month"]), delta))
        months[str(row["month"])].append(delta)
    if not pairs:
        raise ValueError("paired report needs at least one cutoff-eligible accepted event")
    observed = sum(delta for _item, _month, delta in pairs) / len(pairs)
    rng = random.Random(seed)
    bootstrapped = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [pairs[rng.randrange(len(pairs))][2] for _ in pairs]
        bootstrapped.append(sum(sample) / len(sample))
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return {
        "definition": "Ukrainian accuracy minus English accuracy on identical accepted events",
        "n_pairs": len(pairs),
        "accuracy_delta": observed,
        "bootstrap": {
            "seed": seed,
            "samples": BOOTSTRAP_SAMPLES,
            "confidence_level": CONFIDENCE_LEVEL,
            "low": _percentile(bootstrapped, tail),
            "high": _percentile(bootstrapped, 1.0 - tail),
        },
        "monthly": [
            {"month": month, "n": len(values), "accuracy_delta": sum(values) / len(values)}
            for month, values in sorted(months.items())
        ],
    }


def render_paired_markdown(report: dict[str, Any]) -> str:
    paired = report["paired"]
    interval = paired["bootstrap"]
    lines = [
        "# Bilingual Knowledge Cutoff Report",
        "",
        f"- Model: `{report['model']}`",
        f"- Backend: `{report['backend']}`",
        f"- Reviewed dataset revision: `{report['review']['resolved_revision']}`",
        f"- Accepted paired events: {paired['n_pairs']}",
        f"- English effective cutoff: `{report['english']['decay_fit']['effective_cutoff']}`",
        f"- Ukrainian effective cutoff: `{report['ukrainian']['decay_fit']['effective_cutoff']}`",
        f"- Accuracy delta (Ukrainian - English): {paired['accuracy_delta']:.3f}",
        f"- Seeded 95% paired bootstrap interval: [{interval['low']:.3f}, {interval['high']:.3f}]",
        "",
        "## Monthly Language Deltas",
        "",
        "| Month | Paired N | Accuracy delta (uk - en) |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['month']} | {row['n']} | {row['accuracy_delta']:.3f} |"
        for row in paired["monthly"]
    )
    lines.extend(
        [
            "",
            "Only accepted, source-aligned translation rows enter either lane. The paired "
            "difference isolates language sensitivity more directly than comparing unrelated "
            "event sets, but it does not prove why a model answered differently.",
            "",
        ]
    )
    return "\n".join(lines)


def paired_artifacts(report: dict[str, object]) -> dict[str, str]:
    return {
        "report.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        "report.md": render_paired_markdown(report),
    }
