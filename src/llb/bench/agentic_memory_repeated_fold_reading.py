"""Outcome readings for repeated-fold completion and its marker ablation."""

from typing import cast

READING_DECAYS = "completion_decays_with_fold_count"
READING_STABLE = "completion_is_stable_through_measured_folds"
READING_INSUFFICIENT = "insufficient_measured_fold_counts"
MECHANISM_MARKER = "typed_memory_marker_required"
MECHANISM_SUMMARY = "model_written_summary_sufficient"
MECHANISM_MIXED = "mixed_or_inconclusive_mechanism"


def completion_by_fold(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Group shipped-policy cases by what the model actually folded, not the declaration."""
    grouped: dict[int, list[bool]] = {}
    for row in rows:
        if row["arm"] != "typed_marker":
            continue
        for case in cast(list[dict[str, object]], row["cases"]):
            grouped.setdefault(int(cast(int, case["measured_folds"])), []).append(
                bool(case["success"])
            )
    return [
        {
            "measured_folds": folds,
            "n_cases": len(success),
            "n_completed": sum(success),
            "completion": sum(success) / len(success),
        }
        for folds, success in sorted(grouped.items())
    ]


def completion_cost_reading(
    rows: list[dict[str, object]],
) -> tuple[str, str, int | None]:
    """State whether a measured higher-fold group falls below a viable one-fold control."""
    by_fold = {int(cast(int, row["measured_folds"])): row for row in rows}
    if 1 not in by_fold or len(by_fold) < 2:
        return READING_INSUFFICIENT, "the run did not produce one and higher measured folds", None
    one_fold = float(cast(float, by_fold[1]["completion"]))
    if one_fold <= 0.0:
        return (
            READING_INSUFFICIENT,
            "the measured one-fold group completed nothing, so it cannot anchor a decay reading",
            None,
        )
    lower = [
        fold
        for fold, row in by_fold.items()
        if fold > 1 and float(cast(float, row["completion"])) < one_fold
    ]
    matching = [
        fold for fold, row in by_fold.items() if float(cast(float, row["completion"])) >= one_fold
    ]
    if lower:
        first = min(lower)
        return (
            READING_DECAYS,
            f"completion first falls below the one-fold rate at {first} measured folds",
            max((fold for fold in matching if fold < first), default=1),
        )
    limit = max(by_fold)
    return (
        READING_STABLE,
        f"no measured fold group through {limit} folds falls below the one-fold completion rate",
        limit,
    )


def mechanism_reading(rows: list[dict[str, object]]) -> tuple[str, str]:
    """Pair marker preservation against the summary-only ablation on identical cases."""
    paired: dict[tuple[str, str], dict[str, bool]] = {}
    for row in rows:
        for case in cast(list[dict[str, object]], row["cases"]):
            key = (cast(str, row["cell_id"]), cast(str, case["item_id"]))
            paired.setdefault(key, {})[cast(str, row["arm"])] = bool(case["success"])
    complete_pairs = [
        pair for pair in paired.values() if "typed_marker" in pair and "model_summary_only" in pair
    ]
    marker_wins = sum(
        pair["typed_marker"] and not pair["model_summary_only"] for pair in complete_pairs
    )
    summary_wins = sum(
        pair["model_summary_only"] and not pair["typed_marker"] for pair in complete_pairs
    )
    typed_completed = sum(pair["typed_marker"] for pair in complete_pairs)
    if marker_wins and not summary_wins:
        return (
            MECHANISM_MARKER,
            f"typed-marker preservation wins {marker_wins} paired cases and loses none",
        )
    if not marker_wins and not summary_wins and typed_completed:
        return (
            MECHANISM_SUMMARY,
            "removing typed-marker preservation changes no paired completion, so the model-written "
            "summary is sufficient on every completed case",
        )
    return (
        MECHANISM_MIXED,
        f"paired mechanism outcomes are marker wins={marker_wins}, summary-only wins={summary_wins}",
    )
