"""Everything that fits or scores a linkage MODEL: block counts, training, accuracy.

Split from `engine.py` along the obvious seam -- this half decides what the model is, that half
decides what a record table becomes under it.
"""

import copy
from typing import Any

from llb.core.contracts.common import JsonObject
from llb.linkage.comparisons import build_blocking_rule, require_splink
from llb.linkage.constants import ACCURACY_WEIGHT_ROUNDING, RECORDS_TABLE
from llb.linkage.model import AccuracyPoint, BlockingCount, MatchParameter
from llb.linkage.records import LabelTables
from llb.linkage.spec import LinkageSpec


def count_blocking_comparisons(
    con: Any, spec: LinkageSpec, table: str = RECORDS_TABLE
) -> tuple[BlockingCount, ...]:
    """How many comparisons each prediction rule generates, measured before any fitting."""
    require_splink()
    import splink.blocking_analysis as blocking_analysis
    from splink import DuckDBAPI

    counts: list[BlockingCount] = []
    for rule in spec.blocking_rules:
        try:
            report = blocking_analysis.count_comparisons_from_blocking_rule(
                table_or_tables=[table],
                blocking_rule=build_blocking_rule(rule),
                link_type="dedupe_only",
                db_api=DuckDBAPI(connection=con),
                unique_id_column_name=spec.unique_id_column,
            )
        except Exception as exc:
            # Splink wraps a DuckDB binder error, whose message names the missing column but not
            # the rule that wanted it. Broad on purpose: whatever went wrong, the operator needs
            # the rule and the columns beside it, and the original is chained.
            raise ValueError(
                f"blocking rule {rule.label!r} cannot be evaluated over the record table "
                f"(columns: {', '.join(_table_columns(con, table))}). A column used only for "
                "blocking must be listed in the specification's retain_columns.\n"
                f"  {exc}"
            ) from exc
        counts.append(
            BlockingCount(
                rule=rule.label,
                pre_filter=int(report["number_of_comparisons_generated_pre_filter_conditions"]),
                post_filter=int(
                    report["number_of_comparisons_to_be_scored_post_filter_conditions"]
                ),
            )
        )
    return tuple(counts)


def train(linker: Any, spec: LinkageSpec, labels: LabelTables | None) -> bool:
    """Fit u by random sampling, then m from reviewer labels if there are any, else by EM."""
    linker.training.estimate_u_using_random_sampling(max_pairs=spec.max_pairs, seed=spec.seed)
    if labels is not None:
        linker.training.estimate_m_from_pairwise_labels(labels.matches)
        return True
    for rule in spec.em_rules:
        linker.training.estimate_parameters_using_expectation_maximisation(
            build_blocking_rule(rule)
        )
    return False


def labelled_accuracy(linker: Any, labels_table: str) -> tuple[AccuracyPoint, ...]:
    """The precision/recall curve of the fitted model against a reviewer-labelled set.

    Splink's own output for this is an interactive chart; the seam keeps the numbers behind it and
    drops the chart, so an operating point is a value in an artifact rather than a reading off a
    picture nobody can diff.
    """
    table = linker.evaluation.accuracy_analysis_from_labels_table(
        labels_table,
        output_type="table",
        match_weight_round_to_nearest=ACCURACY_WEIGHT_ROUNDING,
        add_metrics=["f1"],
    )
    points = [
        AccuracyPoint(
            threshold=float(row["match_probability"]),
            true_positives=int(row["tp"]),
            false_positives=int(row["fp"]),
            true_negatives=int(row["tn"]),
            false_negatives=int(row["fn"]),
            precision=float(row["precision"]),
            recall=float(row["recall"]),
            f1=float(row["f1"]),
        )
        for row in table.as_record_dict()
    ]
    return tuple(sorted(points, key=lambda point: point.threshold))


def smoothed_model(model: JsonObject, floor: float) -> JsonObject:
    """The same trained model with every fitted m and u pulled into `[floor, 1 - floor]`.

    Expectation-maximisation has a degenerate boundary solution on a small table: a level no pair
    of the match class exhibits gets m driven to machine zero, and every pair that lands on such a
    level then scores at the SAME floor weight -- so the ranking below the top class becomes one
    tie, which is the part of it a reviewer reads. A pseudo-count floor says "nobody observed this
    level", not "this level cannot occur among matches".

    It moves the probability SCALE and not the order within the trained part of the model, and a
    level the fit left as `null` stays null: an estimate that was never made is not smoothed into
    one that was.
    """
    if floor <= 0:
        return dict(model)
    smoothed = copy.deepcopy(dict(model))
    for comparison in smoothed.get("comparisons", ()):
        for level in comparison.get("comparison_levels", ()):
            for key in ("m_probability", "u_probability"):
                value = level.get(key)
                if value is not None:
                    level[key] = min(max(float(value), floor), 1.0 - floor)
    return smoothed


def match_parameters(model: JsonObject) -> tuple[MatchParameter, ...]:
    """Read the fitted m/u of every comparison level out of a saved model."""
    parameters: list[MatchParameter] = []
    for comparison in model.get("comparisons", ()):
        name = str(comparison.get("output_column_name", ""))
        for level in comparison.get("comparison_levels", ()):
            parameters.append(
                MatchParameter(
                    comparison=name,
                    level=str(level.get("label_for_charts", "")),
                    m=_as_float(level.get("m_probability")),
                    u=_as_float(level.get("u_probability")),
                )
            )
    return tuple(parameters)


def _as_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _table_columns(con: Any, table: str) -> list[str]:
    return [str(row[0]) for row in con.execute(f'DESCRIBE "{table}"').fetchall()]
