"""Third-generation report sections: feasibility, identifiability, precision, control roles.

These lanes do not share the null-candidate table shape -- they answer whether an operating point
is reachable at all, how tightly the data pins it, what the operator measurably gets instead, and
whether a constructed control family is a null in the first place -- so each renders its own table.
"""

from llb.core.contracts.common import JsonObject


def feasibility_section(summary: JsonObject) -> list[str]:
    feasibility = summary.get("feasibility")
    if not isinstance(feasibility, dict):
        return []
    datasets = feasibility["datasets"]
    assert isinstance(datasets, dict)
    rows = [
        "## Operating-point feasibility",
        "",
        f"An affordable list of {feasibility['candidate_cap']} rows is the per-pair tail below; "
        f"resolving it needs {feasibility['min_tail_observations']} independent tail observations.",
        "",
        "| dataset | comparable pairs | operating alpha | units required | units available | "
        "rows at nominal FPR | feasible |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, payload in datasets.items():
        rows.append(
            f"| {name} | {payload['comparable_chunk_pairs']} | {payload['operating_tail_alpha']} | "
            f"{payload['required_independent_units']} | "
            f"{payload['available_independent_units']} | "
            f"{payload['rows_selected_at_nominal_fpr']} | "
            f"{'yes' if payload['feasible'] else 'no'} |"
        )
    return [*rows, ""]


def identifiability_section(summary: JsonObject) -> list[str]:
    identifiability = summary.get("identifiability")
    if not isinstance(identifiability, dict):
        return []
    rows = [
        "## Cosine-only identifiability",
        "",
        "Every mixture of the anchored null and the lexically proven related component that the "
        "corpus's independent units cannot distinguish, and the operating points each implies.",
        "",
        "| dataset | related anchors | equivalent mixtures | null shift range | implied rows | "
        "rows ratio | tight | usable |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, payload in identifiability.items():
        if not isinstance(payload, dict):
            continue
        anchor = payload["related_anchor"]
        shift = payload["null_shift_range"]
        selected = payload["implied_selected_rows_range"]
        assert isinstance(anchor, dict)
        rows.append(
            f"| {name} | {anchor['anchor_pairs']} | "
            f"{payload['observationally_equivalent_mixtures']}"
            f"/{payload['grid_mixtures']} | {shift[0]} to {shift[1]} | "
            f"{selected[0]} to {selected[1]} | {payload['implied_selected_rows_ratio']} | "
            f"{'yes' if payload['identified'] else 'no'} | "
            f"{'yes' if payload['operating_point_usable'] else 'no'} |"
        )
    return [*rows, ""]


def precision_section(summary: JsonObject) -> list[str]:
    precision = summary.get("claim_precision")
    if not isinstance(precision, dict):
        return []
    calibration = precision["adjudicator_calibration"]
    datasets = precision["datasets"]
    assert isinstance(calibration, dict) and isinstance(datasets, dict)
    gates = precision["gates"]
    assert isinstance(gates, dict)
    failed = [name for name, passed in gates.items() if name != "accepted" and not passed]
    rows = [
        "## Claim-tier precision",
        "",
        f"- adjudicator agreement with the planted fixture labels: {calibration['accuracy']} "
        f"(95% {calibration['accuracy_wilson_95']}, {calibration['labelled_rows']} rows)",
        f"- accepted: {'yes' if gates['accepted'] else 'no'}; "
        f"failed gates: {', '.join(failed) if failed else 'none'}",
        "",
        "| dataset | budget | actionable | precision | two-way clustered LCB | cosine range |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, payload in datasets.items():
        if not isinstance(payload, dict):
            continue
        for point in payload["precision_curve"]:
            rows.append(
                f"| {name} | {point['budget']} | {point['actionable_rows']} | "
                f"{point['precision']} | {point['two_way_clustered_lcb']} | "
                f"{payload['cosine_range']} |"
            )
    return [*rows, ""]


def role_section(summary: JsonObject) -> list[str]:
    roles = summary.get("control_role_verification")
    if not isinstance(roles, dict):
        return []
    by_type = roles["by_edit_type"]
    assert isinstance(by_type, dict)
    rows = [
        "## Verified control roles",
        "",
        "Constructed edits adjudicated against their own source passage. A control family is a "
        "null only when the verifier finds essentially no conflict relations in it.",
        "",
        "| edit type | verified | conflicting | fraction | 95% interval | eligible as null |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, payload in by_type.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            f"| {name} | {payload['verified_rows']} | {payload['conflicting_rows']} | "
            f"{payload['conflicting_fraction']} | {payload['conflicting_wilson_95']} | "
            f"{'yes' if payload['eligible_as_null'] else 'no'} |"
        )
    return [*rows, ""]


def decision_evidence(summary: JsonObject) -> list[str]:
    """The measured numbers that carry the verdict, when the third-generation lanes ran."""
    feasibility = summary.get("feasibility")
    precision = summary.get("claim_precision")
    if not isinstance(feasibility, dict):
        return []
    datasets = feasibility["datasets"]
    assert isinstance(datasets, dict)
    worst = max(datasets.items(), key=lambda item: int(item[1]["required_independent_units"]))
    lines = [
        f"- the affordable tail on `{worst[0]}` needs "
        f"{worst[1]['required_independent_units']} independent control units; the bank supplies "
        f"{worst[1]['available_independent_units']} "
        f"({worst[1]['unit_deficit_factor']}x short), so no control construction of this scale "
        "can certify it",
    ]
    if isinstance(precision, dict):
        operating = precision["operating_point"]
        assert isinstance(operating, dict)
        measured = ", ".join(
            f"{name}={point['precision']} (LCB {point['two_way_clustered_lcb']})"
            for name, point in operating.items()
            if isinstance(point, dict)
        )
        lines.append(
            f"- measured claim-tier precision at budget {precision['operating_budget']}: "
            f"{measured or 'not resolved'}"
        )
    return [*lines, ""]
