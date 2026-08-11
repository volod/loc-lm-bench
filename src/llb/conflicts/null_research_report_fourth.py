"""Fourth-generation report sections: control synthesis yield/scale and conformal coverage.

Both answer questions the acceptance matrix cannot show. The synthesis table says whether a
generated control bank is a null at all and what it costs per verified claim -- the number that
turns "generate more" into a wall-clock statement. The conformal table says which tail estimator
holds its promise, and on how many independent units.
"""

from llb.core.contracts.common import JsonObject


def synthesis_section(summary: JsonObject) -> list[str]:
    synthesis = summary.get("control_synthesis")
    if not isinstance(synthesis, dict):
        return []
    rows = [
        "## In-support control synthesis",
        "",
        "Control claims written from each corpus's own structure, then adjudicated against the "
        "source they were written from. Only a claim the verifier calls non-conflicting is kept.",
        "",
        "| dataset | sampled | generated | conflicting | retained | yield | s/claim | "
        "units required | years to reach |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, payload in synthesis.items():
        if not isinstance(payload, dict):
            continue
        scale = payload["scale"]
        assert isinstance(scale, dict)
        rows.append(
            f"| {name} | {payload['sampled_sources']} | {payload['generated_claims']} | "
            f"{payload['conflicting_claims']} | {payload['retained_claims']} | "
            f"{payload['verified_yield']} | {scale['seconds_per_retained_claim']} | "
            f"{scale['required_independent_units']} | {scale['years_to_required_units']} |"
        )
    return [*rows, ""]


def certification_section(summary: JsonObject) -> list[str]:
    certification = summary.get("tail_certification")
    if not isinstance(certification, dict):
        return []
    datasets = certification["datasets"]
    assert isinstance(datasets, dict)
    rows = [
        "## Distribution-free tail certification",
        "",
        f"Independent units a group-split conformal threshold needs to certify each corpus's "
        f"affordable tail at confidence {certification['confidence']} -- a floor that holds "
        "whatever the scores look like and whichever estimator reads them.",
        "",
        "| dataset | operating alpha | units required | verified units | short by | certifiable |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, payload in datasets.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            f"| {name} | {payload['operating_tail_alpha']} | "
            f"{payload['certifiable_units_required']} | {payload['verified_units_available']} | "
            f"{payload['unit_deficit_factor']}x | "
            f"{'yes' if payload['certifiable'] else 'no'} |"
        )
    return [*rows, ""]


def conformal_section(summary: JsonObject) -> list[str]:
    conformal = summary.get("conformal")
    if not isinstance(conformal, dict):
        return []
    scenarios = conformal["scenarios"]
    assert isinstance(scenarios, list)
    gates = conformal["gates"]
    assert isinstance(gates, dict)
    rows = [
        "## Group-split conformal tail inference",
        "",
        f"Simulated over {conformal['replications']} replications per grid point, "
        f"{conformal['rows_per_unit']} rows per independent unit. Both methods publish an upper "
        f"bound for the tail rate on a fresh population; a method holds when its bound is right at "
        f"least {conformal['min_coverage_probability']} of the time. A conformal claim rate below "
        "one is a refusal to certify, not a miss.",
        "",
        "| scenario | alpha | units | rank | conformal claims | conformal bound ok | "
        "row bootstrap bound ok |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        for point in scenario["grid"]:
            rows.append(
                f"| {scenario['scenario']} | {scenario['nominal_alpha']} | "
                f"{point['independent_units']} | {point['conformal_rank']} | "
                f"{point['conformal_claim_rate']} | {point['conformal_bound_coverage']} | "
                f"{point['row_bootstrap_bound_coverage']} |"
            )
    rows.extend(
        [
            "",
            "- units the conformal claim needs (simulated / distribution-free): "
            + ", ".join(
                f"{scenario['scenario']}={scenario['conformal_units_required']}"
                f"/{scenario['distribution_free_units']}"
                for scenario in scenarios
                if isinstance(scenario, dict)
            ),
            f"- accepted: {'yes' if gates['accepted'] else 'no'}; failed gates: "
            + (
                ", ".join(
                    name for name, passed in gates.items() if name != "accepted" and not passed
                )
                or "none"
            ),
        ]
    )
    return [*rows, ""]


def _cost_lines(synthesis: dict[str, object]) -> list[str]:
    lines: list[str] = []
    for name, payload in synthesis.items():
        if not isinstance(payload, dict):
            continue
        scale = payload["scale"]
        assert isinstance(scale, dict)
        lines.append(
            f"- `{name}`: {payload['retained_claims']} of {payload['sampled_sources']} generated "
            f"claims cleared the verifier at {scale['seconds_per_retained_claim']}s each, so the "
            f"{scale['required_independent_units']} units its affordable tail needs cost "
            f"{scale['years_to_required_units']} GPU-years on this host"
        )
    return lines


def _membership_lines(methods: list[object]) -> list[str]:
    lines: list[str] = []
    for method in methods:
        if not isinstance(method, dict):
            continue
        diagnostics = method.get("diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        membership = ", ".join(
            f"{dataset}={payload['membership_auc']}"
            for dataset, payload in diagnostics.items()
            if isinstance(payload, dict) and "membership_auc" in payload
        )
        if membership:
            lines.append(f"- `{method['method']}` weighted membership AUC: {membership}")
    return lines


def synthesis_evidence(summary: JsonObject) -> list[str]:
    """The generated-bank numbers that carry the fourth-generation verdict."""
    synthesis = summary.get("control_synthesis")
    methods = summary.get("methods")
    if not isinstance(synthesis, dict) or not isinstance(methods, list):
        return []
    return [*_cost_lines(synthesis), *_membership_lines(methods), ""]
