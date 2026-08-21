"""Orchestration for the third-generation conflict-null research matrix.

Five questions, answered in the order that decides whether the later ones are worth asking:

1. can the affordable operating point be resolved by ANY control bank of the size that exists?
2. does cross-fitted propensity balancing make the control bank exchangeable with the corpus?
3. is the null component identifiable from the observed cosine distribution at all?
4. does a higher-capacity geometry remove the residual corpus shift without losing relations?
5. what does the operator actually get -- measured claim-tier precision with a clustered bound?

The lanes share one control bank and one rank baseline so their verdicts are directly comparable.
"""

from dataclasses import replace

import numpy as np

from llb.conflicts.null_research.generations.advanced import candidate_gates
from llb.conflicts.null_research.controls.balance import (
    BalancedControls,
    balanced_diagnostics,
    balanced_tail_payload,
    build_balanced_controls,
)
from llb.conflicts.null_research.controls.counterfactuals import counterfactual_traces
from llb.conflicts.null_research.evaluation import (
    FIXTURE_POSITIVE_DOC_PAIRS,
    fixture_metrics,
    transfer_payload,
)
from llb.conflicts.null_research.statistics.feasibility import operating_point_feasibility
from llb.conflicts.null_research.controls.geometries import (
    GEOMETRY_METHODS,
    baseline_pair_scores,
    build_geometry_space,
    geometry_candidate,
)
from llb.conflicts.null_research.geometry import CorpusGeometry
from llb.conflicts.null_research.statistics.mixture import (
    RelatedAnchors,
    lexical_related_anchors,
    mixture_identifiability,
)
from llb.conflicts.null_research.statistics.precision import claim_precision_lane
from llb.conflicts.null_research.controls.roles import verify_control_roles
from llb.core.contracts.common import JsonObject
from llb.prep.frontier.telemetry import LLMComplete

THIRD_RESEARCH_METHODS = ("balanced_transport_control", *GEOMETRY_METHODS)


def _balanced_candidate(
    controls: dict[str, BalancedControls],
    corpora: dict[str, CorpusGeometry],
    rank: JsonObject,
    feasibility: JsonObject,
    *,
    fpr: float,
    transfer_threshold: float,
    max_goods_candidates: int,
    seed: int,
) -> JsonObject:
    thresholds = {dataset: control.weighted_threshold(fpr) for dataset, control in controls.items()}
    tails = {
        dataset: balanced_tail_payload(control, thresholds[dataset], fpr, seed + position)
        for position, (dataset, control) in enumerate(controls.items())
    }
    diagnostics = {
        dataset: balanced_diagnostics(control, corpora[dataset].observed_similarities)
        for dataset, control in controls.items()
    }
    fixture = fixture_metrics(
        corpora["fixture"].document_maxima, thresholds["fixture"], FIXTURE_POSITIVE_DOC_PAIRS
    )
    hr = transfer_payload(corpora["hr"].observed_similarities, thresholds["hr"], transfer_threshold)
    goods = transfer_payload(
        corpora["goods"].observed_similarities, thresholds["goods"], transfer_threshold
    )
    gates = candidate_gates(
        fixture,
        rank,
        hr,
        goods,
        tails,
        diagnostics,
        max_goods_candidates=max_goods_candidates,
        eligible=True,
        control_key="exchangeable",
        extra={
            "two_way_uncertainty_resolved": all(
                bool(tail["two_way_uncertainty_resolved"]) for tail in tails.values()
            ),
            "operating_point_feasible": bool(feasibility["feasible"]),
        },
    )
    return {
        "method": "balanced_transport_control",
        "thresholds": {key: round(value, 6) for key, value in thresholds.items()},
        "null_tails": tails,
        "diagnostics": diagnostics,
        "fixture": fixture,
        "hr": hr,
        "goods": goods,
        "gates": gates,
    }


def _geometry_candidates(
    controls: dict[str, BalancedControls],
    corpora: dict[str, CorpusGeometry],
    related_pairs: dict[str, list[tuple[int, int]]],
    rank: JsonObject,
    *,
    fpr: float,
    transfer_threshold: float,
    max_goods_candidates: int,
    seed: int,
) -> list[JsonObject]:
    baselines = {dataset: baseline_pair_scores(corpus) for dataset, corpus in corpora.items()}
    weights = {dataset: control.weights for dataset, control in controls.items()}
    methods: list[JsonObject] = []
    for offset, method in enumerate(GEOMETRY_METHODS):
        spaces = {
            dataset: build_geometry_space(corpus, controls[dataset], related_pairs[dataset], method)
            for dataset, corpus in corpora.items()
        }
        variants = {
            dataset: replace(controls[dataset], scores=spaces[dataset].control_scores)
            for dataset in corpora
        }
        thresholds = {
            dataset: control.weighted_threshold(fpr) for dataset, control in variants.items()
        }
        tails = {
            dataset: balanced_tail_payload(
                control,
                thresholds[dataset],
                fpr,
                seed + (offset + 1) * len(corpora) + position,
            )
            for position, (dataset, control) in enumerate(variants.items())
        }
        methods.append(
            geometry_candidate(
                method,
                spaces,
                baselines,
                tails,
                thresholds,
                rank,
                weights,
                transfer_threshold=transfer_threshold,
                max_goods_candidates=max_goods_candidates,
            )
        )
    return methods


def _identifiability(
    corpora: dict[str, CorpusGeometry],
    controls: dict[str, BalancedControls],
    anchors: dict[str, RelatedAnchors],
    feasibility: JsonObject,
    *,
    candidate_cap: int,
) -> dict[str, JsonObject]:
    datasets = feasibility["datasets"]
    assert isinstance(datasets, dict)
    return {
        dataset: mixture_identifiability(
            corpus,
            controls[dataset].scores.reshape(-1),
            np.tile(controls[dataset].weights, controls[dataset].scores.shape[0]),
            anchors[dataset],
            operating_alpha=float(datasets[dataset]["operating_tail_alpha"]),
            candidate_cap=candidate_cap,
            effective_units=controls[dataset].effective_units,
        )
        for dataset, corpus in corpora.items()
    }


def run_third_generation_candidates(
    corpora: dict[str, CorpusGeometry],
    references: dict[str, CorpusGeometry],
    rank: JsonObject,
    complete: LLMComplete,
    *,
    fpr: float,
    transfer_threshold: float,
    max_goods_candidates: int,
    adjudication_budget: int,
    role_samples_per_type: int,
    seed: int,
) -> JsonObject:
    """Run every third-generation lane over one shared, propensity-balanced control bank."""
    controls = {
        dataset: build_balanced_controls(corpus, references) for dataset, corpus in corpora.items()
    }
    feasibility = operating_point_feasibility(
        corpora,
        {dataset: control.effective_units for dataset, control in controls.items()},
        candidate_cap=max_goods_candidates,
        nominal_fpr=fpr,
    )
    anchors = {dataset: lexical_related_anchors(corpus) for dataset, corpus in corpora.items()}
    related_pairs = {dataset: anchor.pairs for dataset, anchor in anchors.items()}
    methods = [
        _balanced_candidate(
            controls,
            corpora,
            rank,
            feasibility,
            fpr=fpr,
            transfer_threshold=transfer_threshold,
            max_goods_candidates=max_goods_candidates,
            seed=seed,
        ),
        *_geometry_candidates(
            controls,
            corpora,
            related_pairs,
            rank,
            fpr=fpr,
            transfer_threshold=transfer_threshold,
            max_goods_candidates=max_goods_candidates,
            seed=seed,
        ),
    ]
    traces = [trace for corpus in corpora.values() for trace in counterfactual_traces(corpus)]
    precision = claim_precision_lane(
        corpora,
        complete,
        adjudication_budget=adjudication_budget,
        operating_budget=max_goods_candidates,
        seed=seed,
    )
    roles = verify_control_roles(traces, corpora, complete, per_type=role_samples_per_type)
    assert [str(method["method"]) for method in methods] == list(THIRD_RESEARCH_METHODS)
    return {
        "feasibility": feasibility,
        "methods": methods,
        "identifiability": _identifiability(
            corpora, controls, anchors, feasibility, candidate_cap=max_goods_candidates
        ),
        "claim_precision": precision,
        "control_role_verification": roles,
        "control_traces": traces,
    }
