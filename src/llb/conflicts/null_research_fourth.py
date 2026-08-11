"""Orchestration for the fourth-generation conflict-null research matrix.

The third generation closed three doors at once: collected reference banks fail positivity, cosine
alone is unidentifiable at the resolution an operating point needs, and traced counterfactual edits
are planted positives. Exactly three directions survive that verdict, and this generation runs all
of them over one shared, verified, in-support control bank:

1. does a control bank GENERATED from the target's own structure land inside its covariate support?
2. does reading both passages together -- a cross-encoder -- separate relations where one cosine
   could not, with calibration and clustered coverage rather than fixture F1 to show for it?
3. does group-split conformal tail inference hold nominal coverage on fewer independent units than
   the two-way row bootstrap the earlier generations used?

The first two lanes share the generated bank, so their verdicts are directly comparable; the third
is a simulation and touches no corpus.
"""

from llb.conflicts.null_research_advanced import candidate_gates
from llb.conflicts.null_research_balance import (
    BalancedControls,
    balanced_diagnostics,
    balanced_tail_payload,
    build_balanced_controls,
)
from llb.conflicts.null_research_conformal import (
    DEFAULT_CONFIDENCE,
    certifiable_units,
    conformal_lane,
)
from llb.conflicts.null_research_cross_encoder import (
    CROSS_ENCODER_METHOD,
    cross_encoder_candidate,
    score_controls,
    score_shortlist,
)
from llb.conflicts.null_research_evaluation import (
    FIXTURE_POSITIVE_DOC_PAIRS,
    fixture_metrics,
    transfer_payload,
)
from llb.conflicts.null_research_feasibility import operating_point_feasibility
from llb.conflicts.null_research_geometry import CorpusGeometry, EmbedTexts
from llb.conflicts.null_research_precision import adjudicate_rows, top_candidate_rows
from llb.conflicts.null_research_synthesis import SynthesizedControl, synthesize_bank
from llb.core.contracts.common import JsonObject
from llb.prep.frontier_telemetry import LLMComplete
from llb.rag.rerank import RerankScorer

IN_SUPPORT_METHOD = "synthesized_in_support_control"
FOURTH_RESEARCH_METHODS = (IN_SUPPORT_METHOD, CROSS_ENCODER_METHOD)


def _in_support_candidate(
    controls: dict[str, BalancedControls],
    corpora: dict[str, CorpusGeometry],
    synthesis: dict[str, JsonObject],
    rank: JsonObject,
    feasibility: JsonObject,
    *,
    fpr: float,
    transfer_threshold: float,
    max_goods_candidates: int,
    seed: int,
) -> JsonObject:
    """The balanced lane again, with the reference banks replaced by the generated ones."""
    thresholds = {dataset: control.weighted_threshold(fpr) for dataset, control in controls.items()}
    tails = {
        dataset: balanced_tail_payload(control, thresholds[dataset], fpr, seed + position)
        for position, (dataset, control) in enumerate(controls.items())
    }
    diagnostics = {
        dataset: {
            **balanced_diagnostics(control, corpora[dataset].observed_similarities),
            "verified_yield": synthesis[dataset]["verified_yield"],
            "retained_claims": synthesis[dataset]["retained_claims"],
        }
        for dataset, control in controls.items()
    }
    fixture = fixture_metrics(
        corpora["fixture"].document_maxima, thresholds["fixture"], FIXTURE_POSITIVE_DOC_PAIRS
    )
    transfers = {
        dataset: transfer_payload(
            corpora[dataset].observed_similarities, thresholds[dataset], transfer_threshold
        )
        for dataset in ("hr", "goods")
    }
    gates = candidate_gates(
        fixture,
        rank,
        transfers["hr"],
        transfers["goods"],
        tails,
        diagnostics,
        max_goods_candidates=max_goods_candidates,
        eligible=all(bool(payload["yield_sufficient"]) for payload in synthesis.values()),
        control_key="exchangeable",
        extra={
            "two_way_uncertainty_resolved": all(
                bool(tail["two_way_uncertainty_resolved"]) for tail in tails.values()
            ),
            "operating_point_feasible": bool(feasibility["feasible"]),
        },
    )
    return {
        "method": IN_SUPPORT_METHOD,
        "thresholds": {key: round(value, 6) for key, value in thresholds.items()},
        "null_tails": tails,
        "diagnostics": diagnostics,
        "fixture": fixture,
        "hr": transfers["hr"],
        "goods": transfers["goods"],
        "gates": gates,
    }


def _cross_encoder_lane(
    corpora: dict[str, CorpusGeometry],
    retained: dict[str, list[SynthesizedControl]],
    complete: LLMComplete,
    scorer: RerankScorer,
    rank: JsonObject,
    feasibility: JsonObject,
    *,
    fpr: float,
    transfer_threshold: float,
    max_goods_candidates: int,
    cross_encoder_rows: int,
    seed: int,
) -> tuple[JsonObject, list[JsonObject]]:
    """Adjudicate the shortlist once, re-score it and the frozen controls, then gate the scorer."""
    scored = {}
    verdict_rows: list[JsonObject] = []
    for dataset, corpus in corpora.items():
        rows = top_candidate_rows(corpus, cross_encoder_rows)
        verdicts = adjudicate_rows(corpus, rows, complete)
        verdict_rows.extend(verdicts)
        scored[dataset] = score_shortlist(corpus, rows, verdicts, scorer)
    controls = {
        dataset: score_controls(retained[dataset], scorer)
        for dataset in corpora
        if retained.get(dataset)
    }
    units = {dataset: len(controls[dataset][1]) for dataset in controls}
    candidate = cross_encoder_candidate(
        corpora,
        {dataset: scored[dataset] for dataset in controls},
        controls,
        units,
        rank,
        feasibility,
        fpr=fpr,
        transfer_threshold=transfer_threshold,
        max_goods_candidates=max_goods_candidates,
        seed=seed,
    )
    return candidate, verdict_rows


def run_fourth_generation_candidates(
    corpora: dict[str, CorpusGeometry],
    rank: JsonObject,
    complete: LLMComplete,
    embed: EmbedTexts,
    scorer: RerankScorer,
    *,
    fpr: float,
    transfer_threshold: float,
    max_goods_candidates: int,
    synthesis_per_document: int,
    cross_encoder_rows: int,
    seed: int,
) -> JsonObject:
    """Generate the in-support bank, then run every lane the third generation left open."""
    requirement = operating_point_feasibility(
        corpora,
        {name: 0 for name in corpora},
        candidate_cap=max_goods_candidates,
        nominal_fpr=fpr,
    )
    requirements = requirement["datasets"]
    assert isinstance(requirements, dict)
    banks = {
        dataset: synthesize_bank(
            corpus,
            complete,
            embed,
            per_document=synthesis_per_document,
            required_units=int(requirements[dataset]["required_independent_units"]),
        )
        for dataset, corpus in corpora.items()
    }
    synthesis = {dataset: bank.payload for dataset, bank in banks.items()}
    retained = {dataset: bank.retained for dataset, bank in banks.items()}
    feasibility = operating_point_feasibility(
        corpora,
        {dataset: len(bank.retained) for dataset, bank in banks.items()},
        candidate_cap=max_goods_candidates,
        nominal_fpr=fpr,
    )
    controls = {
        dataset: build_balanced_controls(corpus, banks[dataset].geometries)
        for dataset, corpus in corpora.items()
        if banks[dataset].geometries
    }
    methods = [
        _in_support_candidate(
            controls,
            corpora,
            synthesis,
            rank,
            feasibility,
            fpr=fpr,
            transfer_threshold=transfer_threshold,
            max_goods_candidates=max_goods_candidates,
            seed=seed,
        )
    ]
    cross_encoder, verdicts = _cross_encoder_lane(
        corpora,
        retained,
        complete,
        scorer,
        rank,
        feasibility,
        fpr=fpr,
        transfer_threshold=transfer_threshold,
        max_goods_candidates=max_goods_candidates,
        cross_encoder_rows=cross_encoder_rows,
        seed=seed,
    )
    methods.append(cross_encoder)
    assert [str(method["method"]) for method in methods] == list(FOURTH_RESEARCH_METHODS)
    return {
        "feasibility": feasibility,
        "tail_certification": _tail_certification(feasibility, synthesis),
        "control_synthesis": synthesis,
        "methods": methods,
        "conformal": conformal_lane(seed=seed),
        "shortlist_verdicts": verdicts,
    }


def _tail_certification(feasibility: JsonObject, synthesis: dict[str, JsonObject]) -> JsonObject:
    """The distribution-free unit count each corpus's affordable tail needs, and what it has.

    The feasibility lane sizes a bank by how many tail observations an interval needs. This is the
    same question asked without any interval at all: a group-split conformal threshold certifies
    tail `alpha` with confidence `1 - delta` only from `log(delta) / log(1 - alpha)` independent
    units, whatever the scores are. It is the floor no estimator choice can move.
    """
    datasets = feasibility["datasets"]
    assert isinstance(datasets, dict)
    payloads: dict[str, JsonObject] = {}
    for dataset, payload in datasets.items():
        alpha = float(payload["operating_tail_alpha"])
        required = certifiable_units(alpha, DEFAULT_CONFIDENCE)
        available = int(synthesis[dataset]["retained_claims"])
        payloads[dataset] = {
            "operating_tail_alpha": alpha,
            "certifiable_units_required": required,
            "verified_units_available": available,
            "unit_deficit_factor": round(required / available, 1) if available else None,
            "certifiable": available >= required,
        }
    return {
        "confidence": DEFAULT_CONFIDENCE,
        "datasets": payloads,
        "certifiable": all(bool(payload["certifiable"]) for payload in payloads.values()),
    }
