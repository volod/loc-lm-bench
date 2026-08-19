"""Pin-or-expose verdict for each swept restoration constant.

A sweep that only prints per-setting rows leaves the decision to whoever reads it last. Each of the
three constants therefore gets one explicit verdict, decided by the same two readings for all of
them: whether any alternative value RETRIEVED more than the shipped default (paired, on the same
items) and whether it did so without rewriting more of the user's words into something they did not
type.

- `pin` -- no alternative retrieved more, so the conservative default is not costing recall and the
  knob stays at its shipped value.
- `adopt` -- an alternative separates on paired recall and does not raise the wrong-correction
  share: the default IS costing recoverable recall and the default should move.
- `expose` -- an alternative gains recall but either does not separate at this item count or buys
  the gain with more wrong corrections. The trade is real but corpus-dependent, so it stays an
  operator knob rather than a new default.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from llb.eval.restoration_sweep_lanes import SweepResult
from llb.rag.fusion_evidence.evidence_gate import READING_SEPARATED
from llb.rag.fusion_evidence.paired import PairedComparison, paired_comparison, reading_of
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    bootstrap_index_sets,
)
from llb.rag.query_prep.restore_policy import DEFAULT_RESTORATION_POLICY, RestorationPolicy

CONSTANT_SURFACE = "surface_max_distance"
CONSTANT_CUTOFF = "ambiguous_token_max_chars"
CONSTANT_RANK = "rank_order"
SWEPT_CONSTANTS = (CONSTANT_SURFACE, CONSTANT_CUTOFF, CONSTANT_RANK)

VERDICT_PIN = "pin"
VERDICT_ADOPT = "adopt"
VERDICT_EXPOSE = "expose"

# The config knob an operator sets when a constant's verdict is `expose`.
CONSTANT_KNOBS = {
    CONSTANT_SURFACE: "query_prep_surface_max_distance",
    CONSTANT_CUTOFF: "query_prep_ambiguous_max_chars",
    CONSTANT_RANK: "query_prep_restore_rank",
}


@dataclass(frozen=True)
class AlternativeReading:
    """One alternative value of one constant, read against the shipped default."""

    constant: str
    value: object
    label: str
    recall: PairedComparison
    reading: str
    recall_delta: float
    mrr_delta: float
    wrong_share_delta: float
    restoration_recall_delta: float

    @property
    def gains_recall(self) -> bool:
        return self.recall_delta > 0.0

    @property
    def costs_precision(self) -> bool:
        return self.wrong_share_delta > 0.0


@dataclass(frozen=True)
class ConstantVerdict:
    """The decision for one constant plus the alternatives it was decided on."""

    constant: str
    default_value: object
    verdict: str
    rationale: str
    alternatives: tuple[AlternativeReading, ...]


def _varied_constant(policy: RestorationPolicy) -> str | None:
    """The single constant this policy moves off the default, or None when it moves zero or two."""
    varied = [
        name
        for name in SWEPT_CONSTANTS
        if getattr(policy, name) != getattr(DEFAULT_RESTORATION_POLICY, name)
    ]
    return varied[0] if len(varied) == 1 else None


def _alternative(
    result: SweepResult,
    policy: RestorationPolicy,
    constant: str,
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> AlternativeReading:
    default = result.pooled(DEFAULT_RESTORATION_POLICY.label)
    candidate = result.pooled(policy.label)
    index_sets = bootstrap_index_sets(len(default.hits), resamples, seed)
    comparison = paired_comparison(list(candidate.hits), list(default.hits), index_sets, confidence)
    return AlternativeReading(
        constant=constant,
        value=getattr(policy, constant),
        label=policy.label,
        recall=comparison,
        reading=reading_of(comparison, confidence),
        recall_delta=candidate.recall_at_k - default.recall_at_k,
        mrr_delta=candidate.mrr - default.mrr,
        wrong_share_delta=candidate.counts.wrong_share - default.counts.wrong_share,
        restoration_recall_delta=(
            candidate.counts.restoration_recall - default.counts.restoration_recall
        ),
    )


def _rationale(constant: str, best: AlternativeReading | None, verdict: str) -> str:
    knob = CONSTANT_KNOBS[constant]
    if verdict == VERDICT_PIN:
        if best is None:
            return "no alternative value was measured, so the default stands unchallenged"
        return (
            f"the best alternative ({best.value}) retrieves {best.recall_delta:+.4f} recall and "
            f"{best.wrong_share_delta:+.4f} wrong-correction share: the conservative default is "
            "not costing recoverable recall on this corpus"
        )
    assert best is not None
    if verdict == VERDICT_ADOPT:
        return (
            f"{best.value} separates on paired recall ({best.recall_delta:+.4f}) without raising "
            f"the wrong-correction share ({best.wrong_share_delta:+.4f}): move the default"
        )
    return (
        f"{best.value} retrieves {best.recall_delta:+.4f} recall at "
        f"{best.wrong_share_delta:+.4f} wrong-correction share and reads `{best.reading}`; keep "
        f"the default and set `{knob}` when a corpus wants the trade"
    )


def _decide(alternatives: Sequence[AlternativeReading]) -> tuple[str, AlternativeReading | None]:
    if not alternatives:
        return VERDICT_PIN, None
    best = max(alternatives, key=lambda item: (item.recall_delta, -item.wrong_share_delta))
    if not best.gains_recall:
        return VERDICT_PIN, best
    if best.reading == READING_SEPARATED and not best.costs_precision:
        return VERDICT_ADOPT, best
    return VERDICT_EXPOSE, best


def constant_verdicts(
    result: SweepResult,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int,
) -> tuple[ConstantVerdict, ...]:
    """One verdict per constant, over the one-factor-at-a-time settings the sweep measured."""
    by_constant: dict[str, list[AlternativeReading]] = {name: [] for name in SWEPT_CONSTANTS}
    for policy in result.policies:
        constant = _varied_constant(policy)
        if constant is None:
            continue
        by_constant[constant].append(
            _alternative(
                result,
                policy,
                constant,
                resamples=resamples,
                confidence=confidence,
                seed=seed,
            )
        )
    verdicts: list[ConstantVerdict] = []
    for constant, alternatives in by_constant.items():
        verdict, best = _decide(alternatives)
        verdicts.append(
            ConstantVerdict(
                constant=constant,
                default_value=getattr(DEFAULT_RESTORATION_POLICY, constant),
                verdict=verdict,
                rationale=_rationale(constant, best, verdict),
                alternatives=tuple(alternatives),
            )
        )
    return tuple(verdicts)


def recommended_policy(verdicts: Sequence[ConstantVerdict]) -> RestorationPolicy:
    """The default with every `adopt` verdict applied; `pin` and `expose` leave it unchanged."""
    policy = DEFAULT_RESTORATION_POLICY
    for verdict in verdicts:
        if verdict.verdict == VERDICT_ADOPT and verdict.alternatives:
            best = max(verdict.alternatives, key=lambda item: item.recall_delta)
            if verdict.constant == CONSTANT_SURFACE:
                policy = replace(policy, surface_max_distance=int(str(best.value)))
            elif verdict.constant == CONSTANT_CUTOFF:
                policy = replace(policy, ambiguous_token_max_chars=int(str(best.value)))
            else:
                policy = replace(policy, rank_order=str(best.value))
    return policy
