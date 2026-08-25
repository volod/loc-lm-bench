"""Run a whole axiom set over one extraction ledger.

The checker is pure Python over the typed models already in the repo -- no reasoner, no RDF
library, nothing that could fail to install on a scoring host. `crosscheck.py` holds it to OWL
semantics in CI; nothing there runs here.
"""

import logging
from dataclasses import dataclass, field

from llb.prep.ontology.axioms.checks import CHECKS
from llb.prep.ontology.axioms.checks.base import Outcome
from llb.prep.ontology.axioms.constants import N_AXIOM_EXAMPLES
from llb.prep.ontology.axioms.ledger import Ledger
from llb.prep.ontology.axioms.models import (
    Axiom,
    AxiomEvidence,
    AxiomSet,
    AxiomStat,
    LedgerReport,
    ViolationFact,
)
from llb.prep.ontology.axioms.serialize import axiom_turtle

_LOG = logging.getLogger(__name__)


def check_axiom(axiom: Axiom, ledger: Ledger) -> Outcome:
    """Evaluate one axiom; an unknown class is a load-time error, never a silent pass."""
    check = CHECKS.get(axiom.kind)
    if check is None:  # unreachable while `Axiom` validates its kind; kept as a loud floor
        raise ValueError(f"no checker for axiom kind {axiom.kind!r}")
    return check(axiom, ledger)


def _stat(axiom: Axiom, outcome: Outcome) -> AxiomStat:
    return AxiomStat(
        axiom_id=axiom.axiom_id,
        kind=axiom.kind,
        relation=axiom.relation,
        entity_types=list(axiom.entity_types),
        checked=outcome.checked,
        supporting=outcome.supporting,
        violating=outcome.violating,
        unchecked=outcome.unchecked,
        violations=len(outcome.violations),
    )


@dataclass
class LedgerCheck:
    """One ledger's report plus the supporting examples the sign-off worksheet quotes.

    The examples are not part of the published report: a reader wants the base rate and the
    violations, while a REVIEWER deciding whether to enable the axiom also needs to see what it
    accepted. Keeping them beside the report rather than inside it is what stops `report.md` from
    growing a copy of the ledger.
    """

    report: LedgerReport
    examples: dict[str, list[ViolationFact]] = field(default_factory=dict)


def check_ledger(axiom_set: AxiomSet, ledger: Ledger, label: str, source: str) -> LedgerCheck:
    """Check every axiom against one ledger and collect the per-axiom base rates."""
    report = LedgerReport(
        label=label,
        source=source,
        n_docs=ledger.n_docs,
        n_entities=ledger.n_entities,
        n_facts=ledger.n_facts,
        n_relations=ledger.n_relations,
    )
    examples: dict[str, list[ViolationFact]] = {}
    for axiom in axiom_set.axioms:
        outcome = check_axiom(axiom, ledger)
        report.stats.append(_stat(axiom, outcome))
        report.violations.extend(outcome.violations)
        examples[axiom.axiom_id] = outcome.examples[:N_AXIOM_EXAMPLES]
    report.violations.sort(key=lambda v: v.key())
    _LOG.info(
        "[axioms] %s: %d axioms over %d facts -> %d violations",
        label,
        len(axiom_set.axioms),
        ledger.n_facts,
        len(report.violations),
    )
    return LedgerCheck(report=report, examples=examples)


def collect_evidence(axiom_set: AxiomSet, checks: list[LedgerCheck]) -> list[AxiomEvidence]:
    """Per-axiom worksheet rows: the pooled base rate plus a few examples of each side.

    This is the input the sign-off lane reads. It carries the axiom rendered as Turtle beside its
    Ukrainian gloss, so a reviewer decides on the SENTENCE with the corpus evidence in front of
    them rather than on a statistic.
    """
    rows: list[AxiomEvidence] = []
    for axiom in axiom_set.axioms:
        pooled = _pooled_stat(axiom, checks)
        supporting: list[ViolationFact] = []
        contradicting = []
        for check in checks:
            supporting += check.examples.get(axiom.axiom_id, [])
            contradicting += [v for v in check.report.violations if v.axiom_id == axiom.axiom_id]
        rows.append(
            AxiomEvidence(
                axiom_id=axiom.axiom_id,
                kind=axiom.kind,
                gloss=axiom.gloss,
                turtle=axiom_turtle(axiom),
                stat=pooled,
                supporting=supporting[:N_AXIOM_EXAMPLES],
                contradicting=contradicting[:N_AXIOM_EXAMPLES],
            )
        )
    return rows


def _pooled_stat(axiom: Axiom, checks: list[LedgerCheck]) -> AxiomStat:
    """One axiom's counts summed across every ledger the run checked."""
    pooled = AxiomStat(
        axiom_id=axiom.axiom_id,
        kind=axiom.kind,
        relation=axiom.relation,
        entity_types=list(axiom.entity_types),
    )
    for check in checks:
        for stat in check.report.stats:
            if stat.axiom_id != axiom.axiom_id:
                continue
            pooled.checked += stat.checked
            pooled.supporting += stat.supporting
            pooled.violating += stat.violating
            pooled.unchecked += stat.unchecked
            pooled.violations += stat.violations
    return pooled
