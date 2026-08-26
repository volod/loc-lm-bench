"""The committed adversarial fixture, and the two rates it measures on the gate.

A validator is easy to make look good: refuse more, catch more. The only honest way to read one is
to measure both directions on the SAME artifact, so this fixture carries planted violating answers
(one per axiom class the gate decides) AND correct answers a naive checker would refuse -- a
legitimately multi-valued relation, a paraphrase the corpus itself records as an alias, an entity
the model could only type `MISC`, a one-way assertion of a symmetric relation, and a declared
abstention. The false-rejection number is therefore MEASURED on adversarial cases rather than
asserted to be zero.

The axioms are named by id from the committed CANDIDATE set and enabled for the fixture only.
Signing the shipped set is a domain reviewer's decision (`ontology-axiom-signoff`), and a test may
not stand in for one -- so the fixture signs its own copy in memory, and nothing it does can enable
an axiom in a real run.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from llb.core.contracts.rag import ChunkRecord
from llb.core.paths import PROJECT_ROOT
from llb.eval.answer_envelope.models import AnswerEnvelope
from llb.eval.answer_validation.constants import FIXTURE_PATH
from llb.eval.answer_validation.gate import OntologyGate, gate_axioms
from llb.eval.answer_validation.models import GateVerdict
from llb.eval.answer_validation.scope import CorpusLedger
from llb.prep.ontology.axioms.loader import load_axioms
from llb.prep.ontology.axioms.models import AxiomSet
from llb.prep.ontology.models import DocExtraction

EXPECT_REJECT = "reject"
EXPECT_ACCEPT = "accept"

# The three case categories, kept apart because they answer different questions. Only
# `adversarial_correct` is the false-rejection denominator: a `scope` case is an answer that really
# does contradict the corpus, and folding it in would price the gate's DECLARED scope as an error.
CATEGORY_PLANTED = "planted_violation"
CATEGORY_CORRECT = "adversarial_correct"
CATEGORY_SCOPE = "scope"

# What the fixture's in-memory signature says, so a signed set that ever escaped into an artifact
# names itself for what it is.
FIXTURE_SIGNER = "fixture (unreviewed; enabled for this fixture only)"
FIXTURE_SIGN_DATE = "1970-01-01"


@dataclass(frozen=True, slots=True)
class FixtureCase:
    """One fixture case: the answer, the chunks it was given, and what the gate must decide."""

    case_id: str
    expect: str
    category: str
    axiom_class: str | None
    naive_trap: str
    why: str
    chunks: list[ChunkRecord]
    envelope: AnswerEnvelope

    @property
    def should_reject(self) -> bool:
        return self.expect == EXPECT_REJECT


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """What the gate did with one case, beside what it was supposed to do."""

    case: FixtureCase
    verdict: GateVerdict

    @property
    def rejected(self) -> bool:
        return not self.verdict.ok

    @property
    def caught(self) -> bool:
        """A planted violation refused under the axiom CLASS it was planted for."""
        return (
            self.case.should_reject
            and self.case.axiom_class is not None
            and self.case.axiom_class in self.verdict.classes
        )

    @property
    def false_rejection(self) -> bool:
        return not self.case.should_reject and self.rejected


def fixture_path() -> Path:
    return PROJECT_ROOT / FIXTURE_PATH


def load_fixture(path: Path | str | None = None) -> Mapping[str, Any]:
    """The committed fixture payload, validated only for the keys the harness reads."""
    target = Path(path) if path is not None else fixture_path()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("cases"):
        raise ValueError(f"{target}: not an answer-gate fixture (no cases)")
    return payload


def fixture_cases(payload: Mapping[str, Any]) -> list[FixtureCase]:
    chunks = list(payload["chunks"])
    return [
        FixtureCase(
            case_id=str(case["id"]),
            expect=str(case["expect"]),
            category=str(case["category"]),
            axiom_class=case.get("axiom_class"),
            naive_trap=str(case.get("naive_trap", "")),
            why=str(case.get("why", "")),
            chunks=[cast(ChunkRecord, dict(chunks[index])) for index in case["chunks"]],
            envelope=AnswerEnvelope.model_validate(case["envelope"]),
        )
        for case in payload["cases"]
    ]


def sign_for_fixture(axiom_set: AxiomSet, axiom_ids: Sequence[str]) -> AxiomSet:
    """A copy of the set whose NAMED axioms carry the fixture's own in-memory signature.

    Never written anywhere: the gate refuses an unsigned file, and the fixture has to exercise the
    enabled path without claiming a reviewer accepted anything.
    """
    wanted = set(axiom_ids)
    missing = sorted(wanted - {axiom.axiom_id for axiom in axiom_set.axioms})
    if missing:
        raise ValueError(f"the fixture names axioms the candidate set does not carry: {missing}")
    signed = [
        axiom.model_copy(update={"signed_by": FIXTURE_SIGNER, "signed_on": FIXTURE_SIGN_DATE})
        if axiom.axiom_id in wanted
        else axiom
        for axiom in axiom_set.axioms
    ]
    return AxiomSet(version=axiom_set.version, axioms=signed)


def fixture_gate(payload: Mapping[str, Any]) -> OntologyGate:
    """The gate the fixture is measured on: its own ledger, its own named axioms."""
    candidates = load_axioms(PROJECT_ROOT / str(payload["axiom_source"]))
    signed = sign_for_fixture(candidates, [str(i) for i in payload["axiom_ids"]])
    extractions = [DocExtraction.model_validate(doc) for doc in payload["ledger"]]
    return OntologyGate(gate_axioms(signed), CorpusLedger(extractions))


def run_fixture(payload: Mapping[str, Any] | None = None) -> list[CaseOutcome]:
    """Run every fixture case through the gate, in file order."""
    data = payload if payload is not None else load_fixture()
    gate = fixture_gate(data)
    return [
        CaseOutcome(case=case, verdict=gate.check(case.envelope, case.chunks))
        for case in fixture_cases(data)
    ]


@dataclass(frozen=True, slots=True)
class FixtureReport:
    """The two rates the fixture exists to produce, with the populations behind them.

    `catch_rate_by_class` is per AXIOM CLASS because that is the unit the adopt-or-reject verdict
    decides on; `false_rejection_rate` is one number over the adversarial-correct cases, and its
    denominator travels with it -- a rate over seven cases is not a rate over seventy, and stating
    it without the count would invite reading it as one.
    """

    catch_rate_by_class: dict[str, float]
    n_planted_by_class: dict[str, int]
    false_rejections: list[str]
    n_correct: int
    scope_failures: list[str]

    @property
    def false_rejection_rate(self) -> float:
        return len(self.false_rejections) / self.n_correct if self.n_correct else 0.0

    @property
    def all_caught(self) -> bool:
        return all(rate == 1.0 for rate in self.catch_rate_by_class.values())

    def lines(self) -> list[str]:
        """Console lines: what the gate caught, and what it refused that it should not have."""
        out = [
            f"[answer-gate] planted violations caught: "
            f"{sum(self.n_planted_by_class.values())} over "
            f"{len(self.catch_rate_by_class)} axiom classes"
        ]
        out += [
            f"           {name:<20} catch {rate:.3f} (n={self.n_planted_by_class[name]})"
            for name, rate in sorted(self.catch_rate_by_class.items())
        ]
        out.append(
            f"[answer-gate] false-rejection rate {self.false_rejection_rate:.3f} "
            f"({len(self.false_rejections)}/{self.n_correct} adversarial correct answers"
            + (f": {', '.join(self.false_rejections)}" if self.false_rejections else "")
            + ")"
        )
        if self.scope_failures:
            out.append(
                "[answer-gate] SCOPE FAILURE -- refused a contradiction the prompt never "
                f"carried: {', '.join(self.scope_failures)}"
            )
        return out


def fixture_report(outcomes: Sequence[CaseOutcome]) -> FixtureReport:
    """Roll the per-case outcomes into the catch and false-rejection rates."""
    planted = [o for o in outcomes if o.case.category == CATEGORY_PLANTED]
    classes = sorted({str(o.case.axiom_class) for o in planted})
    correct = [o for o in outcomes if o.case.category == CATEGORY_CORRECT]
    return FixtureReport(
        catch_rate_by_class={
            name: _rate([o.caught for o in planted if o.case.axiom_class == name])
            for name in classes
        },
        n_planted_by_class={
            name: sum(1 for o in planted if o.case.axiom_class == name) for name in classes
        },
        false_rejections=[o.case.case_id for o in correct if o.false_rejection],
        n_correct=len(correct),
        scope_failures=[
            o.case.case_id for o in outcomes if o.case.category == CATEGORY_SCOPE and o.rejected
        ],
    )


def _rate(flags: Sequence[bool]) -> float:
    return sum(1 for flag in flags if flag) / len(flags) if flags else 0.0
