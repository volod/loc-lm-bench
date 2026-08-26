"""ontology-validated-answer-gate -- the gate, its scope, its refusals, and its bounded repair.

Everything here is fixture-driven and pure: the gate is a function of an envelope plus the chunks
the prompt carried, so the whole two-step path -- including the semantic repair and the
`ontology_violation` status -- is covered with a fake completer and a fake ledger, no GPU.
"""

import json

import pytest

from llb.backends.base import ERR_TIMEOUT, ChatResult
from llb.eval import common, graph
from llb.eval.answer_envelope import boundary, lane
from llb.eval.answer_envelope.models import AnswerEnvelope
from llb.eval.answer_validation import fixture as gate_fixture
from llb.eval.answer_validation.answer_ledger import answer_extraction
from llb.eval.answer_validation.constants import ANSWER_DOC_ID, GATE_KINDS
from llb.eval.answer_validation.gate import OntologyGate, gate_axioms, load_gate
from llb.eval.answer_validation.scope import CorpusLedger
from llb.executor.durability_journal import _JOURNALED_STATE_KEYS
from llb.prep.ontology.axioms.constants import SYMMETRIC
from llb.core.paths import PROJECT_ROOT
from llb.prep.ontology.axioms.loader import load_axioms, save_axioms
from llb.prep.ontology.axioms.models import Axiom, AxiomSet
from llb.prep.ontology.models import DocExtraction

DOC = "gate_fixture.md"
TEXT = "Львів має населення 717 тисяч осіб.\nКодекс містить розділ."
SENTENCES = [(0, 35), (36, 57)]


def _span(index: int) -> dict:
    start, end = SENTENCES[index]
    return {"doc_id": DOC, "char_start": start, "char_end": end, "text": TEXT[start:end]}


def _ledger(entities=(), facts=()) -> CorpusLedger:
    return CorpusLedger(
        [
            DocExtraction.model_validate(
                {"doc_id": DOC, "entities": list(entities), "facts": list(facts)}
            )
        ]
    )


def _entity(name, etype, index, aliases=()):
    return {"name": name, "type": etype, "aliases": list(aliases), "mentions": [_span(index)]}


def _fact(subject, relation, obj, index):
    return {"subject": subject, "relation": relation, "object": obj, "evidence": _span(index)}


def _envelope(*triples, answer="Відповідь.") -> AnswerEnvelope:
    return AnswerEnvelope.model_validate(
        {
            "answer": answer,
            "abstained": not triples,
            "claims": [
                {
                    "text": f"{t['subject']} {t['relation']} {t['object']}.",
                    "citations": [1],
                    "triple": t,
                }
                for t in triples
            ],
        }
    )


def _triple(subject, relation, obj, subject_type="MISC", object_type="MISC"):
    return {
        "subject": subject,
        "relation": relation,
        "object": obj,
        "subject_type": subject_type,
        "object_type": object_type,
    }


FUNCTIONAL_AXIOM = Axiom(
    axiom_id="func-naselennia",
    kind="functional",
    relation="має населення",
    signed_by="reviewer",
    signed_on="2026-01-01",
)


# --- what the gate refuses, and what it must not -----------------------------------------------


def test_an_answer_contradicting_a_retrieved_ledger_fact_is_a_violation():
    ledger = _ledger(facts=[_fact("Львів", "має населення", "717 тисяч осіб", 0)])
    gate = OntologyGate([FUNCTIONAL_AXIOM], ledger)
    verdict = gate.check(
        _envelope(_triple("Львів", "має населення", "2 мільйони осіб")), [_span(0)]
    )
    assert verdict.classes == ["functional"] and verdict.checked_triples == 1


def test_the_same_contradiction_outside_the_retrieved_chunks_is_not_a_violation():
    # The declared scope is what the model was SHOWN. Refusing on evidence it never saw would make
    # the gate a corpus-wide fact checker and blame the model for a document it was not given.
    ledger = _ledger(facts=[_fact("Львів", "має населення", "717 тисяч осіб", 0)])
    gate = OntologyGate([FUNCTIONAL_AXIOM], ledger)
    verdict = gate.check(
        _envelope(_triple("Львів", "має населення", "2 мільйони осіб")), [_span(1)]
    )
    assert verdict.ok and verdict.scoped_facts == 0


def test_a_contradiction_the_corpus_has_with_itself_never_refuses_an_answer():
    ledger = _ledger(
        facts=[
            _fact("Львів", "має населення", "717 тисяч осіб", 0),
            _fact("Львів", "має населення", "700 тисяч осіб", 0),
        ]
    )
    gate = OntologyGate([FUNCTIONAL_AXIOM], ledger)
    # The answer restates one of the two values the ledger already disagrees about.
    verdict = gate.check(_envelope(_triple("Львів", "має населення", "717 тисяч осіб")), [_span(0)])
    assert verdict.ok


def test_a_bound_the_corpus_already_breaks_never_refuses_an_answer():
    # The retrieved chunk already carries three values of a relation bounded at two, so the axiom
    # is broken whatever the answer says. Blaming the model for joining that group would refuse a
    # correct answer for a data problem `validate-ontology-axioms` reports.
    bounded = Axiom(
        axiom_id="maxcard-ye",
        kind="max_cardinality",
        relation="є",
        max_count=2,
        signed_by="reviewer",
        signed_on="2026-01-01",
    )
    over_budget = _ledger(
        facts=[
            _fact("Машина", "є", "модель", 0),
            _fact("Машина", "є", "пристрій", 0),
            _fact("Машина", "є", "автомат", 0),
        ]
    )
    gate = OntologyGate([bounded], over_budget)
    assert gate.check(_envelope(_triple("Машина", "є", "абстракція")), [_span(0)]).ok
    # ...while the same axiom on a corpus that was WITHIN the bound still refuses the answer that
    # pushes it over, so the subtraction narrows the gate rather than disabling the class.
    within = _ledger(facts=[_fact("Машина", "є", "модель", 0), _fact("Машина", "є", "пристрій", 0)])
    verdict = OntologyGate([bounded], within).check(
        _envelope(_triple("Машина", "є", "абстракція")), [_span(0)]
    )
    assert verdict.classes == ["max_cardinality"]


def test_a_subject_the_corpus_already_disagrees_about_never_refuses_an_answer():
    # Two conflicting corpus values leave no single fact for the answer to contradict, so a third
    # value is not a contradiction the answer committed.
    conflicted = _ledger(
        facts=[
            _fact("Львів", "має населення", "717 тисяч осіб", 0),
            _fact("Львів", "має населення", "700 тисяч осіб", 0),
        ]
    )
    gate = OntologyGate([FUNCTIONAL_AXIOM], conflicted)
    assert gate.check(
        _envelope(_triple("Львів", "має населення", "2 мільйони осіб")), [_span(0)]
    ).ok


def test_an_alias_the_corpus_records_folds_before_the_comparison():
    ledger = _ledger(
        entities=[_entity("717 тисяч осіб", "QUANTITY", 0, aliases=["717000 осіб"])],
        facts=[_fact("Львів", "має населення", "717 тисяч осіб", 0)],
    )
    gate = OntologyGate([FUNCTIONAL_AXIOM], ledger)
    verdict = gate.check(_envelope(_triple("Львів", "має населення", "717000 осіб")), [_span(0)])
    assert verdict.ok


def test_an_alias_two_entities_both_claim_is_dropped_rather_than_resolved():
    ledger = _ledger(
        entities=[
            _entity("Перший", "ORG", 0, aliases=["спільне"]),
            _entity("Другий", "ORG", 0, aliases=["спільне"]),
        ]
    )
    assert ledger.canonical("спільне") == "спільне"


def test_a_misc_endpoint_type_asserts_nothing_on_either_side():
    domain_axiom = Axiom(
        axiom_id="domain-avtor",
        kind="domain",
        relation="автор",
        entity_types=["PERSON"],
        signed_by="reviewer",
        signed_on="2026-01-01",
    )
    gate = OntologyGate([domain_axiom], _ledger())
    verdict = gate.check(_envelope(_triple("Хтось", "автор", "Твір", "MISC", "WORK")), [_span(0)])
    assert verdict.ok
    # ...and the answer ledger carries no type assertion for the MISC endpoint at all.
    extraction = answer_extraction(_envelope(_triple("Хтось", "автор", "Твір", "MISC", "WORK")))
    assert [entity.name for entity in extraction.entities] == ["Твір"]
    assert extraction.doc_id == ANSWER_DOC_ID


def test_an_envelope_declaring_no_triple_is_unchecked_rather_than_cleared():
    verdict = OntologyGate([FUNCTIONAL_AXIOM], _ledger()).check(_envelope(), [_span(0)])
    assert verdict.ok and not verdict.checkable and verdict.checked_triples == 0


def test_the_symmetric_class_can_never_refuse_an_answer():
    # `symmetric` reports the LEDGER GAP a missing counterpart is; an answer is never asked to
    # state both directions, so enabling it here would refuse correct one-way answers.
    assert SYMMETRIC not in GATE_KINDS
    signed = AxiomSet(
        version="t",
        axioms=[
            Axiom(
                axiom_id="sym",
                kind=SYMMETRIC,
                relation="межує з",
                signed_by="reviewer",
                signed_on="2026-01-01",
            )
        ],
    )
    assert gate_axioms(signed) == []


# --- the refusals -------------------------------------------------------------------------------


def test_an_unsigned_axiom_file_is_refused_with_a_named_error(tmp_path):
    unsigned = AxiomSet(
        version="t", axioms=[Axiom(axiom_id="f", kind="functional", relation="діє")]
    )
    turtle, _json_path = save_axioms(unsigned, tmp_path / "axioms.ttl", ["candidate"])
    with pytest.raises(SystemExit, match="none of its 1 axioms is signed"):
        load_gate(turtle, [])


def test_a_signed_file_of_only_excluded_classes_is_refused_too(tmp_path):
    only_symmetric = AxiomSet(
        version="t",
        axioms=[
            Axiom(
                axiom_id="sym",
                kind=SYMMETRIC,
                relation="межує з",
                signed_by="reviewer",
                signed_on="2026-01-01",
            )
        ],
    )
    turtle, _json_path = save_axioms(only_symmetric, tmp_path / "axioms.ttl", ["signed"])
    with pytest.raises(SystemExit, match="classes the gate does not decide"):
        load_gate(turtle, [])


@pytest.mark.parametrize(
    "values,message",
    [
        ({"answer_validation": "ontology"}, "needs answer_format=envelope"),
        (
            {"answer_validation": "ontology", "answer_format": "envelope"},
            "needs ontology_axioms",
        ),
        (
            {
                "answer_validation": "ontology",
                "answer_format": "envelope",
                "ontology_axioms": "samples/ontology/axioms_uk_v1.ttl",
            },
            "needs ontology_ledger",
        ),
        ({"ontology_axioms": "samples/ontology/axioms_uk_v1.ttl"}, "only apply when"),
    ],
)
def test_an_incomplete_gate_configuration_is_refused_before_any_model_call(values, message):
    from llb.core.config import RunConfig

    with pytest.raises(ValueError, match=message):
        RunConfig(**values)


# --- the bounded semantic repair ----------------------------------------------------------------


def _scripted(*results):
    queue = list(results)
    calls: list[list[dict]] = []

    def chat(messages):
        calls.append(messages)
        return queue.pop(0)

    return chat, calls


def _completion(envelope: AnswerEnvelope) -> str:
    return json.dumps(envelope.model_dump(), ensure_ascii=False)


VIOLATING = _envelope(_triple("Львів", "має населення", "2 мільйони осіб"))
CLEAN = _envelope(_triple("Львів", "має населення", "717 тисяч осіб"))
GATE = OntologyGate(
    [FUNCTIONAL_AXIOM], _ledger(facts=[_fact("Львів", "має населення", "717 тисяч осіб", 0)])
)


def _validate(envelope):
    return GATE.check(envelope, [_span(0)])


def test_the_semantic_repair_is_spent_once_and_names_the_broken_constraint():
    chat, calls = _scripted(ChatResult(text=_completion(CLEAN), completion_tokens=40))
    done = boundary.complete_envelope(
        chat, [{"role": "user", "content": "q"}], ChatResult(text=_completion(VIOLATING)), _validate
    )
    assert done.validation_repaired is True and done.validation.ok
    assert done.repaired is False  # a SEMANTIC repair is not a formatting one
    assert len(calls) == 1
    assert "func-naselennia" in calls[0][-1]["content"]


def test_a_semantic_repair_that_still_violates_leaves_the_first_answer_standing():
    chat, _calls = _scripted(ChatResult(text=_completion(VIOLATING), completion_tokens=40))
    done = boundary.complete_envelope(chat, [], ChatResult(text=_completion(VIOLATING)), _validate)
    assert done.validation_repaired is True and not done.validation.ok
    assert done.parse.envelope.answer == VIOLATING.answer


def test_a_semantic_repair_that_stops_parsing_can_never_damage_the_answer():
    chat, _calls = _scripted(ChatResult(text="проза без JSON", completion_tokens=4))
    done = boundary.complete_envelope(chat, [], ChatResult(text=_completion(VIOLATING)), _validate)
    assert done.parse.status == common.OK and not done.validation.ok


def test_a_conformant_and_consistent_answer_spends_no_repair_at_all():
    chat, calls = _scripted()
    done = boundary.complete_envelope(chat, [], ChatResult(text=_completion(CLEAN)), _validate)
    assert done.repaired is False and done.validation_repaired is False and calls == []


def test_a_transport_failure_never_reaches_the_gate():
    chat, calls = _scripted()
    done = boundary.complete_envelope(chat, [], ChatResult(text="", error=ERR_TIMEOUT), _validate)
    assert done.validation is None and calls == []


# --- the graph seam and the recorded columns ----------------------------------------------------


class FakeLauncher:
    def __init__(self, *results):
        self._queue = list(results)

    def chat(self, messages, max_tokens, temperature, timeout):
        return self._queue.pop(0)


def _generate(*results):
    node = graph.make_generate_node(
        FakeLauncher(*results),
        96,
        0.0,
        30.0,
        answer_format=lane.ENVELOPE,
        validator=lambda envelope, chunks: GATE.check(envelope, chunks),
    )
    return node(
        {
            "question": "Яке населення Львова?",
            "context": "[1] текст",
            "retrieved": [_span(0)],
        }
    )


def test_a_violating_answer_ends_in_the_ontology_violation_status():
    state = _generate(
        ChatResult(text=_completion(VIOLATING)), ChatResult(text=_completion(VIOLATING))
    )
    assert state["status"] == common.ONTOLOGY_VIOLATION
    assert state["validation_classes"] == ["functional"]
    assert state["validation_violations"] == 1
    assert state["answer"] == VIOLATING.answer  # a rejection nobody can inspect is not evidence


def test_a_rescued_answer_ends_ok_and_records_the_repair():
    state = _generate(ChatResult(text=_completion(VIOLATING)), ChatResult(text=_completion(CLEAN)))
    assert state["status"] == common.OK
    assert state["validation_repaired"] is True and state["validation_violations"] == 0


def test_an_ungated_envelope_run_records_no_validation_column():
    node = graph.make_generate_node(
        FakeLauncher(ChatResult(text=_completion(VIOLATING))),
        96,
        0.0,
        30.0,
        answer_format=lane.ENVELOPE,
    )
    state = node({"question": "q", "context": "c", "retrieved": [_span(0)]})
    assert "validation_checked_triples" not in state and state["status"] == common.OK


def test_every_validation_state_key_survives_a_resume():
    state = _generate(
        ChatResult(text=_completion(VIOLATING)), ChatResult(text=_completion(VIOLATING))
    )
    missing = [key for key in state if key not in _JOURNALED_STATE_KEYS]
    assert missing == []


# --- the committed adversarial fixture ----------------------------------------------------------


def test_every_planted_violation_is_caught_by_its_own_axiom_class():
    report = gate_fixture.fixture_report(gate_fixture.run_fixture())
    assert set(report.catch_rate_by_class) == set(GATE_KINDS)
    assert report.all_caught, report.catch_rate_by_class


def test_the_false_rejection_rate_is_measured_on_adversarial_correct_answers():
    # NOT asserted to be zero: the fixture carries a paraphrase the corpus never recorded as an
    # alias, which surface folding cannot see. That case IS the measured rate.
    report = gate_fixture.fixture_report(gate_fixture.run_fixture())
    assert report.n_correct >= 8
    assert report.false_rejections == ["ok-unrecorded-paraphrase-001"]
    assert report.false_rejection_rate == pytest.approx(1 / report.n_correct, abs=1e-6)


def test_the_scope_case_is_never_refused():
    report = gate_fixture.fixture_report(gate_fixture.run_fixture())
    assert report.scope_failures == []


def test_the_fixture_may_only_enable_axioms_the_committed_candidate_set_carries():
    # The fixture signs its own in-memory copy so it can exercise the enabled path without
    # claiming a reviewer accepted anything; naming an axiom nobody wrote is an error, not a
    # silently empty gate.
    payload = gate_fixture.load_fixture()
    candidates = load_axioms(PROJECT_ROOT / str(payload["axiom_source"]))
    with pytest.raises(ValueError, match="does not carry"):
        gate_fixture.sign_for_fixture(candidates, ["no-such-axiom"])
    signed = gate_fixture.sign_for_fixture(candidates, [str(i) for i in payload["axiom_ids"]])
    assert {axiom.axiom_id for axiom in signed.signed} == set(payload["axiom_ids"])
