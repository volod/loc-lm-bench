"""answer-gate-equivalence -- value equivalence, surface identity, and the refusal re-labelling.

Every false rejection the answer gate produced was an IDENTITY failure rather than an axiom
failure, at both ends: on the answer side a correct value restated in another written form read as
a second value, and on the reading side a correct short answer to an inflected reference was
labelled a catch. These tests pin both fixes and, just as importantly, the places they must NOT
fire -- a value normalizer that folds two DIFFERENT quantities loses a planted violation, which is
the one failure this gate may not have.
"""

import json

import pytest

from llb.eval import common
from llb.eval.answer_validation.equivalence import value_key
from llb.eval.answer_validation.identity import SurfaceIdentity
from llb.eval.answer_validation.labelling import (
    LABEL_CATCH,
    LABEL_FALSE_REJECTION,
    SIGNAL_ANSWER_WITHIN_REFERENCE,
    SIGNAL_CONTAINS,
    SIGNAL_CONTAINS_LEMMA,
    label_refusals,
    relabelled,
)
from llb.prep.ontology.models import DocExtraction

DOC = "equivalence_fixture.md"


def _span(text: str) -> dict:
    return {"doc_id": DOC, "char_start": 0, "char_end": len(text), "text": text}


def _entity(name: str, entity_type: str, aliases: list[str] | None = None) -> dict:
    return {
        "name": name,
        "type": entity_type,
        "aliases": aliases or [],
        "mentions": [_span(name)],
    }


def _ledger(*entities: dict) -> list[DocExtraction]:
    return [DocExtraction.model_validate({"doc_id": DOC, "entities": list(entities), "facts": []})]


# --- the value key: what two surfaces of one value share ----------------------------------------


@pytest.mark.parametrize(
    "left, right, entity_type",
    [
        ("2,9 млн осіб", "2.9 мільйона осіб", "QUANTITY"),
        ("2 900 000 осіб", "2.9 мільйона осіб", "QUANTITY"),
        ("717 тисяч осіб", "717 000 осіб", "QUANTITY"),
        ("45 %", "45 відсотків", "QUANTITY"),
        ("двадцять років", "20 років", "DURATION"),
        ("20 років", "20 роками", "DURATION"),
        ("27.08.1856", "27 серпня 1856 року", "DATE"),
        ("2021", "2021 року", "DATE"),
    ],
)
def test_two_written_forms_of_one_value_share_a_key(left, right, entity_type):
    assert value_key(left, entity_type) == value_key(right, entity_type) is not None


@pytest.mark.parametrize(
    "left, right, entity_type",
    [
        ("2,9 млн осіб", "2,8 млн осіб", "QUANTITY"),  # the near-value guard
        ("717 тисяч осіб", "717 осіб", "QUANTITY"),
        ("3 млн осіб", "близько 3 млн осіб", "QUANTITY"),  # a hedge is not the bare value
        ("20 років", "20 місяців", "DURATION"),
        ("двадцять років", "тридцять років", "DURATION"),
        ("2021 року", "2020 року", "DATE"),
        ("1 січня 2021", "2 січня 2021", "DATE"),
    ],
)
def test_values_that_differ_keep_different_keys(left, right, entity_type):
    assert value_key(left, entity_type) != value_key(right, entity_type)


def test_the_key_is_type_scoped_so_one_string_is_not_two_types_at_once():
    # `1990 рік` read as a DATE and read as a QUANTITY are different claims, and folding them
    # together would invent exactly the identity the gate refuses.
    assert value_key("1990 рік", "DATE") != value_key("1990 рік", "QUANTITY")


@pytest.mark.parametrize(
    "surface, entity_type",
    [
        ("Київ", "QUANTITY"),  # a name is not a value
        ("Лісова пісня", "DATE"),
        ("кілька років", "DURATION"),  # no number the grammar reads
        ("20", "DURATION"),  # no unit
        ("від 2 до 5 років", "DURATION"),  # two numbers say two things
        ("на честь 2021 року видано", "DATE"),  # words the date grammar does not model
        ("Львів", "PERSON"),  # not a value type at all
    ],
)
def test_a_surface_the_grammar_cannot_read_has_no_key(surface, entity_type):
    assert value_key(surface, entity_type) is None


# --- surface identity: aliases, the resolution overlay, then values -----------------------------


def test_the_corpus_alias_is_what_folds_a_recorded_paraphrase():
    identity = SurfaceIdentity(_ledger(_entity("ООН", "ORG", ["Організація Об'єднаних Націй"])))
    assert identity.fold("Організація Об'єднаних Націй", "ORG") == "ООН"
    assert identity.fold("невідома установа", "ORG") == "невідома установа"


def test_an_alias_two_entities_claim_is_dropped_rather_than_resolved():
    identity = SurfaceIdentity(
        _ledger(_entity("Рада Європи", "ORG", ["Рада"]), _entity("Рада міністрів", "ORG", ["Рада"]))
    )
    assert identity.fold("Рада", "ORG") == "Рада"


def test_a_declared_endpoint_folds_through_the_resolution_overlay(tmp_path):
    # The graph lane already decides which nodes are one entity; the gate reuses that decision
    # rather than inventing a second notion of identity that could refuse what the graph merged.
    overlay = tmp_path / "overlay.jsonl"
    overlay.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in [
                {"kind": "overlay", "threshold": 0.9, "n_clusters": 1},
                {
                    "kind": "cluster",
                    "canonical_id": 1,
                    "size": 2,
                    "member_ids": [1, 2],
                    "canonical_name": "Леся Українка",
                    "member_names": ["Леся Українка", "Лариса Косач"],
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    identity = SurfaceIdentity(_ledger(_entity("Леся Українка", "PERSON")), overlay=overlay)
    assert identity.n_merged_surfaces == 1
    assert identity.fold("Лариса Косач", "PERSON") == "Леся Українка"
    # Without the overlay the same surface stays its own: an overlay is a PROPOSAL a run supplies.
    assert (
        SurfaceIdentity(_ledger(_entity("Леся Українка", "PERSON"))).fold("Лариса Косач", "PERSON")
        == "Лариса Косач"
    )


def test_a_value_folds_onto_the_corpus_surface_the_alias_map_already_chose():
    # The value fold must land where the alias fold lands, or two answers stating one value would
    # end on two different surfaces and read as two values.
    identity = SurfaceIdentity(
        _ledger(_entity("2.9 мільйона осіб", "QUANTITY", ["2 900 000 осіб"]))
    )
    assert identity.fold("2 900 000 осіб", "QUANTITY") == "2.9 мільйона осіб"
    assert identity.fold("2,9 млн осіб", "QUANTITY") == "2.9 мільйона осіб"
    assert identity.fold("2,8 млн осіб", "QUANTITY") == "2,8 млн осіб"


def test_a_value_the_corpus_never_states_folds_onto_nothing():
    identity = SurfaceIdentity(_ledger(_entity("717 тисяч осіб", "QUANTITY")))
    assert identity.fold("5 млн осіб", "QUANTITY") == "5 млн осіб"


def test_only_value_types_are_keyed_at_all():
    # A NAME is folded by the alias map and the overlay, never by a value key: those are the
    # identities a corpus states or a reviewer accepts, not ones this gate computes.
    identity = SurfaceIdentity(_ledger(_entity("Київ", "LOC"), _entity("2021 року", "DATE")))
    assert identity.n_values == 1
    assert identity.fold("Київ", "LOC") == "Київ"


# --- the reading side: a refusal labelled from a signal that survives inflection -----------------


def _refused(item_id: str, answer: str, contains: float = 0.0, **overrides) -> dict:
    row = {
        "item_id": item_id,
        "status": common.ONTOLOGY_VIOLATION,
        "contains": contains,
        "objective_score": 0.2,
        "answer_preview": answer,
        "validation_classes": ["max_cardinality"],
    }
    row.update(overrides)
    return row


def test_a_correct_short_answer_to_an_inflected_reference_is_no_longer_a_catch():
    # The one "catch" the heavy run recorded: `contains` has no morphology, so a correct terse
    # answer to a reference in an oblique case read as wrong and its refusal read as a success.
    rows = [_refused("q0", "Вишивка.")]
    labels = label_refusals(rows, {"q0": "роботою вишивки"})
    assert labels["q0"].label == LABEL_FALSE_REJECTION
    assert labels["q0"].shipped_label == LABEL_CATCH
    assert SIGNAL_ANSWER_WITHIN_REFERENCE in labels["q0"].signals
    assert relabelled(labels) == ["q0"]


def test_an_inflected_reference_inside_a_longer_answer_fires_the_lemma_containment():
    rows = [_refused("q0", "Це була робота вишивки, яку виконували вручну.")]
    labels = label_refusals(rows, {"q0": "роботою вишивки"})
    assert labels["q0"].signals == (SIGNAL_CONTAINS_LEMMA,)
    assert labels["q0"].label == LABEL_FALSE_REJECTION


def test_a_genuinely_wrong_answer_stays_a_catch():
    rows = [_refused("q0", "Одеса.")]
    labels = label_refusals(rows, {"q0": "Львів"})
    assert labels["q0"].label == LABEL_CATCH
    assert labels["q0"].signals == ()
    assert relabelled(labels) == []


def test_the_shipped_surface_signal_still_decides_on_its_own():
    # Nothing is taken away: a refusal `contains` already called correct stays a false rejection,
    # and its shipped label is recorded beside the new one so the two readings are comparable.
    rows = [_refused("q0", "Лос-Анджелес є найбільшим містом", contains=1.0)]
    labels = label_refusals(rows, {"q0": "Лос-Анджелес"})
    assert labels["q0"].signals[0] == SIGNAL_CONTAINS
    assert labels["q0"].shipped_label == LABEL_FALSE_REJECTION
    assert not labels["q0"].relabelled


def test_without_a_reference_the_labelling_degrades_to_the_shipped_reading():
    # A comparison whose gold set has moved loses the added signals and nothing else: the label is
    # then exactly what `contains` alone said, never an invented one.
    rows = [_refused("q0", "Вишивка."), _refused("q1", "Львів", contains=1.0)]
    labels = label_refusals(rows, references=None)
    assert labels["q0"].label == LABEL_CATCH
    assert labels["q1"].label == LABEL_FALSE_REJECTION
    assert relabelled(labels) == []


def test_only_refused_cases_are_labelled():
    # A case the gate let through is scored by the run; labelling it here would invent a second
    # correctness verdict for it.
    rows = [_refused("q0", "Вишивка."), {"item_id": "q1", "status": common.OK, "contains": 1.0}]
    assert set(label_refusals(rows, {"q0": "вишивки", "q1": "щось"})) == {"q0"}


def test_the_per_class_verdict_reports_both_readings_of_the_same_rejections():
    # The verdict decides on the inflection-tolerant labels, and carries what the shipped proxy
    # alone said beside them -- so a run that re-labels its refusals stays comparable to the one
    # recorded before.
    from llb.eval.answer_validation.verdict import class_verdicts

    rows = [_refused("q0", "Вишивка."), _refused("q1", "Одеса.")]
    labels = label_refusals(rows, {"q0": "роботою вишивки", "q1": "Львів"})
    verdict = class_verdicts(rows, labels, [], 0.95)[0]
    assert (verdict["n_catches"], verdict["n_false_rejections"]) == (1, 1)
    assert (verdict["n_catches_contains"], verdict["n_false_rejections_contains"]) == (2, 0)
