"""The committed axiom set is real Turtle, and its two forms cannot drift apart."""

import pytest

from llb.prep.ontology.axioms.constants import AXIOM_KINDS
from llb.prep.ontology.axioms.deserialize import AxiomFileError, load_turtle
from llb.prep.ontology.axioms.loader import load_axioms, read_header, save_axioms
from llb.prep.ontology.axioms.models import Axiom, AxiomSet
from llb.prep.ontology.axioms.serialize import axiom_turtle, dump_turtle
from llb.prep.ontology.axioms.turtle import TurtleError, parse_turtle
from llb.prep.ontology.axioms.vocab import slug

from tests.llb.prep.ontology.axioms.conftest import CANDIDATE_JSON, CANDIDATE_TURTLE


def test_every_axiom_class_is_represented(axiom_set: AxiomSet) -> None:
    assert {axiom.kind for axiom in axiom_set.axioms} == set(AXIOM_KINDS)


def test_turtle_and_json_forms_describe_the_same_set(axiom_set: AxiomSet) -> None:
    assert axiom_set == load_axioms(CANDIDATE_JSON)


def test_turtle_round_trips_byte_for_byte(axiom_set: AxiomSet) -> None:
    text = CANDIDATE_TURTLE.read_text(encoding="utf-8")
    assert dump_turtle(axiom_set, read_header(text)) == text


def test_the_committed_set_is_unsigned(axiom_set: AxiomSet) -> None:
    # An axiom nobody signed gates nothing; the sign-off lane is what changes this.
    assert axiom_set.signed == []


def test_every_axiom_carries_a_reviewable_gloss(axiom_set: AxiomSet) -> None:
    assert all(axiom.gloss for axiom in axiom_set.axioms)


def test_signature_survives_a_round_trip(axiom_set: AxiomSet, tmp_path) -> None:
    signed = axiom_set.axioms[0].model_copy(
        update={"signed_by": "reviewer", "signed_on": "2026-08-25"}
    )
    target = tmp_path / "signed.ttl"
    save_axioms(AxiomSet(version="test", axioms=[signed]), target, ["header"])
    reloaded = load_axioms(target)
    assert reloaded.axioms[0].signed
    assert reloaded.signed[0].signed_by == "reviewer"


def test_reader_understands_collections_and_blank_nodes() -> None:
    multi = Axiom(
        axiom_id="domain-multi", kind="domain", relation="автор", entity_types=["PERSON", "ORG"]
    )
    text = dump_turtle(AxiomSet(version="t", axioms=[multi]))
    assert "owl:unionOf" in text
    assert load_turtle(text).axioms[0].entity_types == ["PERSON", "ORG"]


def test_restriction_round_trips_its_bound() -> None:
    bounded = Axiom(axiom_id="max-3", kind="max_cardinality", relation="є", max_count=3)
    text = dump_turtle(AxiomSet(version="t", axioms=[bounded]))
    assert load_turtle(text).axioms[0].max_count == 3


def test_a_constraint_without_an_annotation_is_ignored() -> None:
    # An axiom with no id cannot be accepted, rejected, or cited in a violation.
    text = dump_turtle(
        AxiomSet(version="t", axioms=[Axiom(axiom_id="a", kind="irreflexive", relation="є")])
    )
    stripped = text.split("[] a owl:Axiom")[0]
    assert load_turtle(stripped).axioms == []


def test_unparseable_turtle_is_an_error() -> None:
    with pytest.raises(TurtleError):
        parse_turtle("@prefix ex: <http://e/> .\nex:a ex:p")


def test_undeclared_prefix_is_an_error() -> None:
    with pytest.raises(TurtleError):
        parse_turtle("ex:a a ex:B .")


def test_an_annotated_predicate_that_expresses_no_class_is_an_error() -> None:
    text = (
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
        "@prefix llb: <https://loc-lm-bench.example/ontology/uk#> .\n"
        "[] a owl:Axiom ; owl:annotatedSource llb:PERSON ; "
        'owl:annotatedProperty rdfs:seeAlso ; owl:annotatedTarget llb:ORG ; rdfs:label "x" .\n'
    )
    with pytest.raises(AxiomFileError):
        load_turtle(text)


def test_slug_is_ascii_and_stable() -> None:
    assert slug("є учасницею") == "ie_uchasnytseiu"
    assert slug("межує з").isascii()


def test_axiom_turtle_renders_one_axiom_alone(axiom_set: AxiomSet) -> None:
    rendered = axiom_turtle(axiom_set.axioms[0])
    assert "owl:Axiom" in rendered and axiom_set.axioms[0].axiom_id in rendered
