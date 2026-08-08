"""What one published value declares it is computed out of, and the arithmetic it names over that.

Driven on SYNTHETIC published values rather than on the restatement's, for the reason the pointer
tests are: the rules being checked are study-agnostic, so a failure here should name the derivation
and not whichever study first adopted it. The design-wide walk over these declarations is exercised
beside the graph, and the one thing the committed design owes this module -- that it declares both
the edge and the arithmetic its readers used to hardcode -- is asserted beside that design.
"""

import pytest

from llb.bench.agentic_published_value_derivation import (
    DERIVED_FROM,
    declared_derivation,
    declared_sources,
    published_key,
    required_derivation,
)
from llb.bench.agentic_published_value_operations import (
    DERIVATION_OPERATIONS,
    OPERATION,
    DerivationOperation,
)
from tests.llb.bench._published_value_fixtures import (
    BOUNDARY,
    DERIVED,
    MEASURED,
    OTHER,
    STUDY,
    key,
    operation_name,
    register_test_operations,
    summed,
    value,
)


@pytest.fixture(autouse=True)
def _registered_test_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    register_test_operations(monkeypatch)


# --- the declared sources -------------------------------------------------------------------------


def test_a_value_that_declares_nothing_is_a_measurement():
    """The default is the right one: most published numbers are measured, and say so by silence."""
    measured = value(STUDY, 6, MEASURED)
    assert declared_sources(measured) == ()
    assert declared_derivation(measured) is None


def test_the_form_is_part_of_the_identity_a_declaration_names():
    """One study can publish two numbers at a depth, so depth alone would be an ambiguous edge."""
    boundary = key(STUDY, 6, BOUNDARY)
    derivation = required_derivation(value(OTHER, 6, DERIVED, boundary))
    assert derivation.sources == (boundary,)
    assert key(STUDY, 6, MEASURED) not in derivation.sources


@pytest.mark.parametrize(
    ("declared", "match"),
    [
        ({"study_kind": STUDY, "depth": 6}, "must name the `study_kind`, `depth`, and `form`"),
        (
            {"study_kind": STUDY, "form": MEASURED},
            "must name the `study_kind`, `depth`, and `form`",
        ),
        ({"study_kind": STUDY, "depth": 6.5, "form": MEASURED}, "must name the `study_kind`"),
        ("the surface guard", "must name the `study_kind`, `depth`, and `form`"),
    ],
)
def test_a_declaration_entry_that_does_not_name_a_value_completely_is_refused(declared, match):
    with pytest.raises(ValueError, match=match):
        declared_sources(value(OTHER, 6, DERIVED, declared))


@pytest.mark.parametrize("declared", [[], {}, "the surface guard"])
def test_a_declaration_that_is_not_a_non_empty_list_of_sources_is_refused(declared):
    """A value that is measured says so by omitting the field, not by declaring an empty list."""
    with pytest.raises(ValueError, match="non-empty list of sources"):
        declared_sources({**value(OTHER, 6, DERIVED), DERIVED_FROM: declared})


def test_a_row_with_no_readable_depth_is_refused_before_it_can_be_an_edge():
    with pytest.raises(ValueError, match="must state a numeric `depth`"):
        published_key({"study_kind": STUDY, "form": MEASURED})


# --- the declared arithmetic -----------------------------------------------------------------------


def test_a_declared_value_is_re_derived_through_the_operation_the_design_names():
    """The point of the seam: the caller supplies values, the DESIGN supplies the arithmetic."""
    derivation = required_derivation(value(OTHER, 6, DERIVED, key(STUDY, 6, MEASURED)))
    assert derivation.compute((7.0,)).value == 7.0


def test_a_two_input_operation_is_computed_over_both_declared_sources_in_order():
    """A derived number can rest on two published values, and nothing about that is a special case."""
    left, right = key(STUDY, 6, MEASURED), key(OTHER, 10, MEASURED)
    derivation = required_derivation(value("a_third_study", 6, DERIVED, left, right))
    assert derivation.sources == (left, right)
    assert derivation.compute((3.0, 4.0)).value == 7.0


def test_an_operation_the_registry_does_not_carry_is_refused():
    """A design naming arithmetic nothing implements publishes a number no reader can reproduce."""
    named = {**value(OTHER, 6, DERIVED, key(STUDY, 6, MEASURED)), OPERATION: "a_quotient_maybe"}
    with pytest.raises(ValueError, match="which no registered re-derivation carries"):
        declared_derivation(named)


def test_a_value_that_declares_sources_but_no_operation_is_refused():
    """Half a derivation: the readers would each carry their own arithmetic again."""
    derived = value(OTHER, 6, DERIVED, key(STUDY, 6, MEASURED))
    del derived[OPERATION]
    with pytest.raises(ValueError, match=f"names no `{OPERATION}`"):
        declared_derivation(derived)


def test_an_operation_over_no_declared_sources_is_refused():
    """The other half: arithmetic over nothing names no measurement a restatement could start from."""
    named = {**value(OTHER, 6, DERIVED), OPERATION: operation_name((MEASURED,))}
    with pytest.raises(ValueError, match="names none"):
        declared_derivation(named)


@pytest.mark.parametrize(
    ("sources", "match"),
    [
        ((key(STUDY, 6, MEASURED), key(STUDY, 10, MEASURED)), "operation is computed over 1"),
        ((key(STUDY, 6, DERIVED),), "takes a measured_form as source 1"),
    ],
)
def test_a_declaration_that_is_not_the_shape_its_operation_takes_is_refused(sources, match):
    """The two halves are checked against each other, so neither can drift from the other."""
    named = {**value(OTHER, 6, DERIVED, *sources), OPERATION: operation_name((MEASURED,))}
    with pytest.raises(ValueError, match=match):
        declared_derivation(named)


def test_an_operation_reading_a_field_the_design_does_not_state_is_refused(
    monkeypatch: pytest.MonkeyPatch,
):
    """A stated operand is part of the declaration too, so a design missing one states nothing."""
    monkeypatch.setitem(
        DERIVATION_OPERATIONS,
        "test_needs_a_share",
        DerivationOperation(
            name="test_needs_a_share",
            source_forms=(MEASURED,),
            stated_fields=("compact_share",),
            compute=summed,
        ),
    )
    named = {
        **value(OTHER, 6, DERIVED, key(STUDY, 6, MEASURED)),
        OPERATION: "test_needs_a_share",
    }
    with pytest.raises(ValueError, match="must state it numerically"):
        declared_derivation(named)


def test_a_reader_asks_for_its_shape_and_the_design_answers_with_the_edge():
    """A row naming which study its figure came from knows the FORM, never the study."""
    source = key(STUDY, 6, MEASURED)
    assert required_derivation(value(OTHER, 6, DERIVED, source)).source_of_form(MEASURED) == source


@pytest.mark.parametrize(
    "sources",
    [
        (),
        (key(STUDY, 6, MEASURED), key(STUDY, 10, MEASURED)),
    ],
)
def test_a_value_that_does_not_declare_exactly_one_source_of_the_asked_form_is_refused(sources):
    with pytest.raises(ValueError, match="nothing in the run can restate it"):
        required_derivation(value(OTHER, 6, DERIVED, *sources)).source_of_form(MEASURED)
