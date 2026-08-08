"""What a published value declares it is computed out of, and what that lets a walk decline to say.

Driven on SYNTHETIC published values rather than on the restatement's, for the reason the pointer
tests are: the rule being checked is study-agnostic, so a failure here should name the derivation and
not whichever study first adopted it. The one thing the committed design owes this module -- that it
declares the edge its resolver used to hardcode -- is asserted beside that design instead.
"""

import pytest

from llb.bench.agentic_published_value_collection import CollectedRefusals
from llb.bench.agentic_published_value_derivation import (
    DERIVED_FROM,
    ValueKey,
    declared_sources,
    derivation_graph,
    derived_source_of_form,
    published_key,
)

MEASURED = "measured_form"
DERIVED = "derived_form"
STUDY = "a_study"
OTHER = "another_study"


def _key(kind: str, depth: int, form: str) -> ValueKey:
    return ValueKey(study_kind=kind, depth=depth, form=form)


def _value(kind: str, depth: int, form: str, *sources: object) -> dict[str, object]:
    """One published value, derived from what it is handed and measured when handed nothing.

    A source is a `ValueKey` for the ordinary cases and anything at all for the malformed ones, so a
    badly declared edge is stated in the test as the JSON a design would actually carry.
    """
    value: dict[str, object] = {"study_kind": kind, "depth": depth, "form": form}
    if sources:
        value[DERIVED_FROM] = [_declaration(source) for source in sources]
    return value


def _declaration(source: object) -> object:
    if not isinstance(source, ValueKey):
        return source
    return {"study_kind": source.study_kind, "depth": source.depth, "form": source.form}


def _published(key: ValueKey, *sources: object) -> dict[str, object]:
    """The same, for a value already named by its key."""
    return _value(key.study_kind, key.depth, key.form, *sources)


# --- the declaration ----------------------------------------------------------------------------


def test_a_value_that_declares_nothing_is_a_measurement():
    """The default is the right one: most published numbers are measured, and say so by silence."""
    measured = _value(STUDY, 6, MEASURED)
    assert declared_sources(measured) == ()
    assert derivation_graph([measured]).sources_of(published_key(measured)) == ()


def test_the_form_is_part_of_the_identity_a_declaration_names():
    """One study can publish two numbers at a depth, so depth alone would be an ambiguous edge."""
    guard = _key(STUDY, 6, MEASURED)
    boundary = _key(STUDY, 6, "boundary_form")
    values = [
        _value(STUDY, 6, MEASURED),
        _value(STUDY, 6, "boundary_form"),
        _value(OTHER, 6, DERIVED, boundary),
    ]
    graph = derivation_graph(values)
    assert graph.sources_of(_key(OTHER, 6, DERIVED)) == (boundary,)
    assert guard not in graph.sources_of(_key(OTHER, 6, DERIVED))


def test_a_declaration_naming_a_value_the_design_does_not_publish_is_refused():
    """Checkable against the design is the whole point: an edge nobody publishes resolves nothing."""
    values = [_value(STUDY, 6, MEASURED), _value(OTHER, 6, DERIVED, _key(STUDY, 10, MEASURED))]
    with pytest.raises(ValueError, match="which this design does not publish"):
        derivation_graph(values)


def test_a_value_that_declares_itself_is_refused():
    values = [_value(STUDY, 6, DERIVED, _key(STUDY, 6, DERIVED))]
    with pytest.raises(ValueError, match="declares itself as its own source"):
        derivation_graph(values)


def test_a_cycle_of_declarations_is_refused():
    """Neither end of a cycle rests on a measurement, so no restatement could start anywhere in it."""
    values = [
        _value(STUDY, 6, MEASURED, _key(OTHER, 6, DERIVED)),
        _value(OTHER, 6, DERIVED, _key(STUDY, 6, MEASURED)),
    ]
    with pytest.raises(ValueError, match="form a cycle"):
        derivation_graph(values)


def test_two_published_values_claiming_one_identity_are_refused():
    """A declaration must resolve to ONE source, not to whichever duplicate the walk reached first."""
    values = [_value(STUDY, 6, MEASURED), _value(STUDY, 6, MEASURED)]
    with pytest.raises(ValueError, match="two published values claim this identity"):
        derivation_graph(values)


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
        declared_sources(_value(OTHER, 6, DERIVED, declared))


@pytest.mark.parametrize("declared", [[], {}, "the surface guard"])
def test_a_declaration_that_is_not_a_non_empty_list_of_sources_is_refused(declared):
    """A value that is measured says so by omitting the field, not by declaring an empty list."""
    with pytest.raises(ValueError, match="non-empty list of sources"):
        declared_sources({**_value(OTHER, 6, DERIVED), DERIVED_FROM: declared})


def test_a_row_with_no_readable_depth_is_refused_before_it_can_be_an_edge():
    with pytest.raises(ValueError, match="must state a numeric `depth`"):
        published_key({"study_kind": STUDY, "form": MEASURED})


# --- what the arithmetic may ask of a declaration -----------------------------------------------


def test_the_arithmetic_asks_for_its_shape_and_the_design_answers_with_the_edge():
    """A re-derivation knows it divides one measured value; it must not know which study's."""
    source = _key(STUDY, 6, MEASURED)
    assert derived_source_of_form(_value(OTHER, 6, DERIVED, source), MEASURED) == source


@pytest.mark.parametrize(
    "sources",
    [
        (),
        (_key(STUDY, 6, MEASURED), _key(STUDY, 10, MEASURED)),
    ],
)
def test_a_value_that_does_not_declare_exactly_one_source_of_the_asked_form_is_refused(sources):
    with pytest.raises(ValueError, match="nothing in the run can restate it"):
        derived_source_of_form(_value(OTHER, 6, DERIVED, *sources), MEASURED)


# --- the consequence marking --------------------------------------------------------------------


def _chain() -> tuple[CollectedRefusals, ValueKey, ValueKey, ValueKey]:
    """A two-step chain: a measurement, a figure derived from it, and one derived from THAT."""
    root = _key(STUDY, 6, MEASURED)
    middle = _key(OTHER, 6, DERIVED)
    top = _key("a_third_study", 6, DERIVED)
    graph = derivation_graph([_published(root), _published(middle, root), _published(top, middle)])
    return CollectedRefusals(derivations=graph), root, middle, top


def test_a_value_whose_sources_all_resolved_is_judged_rather_than_skipped():
    """The marking must stay narrow: a clean walk has to leave every value checkable."""
    collected, _root, middle, top = _chain()
    assert collected.rests_on_unresolved(middle) is False
    assert collected.rests_on_unresolved(top) is False
    assert collected.unjudged == []


def test_a_two_step_consequence_names_only_the_measurement_at_its_root():
    """The figure in between is a consequence too, and restating it would fix nothing."""
    collected, root, middle, top = _chain()
    collected.unresolvable("the aggregate no longer states it", key=root)

    assert collected.rests_on_unresolved(middle) is True
    assert collected.rests_on_unresolved(top) is True
    assert len(collected.unjudged) == 2
    for unjudged in collected.unjudged:
        assert root.label() in unjudged
        assert middle.label() not in unjudged.split(":")[1]
    assert len(collected.unresolved) == 1


def test_a_consequence_is_marked_not_judged_rather_than_counted_as_a_moved_number():
    """Two lists, so the count an operator reads is moved MEASUREMENTS and not their echoes."""
    collected, root, middle, _top = _chain()
    collected.unresolvable("the aggregate no longer states it", key=root)
    collected.rests_on_unresolved(middle)

    with pytest.raises(ValueError) as excinfo:
        collected.refuse(total=3)
    message = str(excinfo.value)
    assert "1/3 published values do not resolve" in message
    assert f"[not judged] {middle.label()}" in message


def test_a_value_derived_from_two_moved_measurements_names_both():
    """A derived figure can rest on more than one number, and each is its own design edit."""
    left = _key(STUDY, 6, MEASURED)
    right = _key(OTHER, 10, MEASURED)
    derived = _key("a_third_study", 6, DERIVED)
    collected = CollectedRefusals(
        derivations=derivation_graph(
            [_published(left), _published(right), _published(derived, left, right)]
        )
    )
    collected.unresolvable("moved", key=left)
    collected.unresolvable("moved", key=right)

    assert collected.rests_on_unresolved(derived) is True
    assert left.label() in collected.unjudged[0]
    assert right.label() in collected.unjudged[0]


def test_a_refusal_collected_without_an_identity_is_the_cause_of_nothing():
    """A key is what makes a refusal nameable as a cause; without one it is only a moved number."""
    collected, root, middle, _top = _chain()
    collected.collect(_raises)
    assert collected.unresolved == ["the aggregate no longer states it"]
    assert collected.rests_on_unresolved(middle) is False
    assert root not in collected.unresolved_keys


def _raises() -> float:
    raise ValueError("the aggregate no longer states it")
