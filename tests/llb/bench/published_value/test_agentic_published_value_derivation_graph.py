"""The design-wide walk over declared derivations, and what it lets a collecting walk decline to say.

Synthetic published values again, for the reason the declaration tests are: what is checked here is
whether a design's edges hold together and what a chain of them means when a measurement moves, and
neither question belongs to whichever study adopted the seam first.
"""

import pytest

from llb.bench.published_value.collection import CollectedRefusals
from llb.bench.published_value.derivation import ValueKey, published_key
from llb.bench.published_value.derivation_graph import derivation_graph
from tests.llb.bench._published_value_fixtures import (
    BOUNDARY,
    DERIVED,
    MEASURED,
    OTHER,
    STUDY,
    key,
    published_value,
    register_test_operations,
    value,
)


@pytest.fixture(autouse=True)
def _registered_test_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    register_test_operations(monkeypatch)


# --- the design's own edges -----------------------------------------------------------------------


def test_a_measurement_contributes_no_edge_to_the_graph():
    """The default is the right one: most published numbers are measured, and say so by silence."""
    measured = value(STUDY, 6, MEASURED)
    assert derivation_graph([measured]).sources_of(published_key(measured)) == ()


def test_an_edge_resolves_to_the_form_it_names_and_not_to_the_depth_it_shares():
    """One study can publish two numbers at a depth, so depth alone would be an ambiguous edge."""
    boundary = key(STUDY, 6, BOUNDARY)
    values = [
        value(STUDY, 6, MEASURED),
        value(STUDY, 6, BOUNDARY),
        value(OTHER, 6, DERIVED, boundary),
    ]
    assert derivation_graph(values).sources_of(key(OTHER, 6, DERIVED)) == (boundary,)


def test_a_declaration_naming_a_value_the_design_does_not_publish_is_refused():
    """Checkable against the design is the whole point: an edge nobody publishes resolves nothing."""
    values = [value(STUDY, 6, MEASURED), value(OTHER, 6, DERIVED, key(STUDY, 10, MEASURED))]
    with pytest.raises(ValueError, match="which this design does not publish"):
        derivation_graph(values)


def test_a_value_that_declares_itself_is_refused():
    values = [value(STUDY, 6, DERIVED, key(STUDY, 6, DERIVED))]
    with pytest.raises(ValueError, match="declares itself as its own source"):
        derivation_graph(values)


def test_a_cycle_of_declarations_is_refused():
    """Neither end of a cycle rests on a measurement, so no restatement could start anywhere in it."""
    values = [
        value(STUDY, 6, MEASURED, key(OTHER, 6, DERIVED)),
        value(OTHER, 6, DERIVED, key(STUDY, 6, MEASURED)),
    ]
    with pytest.raises(ValueError, match="form a cycle"):
        derivation_graph(values)


def test_two_published_values_claiming_one_identity_are_refused():
    """A declaration must resolve to ONE source, not to whichever duplicate the walk reached first."""
    values = [value(STUDY, 6, MEASURED), value(STUDY, 6, MEASURED)]
    with pytest.raises(ValueError, match="two published values claim this identity"):
        derivation_graph(values)


def test_the_walk_refuses_an_operation_the_registry_does_not_carry():
    """One walk reads both halves, so an unregistered arithmetic never reaches a value read."""
    named = {**value(OTHER, 6, DERIVED, key(STUDY, 6, MEASURED)), "operation": "a_quotient_maybe"}
    with pytest.raises(ValueError, match="which no registered re-derivation carries"):
        derivation_graph([value(STUDY, 6, MEASURED), named])


# --- the consequence marking --------------------------------------------------------------------


def _chain() -> tuple[CollectedRefusals, ValueKey, ValueKey, ValueKey]:
    """A two-step chain: a measurement, a figure derived from it, and one derived from THAT."""
    root = key(STUDY, 6, MEASURED)
    middle = key(OTHER, 6, DERIVED)
    top = key("a_third_study", 6, DERIVED)
    graph = derivation_graph(
        [published_value(root), published_value(middle, root), published_value(top, middle)]
    )
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
    left = key(STUDY, 6, MEASURED)
    right = key(OTHER, 10, MEASURED)
    derived = key("a_third_study", 6, DERIVED)
    collected = CollectedRefusals(
        derivations=derivation_graph(
            [
                published_value(left),
                published_value(right),
                published_value(derived, left, right),
            ]
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
