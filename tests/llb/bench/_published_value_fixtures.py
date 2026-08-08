"""Synthetic published values and throwaway arithmetic, shared by the derivation tests.

Synthetic rather than the restatement's own rows, for the reason the pointer tests are: the rules
under test are study-agnostic, so a failure should name the derivation and not whichever study first
adopted it. The operations are registered per test for the same reason -- what is being checked is
the DECLARATION, which must not depend on which arithmetic a real study happens to ship.
"""

import pytest

from llb.bench.agentic_published_value_derivation import DERIVED_FROM, ValueKey
from llb.bench.agentic_published_value_operations import (
    DERIVATION_OPERATIONS,
    OPERATION,
    DerivationInputs,
    DerivationOperation,
    DerivedValue,
)

MEASURED = "measured_form"
DERIVED = "derived_form"
BOUNDARY = "boundary_form"
STUDY = "a_study"
OTHER = "another_study"

# The shapes of arithmetic these synthetic values are declared with, one registered operation each.
TEST_SHAPES = ((MEASURED,), (DERIVED,), (BOUNDARY,), (MEASURED, MEASURED), (MEASURED, DERIVED))

# The point the registry self-check would call a throwaway operation at; any number computes.
PROBE_SOURCE = 1.0


def summed(inputs: DerivationInputs) -> DerivedValue:
    """Arithmetic simple enough to disappear: what is asserted is which inputs reached it."""
    return DerivedValue(value=sum(inputs.sources))


def operation_name(forms: tuple[str, ...]) -> str:
    return f"test_sum_of_{'_and_'.join(forms)}"


def register_test_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Add the throwaway operations to the registry for one test, and no longer."""
    for forms in TEST_SHAPES:
        monkeypatch.setitem(
            DERIVATION_OPERATIONS,
            operation_name(forms),
            DerivationOperation(
                name=operation_name(forms),
                source_forms=forms,
                compute=summed,
                probe=DerivationInputs(sources=tuple(PROBE_SOURCE for _ in forms), stated={}),
            ),
        )


def key(kind: str, depth: int, form: str) -> ValueKey:
    return ValueKey(study_kind=kind, depth=depth, form=form)


def value(kind: str, depth: int, form: str, *sources: object) -> dict[str, object]:
    """One published value, derived from what it is handed and measured when handed nothing.

    A source is a `ValueKey` for the ordinary cases and anything at all for the malformed ones, so a
    badly declared edge is stated in the test as the JSON a design would actually carry. The
    operation follows the shape of what it is handed, because a design states both halves or neither.
    """
    published: dict[str, object] = {"study_kind": kind, "depth": depth, "form": form}
    if sources:
        published[DERIVED_FROM] = [_declaration(source) for source in sources]
        published[OPERATION] = operation_name(
            tuple(source.form if isinstance(source, ValueKey) else MEASURED for source in sources)
        )
    return published


def published_value(named: ValueKey, *sources: object) -> dict[str, object]:
    """The same, for a value already named by its key."""
    return value(named.study_kind, named.depth, named.form, *sources)


def _declaration(source: object) -> object:
    if not isinstance(source, ValueKey):
        return source
    return {"study_kind": source.study_kind, "depth": source.depth, "form": source.form}
