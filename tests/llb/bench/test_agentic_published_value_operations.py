"""The arithmetic a design names, and the two readers that must both call it.

The registry is what turns "this number is derived from that one" into "and here is how", so the
tests that matter are the ones a resolver edit used to be needed for: an operation nobody registered
is refused, an operation is computed over the inputs it declared it takes, and the SAME registered
function is what both design validation and the restated row run -- which is checked by swapping the
arithmetic and watching both readers change, since two modules carrying one quotient each is exactly
what would pass every other test in this file.
"""

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from llb.bench.agentic_memory_boundary_crossover import READING_BRACKETED
from llb.bench.agentic_memory_crossover_restatement_design import (
    load_restatement_design,
    published_crossovers,
    validate_restatement_design,
)
from llb.bench.agentic_memory_crossover_restatement_forms import crossover_row
from llb.bench.agentic_memory_crossover_restatement_reading import (
    BASIS_DERIVED,
    FORM_PORTABLE_RATIO,
)
from llb.bench.agentic_memory_fold_step_ladder import compaction_trigger_chars
from llb.bench.agentic_policy_change_audit import KIND_COLLAPSE
from llb.bench.agentic_published_value_operations import (
    DERIVATION_OPERATIONS,
    OPERATION_TRIGGER_OVER_OWN_CAP_PEAK,
    TERM_TRIGGER_CHARS,
    DerivationInputs,
    DerivationOperation,
    DerivedValue,
    registered_operation,
)

ROOT = Path(__file__).resolve().parents[3]
DESIGN_PATH = ROOT / "samples/benchmarks/agentic_compact_crossover_restatement_design.json"

GUARD = 14159.929807575942
PEAK = 16384.0
SHARE = 0.5

# What the swapped arithmetic answers, chosen so no real quotient could produce it by accident.
SWAPPED_VALUE = 3.25
SWAPPED_TRIGGER = 99.0


def _trigger() -> DerivationOperation:
    return DERIVATION_OPERATIONS[OPERATION_TRIGGER_OVER_OWN_CAP_PEAK]


def _inputs(**overrides: object) -> DerivationInputs:
    fields: dict[str, object] = {
        "sources": (GUARD,),
        "stated": {"compact_share": SHARE},
        "measured": PEAK,
    }
    fields.update(overrides)
    return DerivationInputs(**cast(dict, fields))


# --- the registered trigger ratio ----------------------------------------------------------------


def test_the_trigger_ratio_is_the_runtimes_own_arithmetic_over_the_declared_guard():
    """The operation IS the quotient the two readers used to carry a copy of each."""
    derived = _trigger().compute(_inputs())
    trigger = compaction_trigger_chars(int(GUARD), SHARE)
    assert derived.value == pytest.approx(trigger / PEAK)
    assert derived.terms[TERM_TRIGGER_CHARS] == float(trigger)


def test_the_operation_names_its_intermediate_so_a_row_reports_it_without_recomputing_it():
    """A reported trigger that a reader re-derived is a second arithmetic waiting to disagree."""
    assert TERM_TRIGGER_CHARS in _trigger().compute(_inputs()).terms


@pytest.mark.parametrize("peak", [0.0, -1.0])
def test_a_ratio_over_a_non_positive_peak_is_refused(peak):
    with pytest.raises(ValueError, match="the peak must be positive"):
        _trigger().compute(_inputs(measured=peak))


# --- the registry --------------------------------------------------------------------------------


@pytest.mark.parametrize("named", ["a_quotient_maybe", None, 7])
def test_an_operation_the_registry_does_not_carry_is_refused(named):
    """A design may not name arithmetic nothing implements: the number would be unreproducible."""
    with pytest.raises(ValueError, match="which no registered re-derivation carries"):
        registered_operation(named, where="a_study depth 6 a_form")


@pytest.mark.parametrize(
    ("call", "match"),
    [
        ({"sources": (GUARD, GUARD)}, "is computed over 1 declared source"),
        ({"stated": {}}, "reads this value's own compact_share"),
        ({"measured": None}, "computed against the figure this value's own aggregate measured"),
    ],
)
def test_an_operation_called_with_inputs_it_did_not_declare_is_refused(call, match):
    """Checked once in `apply`, so every registered function is written as the arithmetic alone."""
    inputs = _inputs(**call)
    with pytest.raises(ValueError, match=match):
        _trigger().apply(
            inputs.sources, inputs.stated, measured=inputs.measured, where="a_study depth 6 a_form"
        )


# --- one arithmetic, two readers -----------------------------------------------------------------


def _swapped(_inputs: DerivationInputs) -> DerivedValue:
    return DerivedValue(value=SWAPPED_VALUE, terms={TERM_TRIGGER_CHARS: SWAPPED_TRIGGER})


@pytest.fixture()
def swapped_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the registered arithmetic, leaving both readers' own code untouched."""
    operation = _trigger()
    monkeypatch.setitem(
        DERIVATION_OPERATIONS,
        OPERATION_TRIGGER_OVER_OWN_CAP_PEAK,
        DerivationOperation(
            name=operation.name,
            source_forms=operation.source_forms,
            stated_fields=operation.stated_fields,
            reads_own_measurement=operation.reads_own_measurement,
            compute=_swapped,
        ),
    )


def test_the_design_validation_re_derives_the_band_through_the_registered_operation(
    swapped_trigger: None,
):
    """Swapping the arithmetic moves the band the committed design has to resolve to."""
    with pytest.raises(ValueError, match="the band was transcribed rather than resolved"):
        validate_restatement_design(load_restatement_design(DESIGN_PATH), root=ROOT)


def test_the_restated_row_re_derives_its_ratio_through_the_same_registered_operation(
    swapped_trigger: None,
):
    """The other reader, on the same swap: one design statement, one arithmetic, two callers."""
    row = crossover_row(_published_ratio(), {}, _audit(), _restated_surface())
    assert row["basis"] == BASIS_DERIVED
    assert row["restated_value"] == SWAPPED_VALUE
    assert row["restated_trigger_chars"] == int(SWAPPED_TRIGGER)


def _published_ratio() -> dict[str, object]:
    """The committed design's own derived value at depth 6, declaration and all."""
    return deepcopy(
        next(
            row
            for row in published_crossovers(load_restatement_design(DESIGN_PATH))
            if row["form"] == FORM_PORTABLE_RATIO and row["depth"] == 6
        )
    )


def _audit() -> dict[str, object]:
    """Every collapse cell bound-invariant, which is the case the ratio is restated in anyway."""
    return {"per_study": {KIND_COLLAPSE: []}}


def _restated_surface() -> list[dict[str, object]]:
    return [
        {
            "depth": 6,
            "reading": READING_BRACKETED,
            "crossover_max_prompt_chars": GUARD,
            "cap_peak_prompt_chars": int(PEAK),
        }
    ]
