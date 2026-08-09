"""The declaration read in the other direction: against the function, not against the design.

`source_forms` / `stated_fields` / `reads_own_measurement` were checked against the DESIGN only, so
an operation whose body disagreed with its own declaration was a defect nothing could see until a
study adopted the arithmetic and published a number out of it. These tests drive the three shapes
that defect takes -- a body reaching past its declaration, a declaration listing an input the body
never reads, and arithmetic no registered design names at all -- plus the CI gate over the shipped
registry, which is the whole point of closing the loop.

Each shape is driven twice over, because a body reads its declaration along a PATH: once at a probe
set that misses the branch the read is on, and once at a set that takes it. That pair is what the
probe set is for, in both directions -- a declared input read only on a missed branch is refused as
over-declaration (a false refusal an author fixes by moving the probe), and an UNdeclared reach on a
missed branch is simply not seen, which is the direction that decides whether the check is worth
anything.
"""

from pathlib import Path

import pytest

from llb.bench.agentic_published_value_operation_audit import (
    operation_refusals,
    published_operations,
    report_operation_registry,
    unpublished_operations,
    validate_operation_registry,
)
from llb.bench.agentic_published_value_operation_probe import MEASURED_READ, declared_reads
from llb.bench.agentic_published_value_operations import (
    DERIVATION_OPERATIONS,
    OPERATION,
    OPERATION_TRIGGER_OVER_OWN_CAP_PEAK,
    DerivationInputs,
    DerivationOperation,
    DerivedValue,
    probe_point_named,
)
from llb.bench.agentic_published_value_registry import (
    KIND_CROSSOVER_RESTATEMENT,
    PUBLISHED_VALUE_DESIGNS,
    PublishedValueDesign,
)
from llb.core.paths import PROJECT_ROOT

FORM = "a_declared_form"
STATED = "a_stated_field"
OTHER_STATED = "a_field_the_design_never_states"

SOURCE_VALUE = 8.0
STATED_VALUE = 0.5
MEASURED_VALUE = 4.0
ANSWER = 1.0

# A body that branches on the source it was handed reads one thing below this and another above it,
# so a probe SET declaring a point on each side is what certifies the declaration on both paths.
BRANCH_THRESHOLD = 16.0
OTHER_SOURCE_VALUE = 32.0


def _operation(
    compute,
    *,
    source_forms: tuple[str, ...] = (FORM,),
    stated_fields: tuple[str, ...] = (),
    reads_own_measurement: bool = False,
    name: str = "an_operation",
    at: tuple[float, ...] = (SOURCE_VALUE,),
) -> DerivationOperation:
    """A registered arithmetic probed once per source value in `at`, answering only its declaration.

    The set is spelled as the source values it is checked at, because a body that branches branches
    on what it was handed -- which is the whole subject of a probe SET rather than a probe point.
    """
    return DerivationOperation(
        name=name,
        source_forms=source_forms,
        stated_fields=stated_fields,
        reads_own_measurement=reads_own_measurement,
        compute=compute,
        probes=tuple(
            DerivationInputs(
                sources=tuple(source for _ in source_forms),
                stated={field: STATED_VALUE for field in stated_fields},
                measured=MEASURED_VALUE if reads_own_measurement else None,
            )
            for source in at
        ),
    )


# --- a body that reaches past its declaration ----------------------------------------------------


def test_an_operation_reading_a_stated_field_it_did_not_declare_is_refused():
    """Today's failure is a `KeyError` in whichever reader got there first; this is the refusal."""

    def reads_more(inputs: DerivationInputs) -> DerivedValue:
        return DerivedValue(value=inputs.stated[OTHER_STATED])

    (refusal,) = operation_refusals(_operation(reads_more, stated_fields=(STATED,)))
    assert f"`{OTHER_STATED}`" in refusal
    assert "which its declaration does not carry" in refusal


def test_an_operation_reading_the_measurement_it_did_not_declare_is_refused():
    """`reads_own_measurement` is false, so the value's own aggregate is not an input it may use."""

    def reads_the_peak(inputs: DerivationInputs) -> DerivedValue:
        return DerivedValue(value=float(inputs.measured or 0.0))

    (refusal,) = operation_refusals(_operation(reads_the_peak))
    assert MEASURED_READ in refusal
    assert "which its declaration does not carry" in refusal


def test_an_operation_reading_a_source_it_did_not_declare_is_refused():
    """The probe answers exactly the declared arity, so a second source is not there to read."""

    def reads_two(inputs: DerivationInputs) -> DerivedValue:
        return DerivedValue(value=inputs.sources[0] + inputs.sources[1])

    (refusal,) = operation_refusals(_operation(reads_two))
    assert "reaches outside the inputs it declares (IndexError" in refusal


def test_an_operation_that_does_not_compute_at_its_own_probe_point_is_refused():
    """A probe that cannot run leaves the declaration unchecked, which is the state being fixed."""

    def refuses(_inputs: DerivationInputs) -> DerivedValue:
        raise ValueError("this arithmetic has a domain its probe point is outside of")

    (refusal,) = operation_refusals(_operation(refuses))
    assert "does not compute at the probe point it declares" in refusal


# --- a declaration listing what the body never reads ---------------------------------------------


@pytest.mark.parametrize(
    ("declaration", "unread"),
    [
        ({"source_forms": (FORM, FORM)}, "source 2"),
        ({"stated_fields": (STATED,)}, f"`{STATED}`"),
        ({"reads_own_measurement": True}, MEASURED_READ),
    ],
)
def test_an_operation_declaring_an_input_it_never_reads_is_refused(declaration, unread):
    """Over-declaration makes every adopting design state a number nothing then checks."""

    def reads_one(inputs: DerivationInputs) -> DerivedValue:
        return DerivedValue(value=float(inputs.sources[0]))

    (refusal,) = operation_refusals(_operation(reads_one, **declaration))
    assert unread in refusal
    assert "and never reads it" in refusal


def test_the_arity_check_asking_which_fields_were_supplied_does_not_count_as_reading_them():
    """`apply` asks `name not in stated` per declared field, and membership is not a read.

    Stated because it is the one way the over-declaration refusal could silently stop firing: if the
    probe counted that membership test, every declared field would look read before `compute` ran.
    """

    def ignores_the_field(inputs: DerivationInputs) -> DerivedValue:
        return DerivedValue(value=float(inputs.sources[0]))

    (refusal,) = operation_refusals(_operation(ignores_the_field, stated_fields=(STATED,)))
    assert f"declares the value's own stated `{STATED}` and never reads it" in refusal


def test_an_operation_whose_body_matches_its_declaration_raises_nothing():
    """The passing case, which is what every entry in the shipped registry has to look like."""

    def reads_everything(inputs: DerivationInputs) -> DerivedValue:
        return DerivedValue(
            value=float(inputs.sources[0]) * inputs.stated[STATED] / float(inputs.measured or 1.0)
        )

    operation = _operation(reads_everything, stated_fields=(STATED,), reads_own_measurement=True)
    assert operation_refusals(operation) == ()
    assert declared_reads(operation) == (
        f"source 1 (the declared {FORM})",
        f"the value's own stated `{STATED}`",
        MEASURED_READ,
    )


# --- a read that happens on one branch only ------------------------------------------------------


def _reads_the_declared_field_when_large(inputs: DerivationInputs) -> DerivedValue:
    """Reads a DECLARED stated field on one branch and not the other."""
    source = float(inputs.sources[0])
    if source < BRANCH_THRESHOLD:
        return DerivedValue(value=source)
    return DerivedValue(value=source * inputs.stated[STATED])


def _reads_an_undeclared_field_when_large(inputs: DerivationInputs) -> DerivedValue:
    """Reaches OUTSIDE its declaration on one branch and not the other."""
    source = float(inputs.sources[0])
    if source < BRANCH_THRESHOLD:
        return DerivedValue(value=source)
    return DerivedValue(value=source * inputs.stated[OTHER_STATED])


def test_a_declared_input_read_only_on_a_branch_no_point_takes_is_refused_as_over_declared():
    """The false refusal, which an author fixes by moving the probe rather than the arithmetic."""
    (refusal,) = operation_refusals(
        _operation(_reads_the_declared_field_when_large, stated_fields=(STATED,))
    )

    assert f"declares the value's own stated `{STATED}` and never reads it" in refusal
    assert "at any of its" not in refusal


def test_a_point_on_that_branch_is_what_makes_the_declaration_hold():
    """Union across the set: read at ONE point is read, so the declaration carries the field."""
    operation = _operation(
        _reads_the_declared_field_when_large,
        stated_fields=(STATED,),
        at=(SOURCE_VALUE, OTHER_SOURCE_VALUE),
    )

    assert operation_refusals(operation) == ()


def test_an_undeclared_reach_on_a_branch_no_point_takes_is_not_seen():
    """The residual the set makes DECLARABLE rather than removes, stated so it is not folklore."""
    assert operation_refusals(_operation(_reads_an_undeclared_field_when_large)) == ()


def test_a_point_on_that_branch_is_what_catches_the_undeclared_reach():
    """First undeclared reach across the set, named at the point that took the branch."""
    (refusal,) = operation_refusals(
        _operation(_reads_an_undeclared_field_when_large, at=(SOURCE_VALUE, OTHER_SOURCE_VALUE))
    )

    assert f"`{OTHER_STATED}`" in refusal
    assert f"at {probe_point_named(1)}" in refusal


def test_the_over_declaration_refusal_says_the_whole_set_missed_the_input():
    """A set of points that all miss a read reads differently from one point that missed it."""

    def reads_one(inputs: DerivationInputs) -> DerivedValue:
        return DerivedValue(value=float(inputs.sources[0]))

    (refusal,) = operation_refusals(
        _operation(reads_one, stated_fields=(STATED,), at=(SOURCE_VALUE, OTHER_SOURCE_VALUE))
    )

    assert "never reads it at any of its 2 points" in refusal


# --- a probe set that cannot check the declaration ------------------------------------------------


def test_a_probe_set_whose_points_cannot_differ_is_refused_at_registration():
    """Two equal points take one path, so the second states no branch the first did not."""
    with pytest.raises(ValueError, match="takes the same path through the body"):
        _operation(lambda _inputs: DerivedValue(value=ANSWER), at=(SOURCE_VALUE, SOURCE_VALUE))


def test_a_probe_set_differing_in_a_stated_field_alone_is_accepted():
    """ANY declared input, not the sources only -- a body may branch on what its design states."""
    operation = DerivationOperation(
        name="an_operation",
        source_forms=(FORM,),
        stated_fields=(STATED,),
        compute=lambda inputs: DerivedValue(value=inputs.stated[STATED]),
        probes=(
            DerivationInputs(sources=(SOURCE_VALUE,), stated={STATED: STATED_VALUE}),
            DerivationInputs(sources=(SOURCE_VALUE,), stated={STATED: STATED_VALUE + ANSWER}),
        ),
    )

    assert operation_refusals(operation) == (
        f"the `an_operation` operation declares source 1 (the declared {FORM}) and never reads it "
        "at any of its 2 points, so every design adopting it would carry a number for nothing and "
        "nothing would ever check that number",
    )


def test_an_operation_declaring_no_probe_point_at_all_is_refused_at_registration():
    """A set of nothing is the state the probe was made required to rule out."""
    with pytest.raises(ValueError, match="declares no probe point"):
        _operation(lambda _inputs: DerivedValue(value=ANSWER), at=())


def test_the_point_that_does_not_answer_the_declaration_is_named_by_position():
    """Which point to fix, not that some point is wrong -- the points are otherwise alike."""
    with pytest.raises(ValueError, match=f"the {probe_point_named(1)} it is checked at"):
        DerivationOperation(
            name="an_operation",
            source_forms=(FORM,),
            compute=lambda _inputs: DerivedValue(value=ANSWER),
            probes=(
                DerivationInputs(sources=(SOURCE_VALUE,), stated={}),
                DerivationInputs(sources=(SOURCE_VALUE, OTHER_SOURCE_VALUE), stated={}),
            ),
        )


@pytest.mark.parametrize(
    ("probe", "match"),
    [
        (DerivationInputs(sources=(), stated={STATED: STATED_VALUE}), "source_forms"),
        (DerivationInputs(sources=(SOURCE_VALUE,), stated={}), "stated_fields"),
        (
            DerivationInputs(
                sources=(SOURCE_VALUE,), stated={STATED: STATED_VALUE}, measured=MEASURED_VALUE
            ),
            "reads_own_measurement",
        ),
    ],
)
def test_a_probe_point_that_does_not_answer_the_declaration_is_refused_at_registration(
    probe, match
):
    """Answering MORE than the declaration is what would hide a read the declaration lacks."""
    with pytest.raises(ValueError, match=match):
        DerivationOperation(
            name="an_operation",
            source_forms=(FORM,),
            stated_fields=(STATED,),
            compute=lambda _inputs: DerivedValue(value=ANSWER),
            probes=(probe,),
        )


# --- arithmetic no design names ------------------------------------------------------------------


def _registered(monkeypatch, values: list[dict[str, object]]) -> Path:
    """One registered design publishing exactly these values, read out of a file that exists."""
    design = PublishedValueDesign(
        design_path="samples/benchmarks/agentic_compact_crossover_restatement_design.json",
        published_values=lambda _path: values,
        cited_artifacts=lambda _path: [],
        validate_published_values=lambda _path, *, root, data_dir: None,
    )
    monkeypatch.setattr(
        "llb.bench.agentic_published_value_operation_audit.PUBLISHED_VALUE_DESIGNS",
        {"a_study": design},
    )
    return PROJECT_ROOT


def test_a_registered_operation_no_design_names_is_refused(monkeypatch):
    """Arithmetic nobody exercises is where a wrong quotient sits until the first study adopts it."""
    design_root = _registered(monkeypatch, [{"form": "a_measured_form"}])

    assert unpublished_operations(design_root) == tuple(sorted(DERIVATION_OPERATIONS))
    with pytest.raises(ValueError, match="no registered design names it"):
        validate_operation_registry(design_root=design_root)


def test_the_walk_names_which_designs_name_each_arithmetic(monkeypatch):
    """Named rather than counted, so a refusal can say whose design it is about."""
    design_root = _registered(
        monkeypatch, [{OPERATION: OPERATION_TRIGGER_OVER_OWN_CAP_PEAK}, {OPERATION: 7}]
    )

    assert published_operations(design_root) == {OPERATION_TRIGGER_OVER_OWN_CAP_PEAK: ["a_study"]}
    assert unpublished_operations(design_root) == ()


# --- the shipped registry ------------------------------------------------------------------------


def test_every_registered_operation_agrees_with_its_own_declaration_in_ci():
    """The CI gate the self-check buys: no run on the host, no per-operation test to remember."""
    assert validate_operation_registry(design_root=PROJECT_ROOT) == [
        OPERATION_TRIGGER_OVER_OWN_CAP_PEAK
    ]


def test_the_self_check_names_what_it_exercised_rather_than_only_passing():
    """A self-check that exercised nothing passes exactly like one that exercised every entry."""
    report = report_operation_registry(design_root=PROJECT_ROOT)

    assert report.checked == tuple(sorted(DERIVATION_OPERATIONS))
    assert report.refusals == ()


def test_the_shipped_design_registry_reads_its_own_published_values():
    """The walk above is only as good as each entry's reader, so the real one is exercised too."""
    design = PUBLISHED_VALUE_DESIGNS[KIND_CROSSOVER_RESTATEMENT]
    values = design.published_values(PROJECT_ROOT / design.design_path)

    assert {value.get(OPERATION) for value in values if OPERATION in value} == {
        OPERATION_TRIGGER_OVER_OWN_CAP_PEAK
    }
