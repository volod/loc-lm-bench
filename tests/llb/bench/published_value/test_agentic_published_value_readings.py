"""The reading a design names beside a derived value's registered arithmetic."""

import pytest

from llb.bench.published_value.readings import (
    READING,
    READING_POINT_TOLERANCE,
    READING_ROUNDED_BAND,
    READING_UPPER_BOUND,
    required_reading,
)


def _reading(name: str, **statement: object):
    published = {READING: name, **statement}
    return required_reading(published, where="a_study depth 6 derived_form")


def test_the_rounded_band_uses_one_rule_for_resolution_and_restatement_membership():
    reading = _reading(READING_ROUNDED_BAND, published_band=[0.85, 0.92], band_decimals=2)

    assert reading.resolves([0.8451, 0.918]).holds is True
    assert reading.resolves([0.844, 0.918]).holds is False
    # Round before comparing in both directions: the raw value is below the edge, the quoted one is
    # the published edge, so it still supports the statement.
    result = reading.read(0.849)
    assert result.holds is True
    assert "inside the published 0.85-0.92x band" in result.phrase


@pytest.mark.parametrize(
    ("name", "statement", "inside", "outside"),
    [
        (
            READING_POINT_TOLERANCE,
            {"published_value": 10.0, "absolute_tolerance": 0.5},
            10.5,
            10.5001,
        ),
        (READING_UPPER_BOUND, {"published_upper_bound": 10.0}, 10.0, 10.0001),
    ],
)
def test_a_non_band_statement_inherits_resolution_and_restatement_without_a_reader_edit(
    name, statement, inside, outside
):
    reading = _reading(name, **statement)

    assert reading.resolves([inside]).holds is True
    assert reading.read(inside).holds is True
    assert reading.read(outside).holds is False


@pytest.mark.parametrize("named", ["compare_somehow", 7])
def test_a_reading_the_registry_does_not_carry_is_refused(named):
    with pytest.raises(ValueError, match="no registered published-value reading carries"):
        required_reading({READING: named}, where="a_study depth 6 derived_form")


def test_a_derived_value_that_does_not_name_its_reading_is_refused():
    with pytest.raises(ValueError, match=f"must name the `{READING}`"):
        required_reading({}, where="a_study depth 6 derived_form")
