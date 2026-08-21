"""How a re-derived value is read against the statement its design publishes.

Arithmetic answers what number a derivation produces.  A reading answers the separate question a
publisher and a restatement both need: does that number still support the statement in the design?
The design names that rule beside its operation, and both readers bind it through this registry.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

# The field a derived published value uses to name its comparison rule.
READING = "reading"

READING_ROUNDED_BAND = "rounded_extent_and_membership"
READING_POINT_TOLERANCE = "within_absolute_tolerance"
READING_UPPER_BOUND = "at_or_below_upper_bound"

CRITERION_ROUNDED_BAND = "restated_value_supports_the_published_rounded_band"
CRITERION_POINT_TOLERANCE = "restated_value_stays_within_the_published_tolerance"
CRITERION_UPPER_BOUND = "restated_value_stays_at_or_below_the_published_bound"

_PUBLISHED_BAND = "published_band"
_BAND_DECIMALS = "band_decimals"
_PUBLISHED_VALUE = "published_value"
_ABSOLUTE_TOLERANCE = "absolute_tolerance"
_PUBLISHED_UPPER_BOUND = "published_upper_bound"


@dataclass(frozen=True, slots=True)
class ReadingResult:
    """One comparison and the phrase a report can show without reconstructing its rule."""

    holds: bool
    phrase: str


@dataclass(frozen=True, slots=True)
class PublishedValueReading:
    """A registered reading bound to one design row's validated published statement."""

    name: str
    criterion: str
    statement_kind: str
    statement: tuple[object, ...]
    resolve: Callable[[tuple[object, ...], Sequence[float]], ReadingResult]
    compare: Callable[[tuple[object, ...], float], ReadingResult]

    @property
    def identity(self) -> tuple[object, ...]:
        """A stable grouping key for rows that jointly publish the same statement."""
        return (self.name, *self.statement)

    def resolves(self, values: Sequence[float]) -> ReadingResult:
        """Whether the design's own re-derived values produce its published statement."""
        return self.resolve(self.statement, values)

    def read(self, value: float) -> ReadingResult:
        """Whether one newly re-derived value still supports the published statement."""
        return self.compare(self.statement, value)


@dataclass(frozen=True, slots=True)
class RegisteredReading:
    """One comparison rule and the validator for the statement fields it consumes."""

    name: str
    criterion: str
    statement_kind: str
    statement: Callable[[Mapping[str, object], str], tuple[object, ...]]
    resolve: Callable[[tuple[object, ...], Sequence[float]], ReadingResult]
    compare: Callable[[tuple[object, ...], float], ReadingResult]

    def bind(self, published: Mapping[str, object], *, where: str) -> PublishedValueReading:
        return PublishedValueReading(
            name=self.name,
            criterion=self.criterion,
            statement_kind=self.statement_kind,
            statement=self.statement(published, where),
            resolve=self.resolve,
            compare=self.compare,
        )


def _number(value: object, field: str, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: `{field}` must be numeric for its published-value reading")
    return float(value)


def _band_statement(published: Mapping[str, object], where: str) -> tuple[object, ...]:
    band = published.get(_PUBLISHED_BAND)
    if (
        not isinstance(band, list)
        or len(band) != 2
        or any(isinstance(edge, bool) or not isinstance(edge, (int, float)) for edge in band)
    ):
        raise ValueError(
            f"{where}: a rounded band reading needs `published_band` as an ascending pair of "
            "positive numeric edges"
        )
    low, high = (float(cast(float, edge)) for edge in band)
    if low <= 0.0 or low > high:
        raise ValueError(f"{where}: `published_band` must be an ascending pair of positive edges")
    decimals = published.get(_BAND_DECIMALS)
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 1:
        raise ValueError(f"{where}: a rounded band reading needs positive integer `band_decimals`")
    return low, high, decimals


def _band_resolution(statement: tuple[object, ...], values: Sequence[float]) -> ReadingResult:
    low, high, decimals = cast(tuple[float, float, int], statement)
    rounded = [round(value, decimals) for value in values]
    if not rounded:
        return ReadingResult(False, "no re-derived value produces the published band")
    resolved = min(rounded), max(rounded)
    holds = resolved == (low, high)
    phrase = (
        f"the re-derived values derive {resolved[0]:.{decimals}f}-{resolved[1]:.{decimals}f}x, "
        f"{'the published' if holds else 'not the published'} {low:.{decimals}f}-"
        f"{high:.{decimals}f}x band"
    )
    return ReadingResult(holds, phrase)


def _band_comparison(statement: tuple[object, ...], value: float) -> ReadingResult:
    low, high, decimals = cast(tuple[float, float, int], statement)
    quoted = round(value, decimals)
    holds = low <= quoted <= high
    position = "inside the published" if holds else "outside the"
    return ReadingResult(
        holds,
        f"portable ratio {value:.3f}x {position} {low:.{decimals}f}-{high:.{decimals}f}x band",
    )


def _point_statement(published: Mapping[str, object], where: str) -> tuple[object, ...]:
    point = _number(published.get(_PUBLISHED_VALUE), _PUBLISHED_VALUE, where)
    tolerance = _number(published.get(_ABSOLUTE_TOLERANCE), _ABSOLUTE_TOLERANCE, where)
    if tolerance < 0.0:
        raise ValueError(f"{where}: `{_ABSOLUTE_TOLERANCE}` must be non-negative")
    return point, tolerance


def _point_comparison(statement: tuple[object, ...], value: float) -> ReadingResult:
    point, tolerance = cast(tuple[float, float], statement)
    holds = abs(value - point) <= tolerance
    position = "within" if holds else "outside"
    return ReadingResult(
        holds,
        f"re-derived value {value:g} is {position} the published {point:g} +/- {tolerance:g}",
    )


def _each_point(statement: tuple[object, ...], values: Sequence[float]) -> ReadingResult:
    results = [_point_comparison(statement, value) for value in values]
    if not results:
        return ReadingResult(False, "no re-derived value supports the published point")
    failed = next((result for result in results if not result.holds), None)
    return failed or ReadingResult(True, "every re-derived value supports the published point")


def _bound_statement(published: Mapping[str, object], where: str) -> tuple[object, ...]:
    return (_number(published.get(_PUBLISHED_UPPER_BOUND), _PUBLISHED_UPPER_BOUND, where),)


def _bound_comparison(statement: tuple[object, ...], value: float) -> ReadingResult:
    (bound,) = cast(tuple[float], statement)
    holds = value <= bound
    position = "at or below" if holds else "above"
    return ReadingResult(
        holds, f"re-derived value {value:g} is {position} the published upper bound {bound:g}"
    )


def _each_bound(statement: tuple[object, ...], values: Sequence[float]) -> ReadingResult:
    results = [_bound_comparison(statement, value) for value in values]
    if not results:
        return ReadingResult(False, "no re-derived value supports the published upper bound")
    failed = next((result for result in results if not result.holds), None)
    return failed or ReadingResult(
        True, "every re-derived value supports the published upper bound"
    )


PUBLISHED_VALUE_READINGS: dict[str, RegisteredReading] = {
    READING_ROUNDED_BAND: RegisteredReading(
        name=READING_ROUNDED_BAND,
        criterion=CRITERION_ROUNDED_BAND,
        statement_kind="band",
        statement=_band_statement,
        resolve=_band_resolution,
        compare=_band_comparison,
    ),
    READING_POINT_TOLERANCE: RegisteredReading(
        name=READING_POINT_TOLERANCE,
        criterion=CRITERION_POINT_TOLERANCE,
        statement_kind="point statement",
        statement=_point_statement,
        resolve=_each_point,
        compare=_point_comparison,
    ),
    READING_UPPER_BOUND: RegisteredReading(
        name=READING_UPPER_BOUND,
        criterion=CRITERION_UPPER_BOUND,
        statement_kind="upper bound",
        statement=_bound_statement,
        resolve=_each_bound,
        compare=_bound_comparison,
    ),
}


def registered_reading(name: object, *, where: str) -> RegisteredReading:
    """Resolve a declared reading, refusing a rule neither reader can apply."""
    reading = PUBLISHED_VALUE_READINGS.get(name) if isinstance(name, str) else None
    if reading is None:
        raise ValueError(
            f"{where}: it declares the `{READING}` {name!r}, which no registered published-value "
            f"reading carries; choose from {sorted(PUBLISHED_VALUE_READINGS)}"
        )
    return reading


def required_reading(published: Mapping[str, object], *, where: str) -> PublishedValueReading:
    """Bind the reading a derived value must state beside its operation."""
    named = published.get(READING)
    if named is None:
        raise ValueError(
            f"{where}: this value is derived, so it must name the `{READING}` that judges its "
            "published statement and any re-derived value"
        )
    return registered_reading(named, where=where).bind(published, where=where)
