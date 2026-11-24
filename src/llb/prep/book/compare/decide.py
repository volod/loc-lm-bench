"""Choose one engine and one variant per page class.

A page reader wins a class only when its character-error margin over the baseline
is a gain on every pilot page of that class and its pooled dangerous-substitution
rate is not higher. Otherwise the existing lane stays. A variant policy is adopted
on the same rule against the original-variant lane of the same engine. Two classes
that want different dewarp methods cannot both have one: the stronger margin keeps
its method and the other class stays on the original.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from llb.prep.book.compare.bootstrap import MarginStatus, interval_status, page_bootstrap_interval
from llb.prep.book.compare.constants import ROLE_PAGE, ROLE_VARIANT, STATUS_MEASURED


@dataclass(frozen=True)
class PagePoint:
    """One page of one lane. ``cer`` is None when the rate was not scored."""

    pdf_page: int
    cer: float | None
    dangerous_count: int
    reference_chars: int


@dataclass(frozen=True)
class Margin:
    """A mean margin and the page-bootstrap interval around it."""

    point: float
    low: float
    high: float
    status: MarginStatus


@dataclass(frozen=True)
class CompetingLane:
    """A measured lane, with one point list per reference class."""

    lane: str
    engine: str
    role: str
    variant: str
    dewarp: str | None
    vram_mb: int
    by_class: dict[str, list[PagePoint]]


@dataclass(frozen=True)
class LaneClassResult:
    """How one lane compared with the baseline on one class."""

    lane: str
    engine: str
    margin: Margin | None
    dangerous_rate: float | None
    baseline_dangerous_rate: float | None
    dangerous_margin: Margin | None
    blocked_dangerous: bool


@dataclass(frozen=True)
class VariantPick:
    """The variant a class adopted. ``lane`` is None when the original stayed."""

    variant: str
    dewarp: str | None
    lane: str | None
    low: float


@dataclass(frozen=True)
class ClassChoice:
    """The winner of one class and every lane that was allowed to compete."""

    page_class: str
    pages: int
    winner_lane: str
    configured_engine: str
    variant: VariantPick
    lanes: list[LaneClassResult]


@dataclass(frozen=True)
class Selection:
    """Per-class winners, plus the candidates that won nothing."""

    classes: list[ClassChoice]
    won_no_class: list[str]
    variants_not_adopted: list[str]
    dewarp_method: str
    reverted_dewarp: list[str]
    negative: bool


def select_pipelines(
    class_names: Sequence[str],
    baseline_lane: str,
    baseline: CompetingLane,
    lanes: Sequence[CompetingLane],
) -> Selection:
    """The per-class selection. ``baseline`` is the existing lane."""
    readers = [lane for lane in lanes if lane.role == ROLE_PAGE and lane.lane != baseline_lane]
    choices = [_class_choice(name, baseline_lane, baseline, lanes) for name in class_names]
    picks, method, reverted = _variants(class_names, lanes)
    for index, choice in enumerate(choices):
        choices[index] = ClassChoice(
            choice.page_class,
            choice.pages,
            choice.winner_lane,
            choice.configured_engine,
            picks[choice.page_class],
            choice.lanes,
        )
    winners = {choice.winner_lane for choice in choices}
    adopted = {pick.lane for pick in picks.values() if pick.lane}
    won_none = sorted(lane.lane for lane in readers if lane.lane not in winners)
    not_adopted = sorted(
        lane.lane
        for lane in lanes
        if lane.role == ROLE_VARIANT and lane.variant != "original" and lane.lane not in adopted
    )
    negative = all(choice.winner_lane == baseline_lane for choice in choices)
    return Selection(choices, won_none, not_adopted, method, reverted, negative)


def _class_choice(
    name: str,
    baseline_lane: str,
    baseline: CompetingLane,
    lanes: Sequence[CompetingLane],
) -> ClassChoice:
    readers = [lane for lane in lanes if lane.role == ROLE_PAGE and lane.lane != baseline_lane]
    others = [lane for lane in lanes if lane.role != ROLE_PAGE]
    base_points = baseline.by_class.get(name, [])
    results = [_against_baseline(baseline, lane, name) for lane in readers]
    rest = [_against_baseline(baseline, lane, name) for lane in others]
    winner = baseline_lane
    engine = baseline.engine
    best: tuple[float, int, str] | None = None
    for lane, result in zip(readers, results, strict=True):
        if result.margin is None or result.margin.status != "gain" or result.blocked_dangerous:
            continue
        key = (-result.margin.low, lane.vram_mb, lane.lane)
        if best is None or key < best:
            best = key
            winner = lane.lane
            engine = lane.engine
    base_rate = _pooled(base_points)
    baseline_row = LaneClassResult(
        baseline_lane, baseline.engine, None, base_rate, base_rate, None, False
    )
    return ClassChoice(
        name,
        len(base_points),
        winner,
        engine,
        VariantPick("original", None, None, 0.0),
        [baseline_row, *results, *rest],
    )


def _against_baseline(baseline: CompetingLane, lane: CompetingLane, name: str) -> LaneClassResult:
    base_points = baseline.by_class.get(name, [])
    cand_points = lane.by_class.get(name, [])
    base_rate = _pooled(base_points)
    cand_rate = _pooled(cand_points)
    return LaneClassResult(
        lane.lane,
        lane.engine,
        _margin(_cer_margins(base_points, cand_points)),
        cand_rate,
        base_rate,
        _margin(_rate_margins(base_points, cand_points)),
        _rose(base_rate, cand_rate),
    )


def _variants(
    class_names: Sequence[str], lanes: Sequence[CompetingLane]
) -> tuple[dict[str, VariantPick], str, list[str]]:
    grouped: dict[str, list[CompetingLane]] = {}
    for lane in lanes:
        if lane.role == ROLE_VARIANT:
            grouped.setdefault(lane.engine, []).append(lane)
    policies = grouped.get("rapidocr") or next(iter(grouped.values()), [])
    control = next((lane for lane in policies if lane.variant == "original"), None)
    picks: dict[str, VariantPick] = {}
    for name in class_names:
        picks[name] = _pick_variant(name, control, policies)
    return _one_dewarp(picks)


def _pick_variant(
    name: str, control: CompetingLane | None, policies: Sequence[CompetingLane]
) -> VariantPick:
    if control is None:
        return VariantPick("original", None, None, 0.0)
    best: VariantPick | None = None
    best_key: tuple[float, int, str] | None = None
    for lane in policies:
        if lane.variant == "original":
            continue
        margin = _margin(_cer_margins(control.by_class.get(name, []), lane.by_class.get(name, [])))
        if margin is None or margin.status != "gain":
            continue
        if _rose(_pooled(control.by_class.get(name, [])), _pooled(lane.by_class.get(name, []))):
            continue
        key = (-margin.low, lane.vram_mb, lane.lane)
        if best_key is None or key < best_key:
            best_key = key
            best = VariantPick(lane.variant, lane.dewarp, lane.lane, margin.low)
    return best or VariantPick("original", None, None, 0.0)


def _one_dewarp(
    picks: dict[str, VariantPick],
) -> tuple[dict[str, VariantPick], str, list[str]]:
    methods = {pick.dewarp for pick in picks.values() if pick.variant == "dewarp" and pick.dewarp}
    if len(methods) <= 1:
        return picks, next(iter(methods), "opencv"), []
    ranked = sorted(
        (pick for pick in picks.values() if pick.variant == "dewarp"),
        key=lambda pick: (-pick.low, pick.lane or ""),
    )
    keep = ranked[0].dewarp or "opencv"
    resolved = {}
    reverted: list[str] = []
    for name, pick in picks.items():
        if pick.variant == "dewarp" and pick.dewarp != keep:
            resolved[name] = VariantPick("original", None, None, 0.0)
            reverted.append(name)
        else:
            resolved[name] = pick
    return resolved, keep, reverted


def _margin(values: list[float] | None) -> Margin | None:
    if not values:
        return None
    low, high = page_bootstrap_interval(values)
    point = sum(values) / len(values)
    return Margin(point, low, high, interval_status(low, high))


def _cer_margins(base: Sequence[PagePoint], cand: Sequence[PagePoint]) -> list[float] | None:
    return _paired(base, cand, lambda left, right: _subtract(left.cer, right.cer))


def _rate_margins(base: Sequence[PagePoint], cand: Sequence[PagePoint]) -> list[float] | None:
    return _paired(base, cand, lambda left, right: _rate_delta(left, right))


def _paired(
    base: Sequence[PagePoint],
    cand: Sequence[PagePoint],
    delta: Callable[[PagePoint, PagePoint], float | None],
) -> list[float] | None:
    by_page = {point.pdf_page: point for point in base}
    if len(by_page) != len(cand) or len(by_page) != len(base):
        return None
    values: list[float] = []
    for point in cand:
        other = by_page.get(point.pdf_page)
        if other is None:
            return None
        value = delta(other, point)
        if value is None:
            return None
        values.append(value)
    return values


def _subtract(base: float | None, cand: float | None) -> float | None:
    if base is None or cand is None:
        return None
    return base - cand


def _rate_delta(base: PagePoint, cand: PagePoint) -> float | None:
    if base.reference_chars == 0 or cand.reference_chars == 0:
        return None
    return _page_rate(base) - _page_rate(cand)


def _page_rate(point: PagePoint) -> float:
    if point.reference_chars == 0:
        return 0.0
    return point.dangerous_count / point.reference_chars


def _pooled(points: Sequence[PagePoint]) -> float | None:
    chars = sum(point.reference_chars for point in points)
    if chars == 0:
        return None
    return sum(point.dangerous_count for point in points) / chars


def _rose(baseline: float | None, candidate: float | None) -> bool:
    if baseline is None or candidate is None:
        return False
    return candidate > baseline


def measured(status: str) -> bool:
    """True when the lane was actually read and may compete."""
    return status == STATUS_MEASURED
