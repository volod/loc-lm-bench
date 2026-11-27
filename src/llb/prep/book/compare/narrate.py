"""Turn a selection into the comparison report.

The pre-draft agreement is attached beside each lane. It is not an input of the
decision. A lane that was not measured is listed and wins nothing.
"""

from collections.abc import Callable, Mapping, Sequence

from llb.prep.book.compare.constants import (
    REOPEN_WHEN,
    ROLE_PAGE,
    ROLE_VARIANT,
    STATUS_OVER_VRAM,
    STATUS_UNAVAILABLE,
)
from llb.prep.book.compare.decide import ClassChoice, Margin, Selection
from llb.prep.book.compare.examples import worked_examples
from llb.prep.book.compare.inputs import reference_digest
from llb.prep.book.compare.lanes import LaneInput
from llb.prep.book.compare.report import (
    ClassDecision,
    ClassLaneReport,
    ComparisonReport,
    MarginReport,
    NotCompared,
    SubstitutionExample,
)
from llb.prep.book.metrics.page_score import alignment_pair
from llb.prep.book.metrics.reference import RefPage
from llb.prep.book.metrics.report import LaneScore, ScoreReport
from llb.prep.book.metrics.submission import PageSubmission


def build_report(
    pages: Sequence[RefPage],
    draft_score: ScoreReport,
    by_lane: Mapping[str, LaneScore],
    selection: Selection,
    lanes: Sequence[LaneInput],
    language: str,
    baseline_lane: str,
    submissions: Sequence[PageSubmission],
    projected: float | None,
    unprojected: list[str],
    vram: Mapping[str, int],
    measured: Callable[[LaneInput], bool],
) -> ComparisonReport:
    """The report for one accepted comparison. ``measured`` says which lanes were scored."""
    agreements = _agreements(draft_score)
    classes = [
        _decision(choice, by_lane, agreements, pages, selection.reverted_dewarp)
        for choice in selection.classes
    ]
    skipped = [lane for lane in lanes if not measured(lane)]
    won = list(selection.won_no_class)
    won.extend(sorted(lane.lane for lane in skipped if lane.role == ROLE_PAGE))
    peak_mb, peak_engine = _peak(vram)
    return ComparisonReport(
        language=language,
        baseline_lane=baseline_lane,
        negative=selection.negative,
        reopen_when=REOPEN_WHEN if selection.negative else None,
        classes=classes,
        won_no_class=won,
        variants_not_adopted=_not_adopted(selection, skipped),
        not_compared=[_not_compared(lane) for lane in skipped],
        examples=[
            SubstitutionExample.model_validate(row)
            for row in worked_examples(_pairs(pages, submissions))
        ],
        projected_seconds=projected,
        unprojected_classes=unprojected,
        peak_vram_mb=peak_mb,
        peak_vram_engine=peak_engine,
        reference_digest=reference_digest(pages),
    )


def _not_compared(lane: LaneInput) -> NotCompared:
    if lane.status == STATUS_OVER_VRAM:
        return NotCompared(
            lane=lane.lane, engine=lane.engine, reason="over-vram", detail=lane.detail
        )
    if lane.status == STATUS_UNAVAILABLE:
        return NotCompared(
            lane=lane.lane, engine=lane.engine, reason="unavailable", detail=lane.detail
        )
    raise ValueError(f"lane {lane.lane} was scored and cannot be listed as skipped")


def _decision(
    choice: ClassChoice,
    by_lane: Mapping[str, LaneScore],
    agreements: Mapping[tuple[str, int], float],
    pages: Sequence[RefPage],
    reverted: Sequence[str],
) -> ClassDecision:
    group = [page for page in pages if page.page_class == choice.page_class]
    rows = []
    for result in choice.lanes:
        margin = _margin(result.margin)
        danger = _margin(result.dangerous_margin)
        rows.append(
            ClassLaneReport(
                lane=result.lane,
                engine=result.engine,
                metrics=by_lane[result.lane].by_page_class[choice.page_class],
                margin=margin,
                dangerous_rate=result.dangerous_rate,
                baseline_dangerous_rate=result.baseline_dangerous_rate,
                dangerous_margin=danger,
                blocked_dangerous=result.blocked_dangerous,
                predraft_character_agreement=_mean(agreements, result.lane, group),
            )
        )
    return ClassDecision(
        page_class=choice.page_class,
        pages=choice.pages,
        winner_lane=choice.winner_lane,
        configured_engine=choice.configured_engine,
        variant=choice.variant.variant,
        dewarp=choice.variant.dewarp,
        rationale=_rationale(choice, choice.page_class in reverted),
        lanes=rows,
    )


def _margin(margin: Margin | None) -> MarginReport | None:
    if margin is None:
        return None
    return MarginReport(point=margin.point, low=margin.low, high=margin.high, status=margin.status)


def _rationale(choice: ClassChoice, reverted: bool) -> str:
    parts = [
        f"{choice.page_class}: {choice.pages} page(s), winner {choice.winner_lane}, "
        f"engine {choice.configured_engine}."
    ]
    for result in choice.lanes:
        if result.margin is None:
            continue
        blocked = ""
        if result.blocked_dangerous:
            blocked = " blocked because the dangerous-substitution rate rose"
        margin = result.margin
        parts.append(
            f"{result.lane} character-error margin {margin.point:.3f} "
            f"[{margin.low:.3f}, {margin.high:.3f}] {margin.status}{blocked}."
        )
    parts.append(f"variant {choice.variant.variant}.")
    if reverted:
        parts.append("dewarp method conflict; this class kept the original variant.")
    return " ".join(parts)


def _pairs(
    pages: Sequence[RefPage], submissions: Sequence[PageSubmission]
) -> list[tuple[str, int, str, str, str]]:
    rows = []
    by_page = {page.pdf_page: page for page in pages}
    for sub in submissions:
        page = by_page.get(sub.pdf_page)
        if page is None:
            continue
        reference, hypothesis = alignment_pair(page, sub)
        rows.append((sub.lane, page.pdf_page, page.page_class, reference, hypothesis))
    return sorted(rows, key=lambda row: (row[0], row[1]))


def _agreements(score: ScoreReport) -> dict[tuple[str, int], float]:
    found = {}
    for lane in score.lanes:
        for page in lane.pages:
            cer = page.character_error_rate
            if cer.status == "scored" and cer.value is not None:
                found[(lane.lane, page.pdf_page)] = 1.0 - float(cer.value)
    return found


def _mean(
    found: Mapping[tuple[str, int], float], lane: str, pages: Sequence[RefPage]
) -> float | None:
    values = [found[(lane, page.pdf_page)] for page in pages if (lane, page.pdf_page) in found]
    if not values:
        return None
    return float(sum(values) / len(values))


def _not_adopted(selection: Selection, skipped: Sequence[LaneInput]) -> list[str]:
    names = list(selection.variants_not_adopted)
    names.extend(
        sorted(
            lane.lane
            for lane in skipped
            if lane.role == ROLE_VARIANT and lane.variant != "original"
        )
    )
    return names


def _peak(vram: Mapping[str, int]) -> tuple[int | None, str | None]:
    if not vram:
        return None, None
    engine = min(vram, key=lambda name: (-vram[name], name))
    return vram[engine], engine
