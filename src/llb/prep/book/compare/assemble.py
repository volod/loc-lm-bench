"""Score saved ledgers and choose the book-ocr configuration.

Nothing here loads a model. The reference must be accepted, the pre-draft must
cover the same pages, and the baseline sidecar's language must be the one the
report records.
"""

from collections.abc import Callable, Mapping, Sequence

from llb.core.contracts.data_prep.book_classes import PAGE_CLASS_IDS
from llb.prep.book.compare.config_out import (
    PipelineChoice,
    StageRouting,
    pipeline_document,
    stage_routings,
)
from llb.prep.book.compare.constants import (
    BASELINE_ENGINE,
    ROLE_PAGE,
    ROLE_VARIANT,
    STATUS_MEASURED,
    STATUS_OVER_VRAM,
    STATUS_UNAVAILABLE,
)
from llb.prep.book.compare.decide import (
    CompetingLane,
    PagePoint,
    Selection,
    select_pipelines,
)
from llb.prep.book.compare.inputs import AcceptRecord, require_accepts
from llb.prep.book.compare.lanes import LaneInput
from llb.prep.book.compare.narrate import build_report
from llb.prep.book.compare.project import projected_seconds
from llb.prep.book.compare.report import ComparisonReport
from llb.prep.book.metrics.constants import LANE_EXISTING
from llb.prep.book.metrics.latex import latex_parses
from llb.prep.book.metrics.reference import RefPage
from llb.prep.book.metrics.report import LaneScore
from llb.prep.book.metrics.score import score_submissions
from llb.prep.book.metrics.submission import PageSubmission

_STATUSES = {STATUS_MEASURED, STATUS_OVER_VRAM, STATUS_UNAVAILABLE}
_ROLES = {ROLE_PAGE, ROLE_VARIANT}


def compare_submissions(
    reference: Sequence[RefPage],
    predraft: Sequence[RefPage],
    submissions: Sequence[PageSubmission],
    lanes: Sequence[LaneInput],
    *,
    language: str,
    accepts: AcceptRecord,
    baseline_vram_mb: int,
    class_page_counts: Mapping[str, int],
    serves: Callable[[str, str], bool],
    baseline_seconds_per_page: float | None = None,
    baseline_lane: str = LANE_EXISTING,
    baseline_engine: str = BASELINE_ENGINE,
    recorded_language: str | None = None,
    source_sha256: str | None = None,
) -> tuple[ComparisonReport, dict[str, object]]:
    """The report and the book-ocr config. Refuses an unaccepted or mismatched input."""
    pages = _checked_pages(reference, predraft, accepts, language, recorded_language, source_sha256)
    draft = sorted(predraft, key=lambda page: page.pdf_page)
    _checked_lanes(lanes)
    covered = _cover(list(submissions), lanes, pages, baseline_lane)
    reference_score = score_submissions(pages, covered)
    draft_score = score_submissions(draft, covered)
    by_lane = {lane.lane: lane for lane in reference_score.lanes}
    class_names = _class_names(pages)
    competing = [_competing(lane, by_lane, class_names) for lane in lanes if _measured(lane)]
    baseline = _competing(
        LaneInput(baseline_lane, baseline_engine, ROLE_PAGE, baseline_vram_mb),
        by_lane,
        class_names,
    )
    if baseline_lane not in by_lane:
        raise ValueError(f"baseline lane {baseline_lane} has no submission")
    selection = select_pipelines(class_names, baseline_lane, baseline, competing)
    class_engines = {choice.page_class: choice.configured_engine for choice in selection.classes}
    routings = stage_routings(class_engines, baseline_engine, serves)
    vram = _vram(routings, lanes, baseline_engine, baseline_vram_mb)
    choice = _pipeline(language, selection, routings, vram)
    config = pipeline_document(choice)
    projected, unprojected = _project(
        class_names,
        class_page_counts,
        class_engines,
        routings,
        lanes,
        baseline_engine,
        baseline_seconds_per_page,
    )
    report = build_report(
        pages,
        draft_score,
        by_lane,
        selection,
        lanes,
        language,
        baseline_lane,
        covered,
        projected,
        unprojected,
        vram,
        _measured,
    )
    return report, config


def _checked_pages(
    reference: Sequence[RefPage],
    predraft: Sequence[RefPage],
    accepts: AcceptRecord,
    language: str,
    recorded: str | None,
    source_sha256: str | None,
) -> list[RefPage]:
    if not language:
        raise ValueError("baseline OCR language is empty")
    if recorded is not None and recorded != language:
        raise ValueError(f"baseline OCR language is {recorded}, not {language}")
    pages = sorted(reference, key=lambda page: page.pdf_page)
    _unique(pages)
    _unique(list(predraft))
    if {page.pdf_page for page in predraft} != {page.pdf_page for page in pages}:
        raise ValueError("pre-draft pages do not match the reference")
    _require_formulas(pages)
    _require_formulas(list(predraft))
    require_accepts(pages, accepts, source_sha256)
    return pages


def _checked_lanes(lanes: Sequence[LaneInput]) -> None:
    seen: set[str] = set()
    for lane in lanes:
        if lane.lane in seen:
            raise ValueError(f"lane {lane.lane} is repeated")
        seen.add(lane.lane)
        if lane.role not in _ROLES or lane.status not in _STATUSES:
            raise ValueError(f"lane {lane.lane} has an unknown role or status")
        if lane.variant == "dewarp" and not lane.dewarp:
            raise ValueError(f"lane {lane.lane} dewarp variant needs a method")


def _cover(
    submissions: list[PageSubmission],
    lanes: Sequence[LaneInput],
    pages: Sequence[RefPage],
    baseline_lane: str,
) -> list[PageSubmission]:
    wanted = {lane.lane for lane in lanes if _measured(lane)} | {baseline_lane}
    kept = [sub for sub in submissions if sub.lane in wanted]
    have = {(sub.lane, sub.pdf_page) for sub in kept}
    text_only = {sub.lane for sub in kept if sub.text_only}
    text_only.add(baseline_lane)
    for lane in wanted:
        for page in pages:
            if (lane, page.pdf_page) in have:
                continue
            kept.append(PageSubmission(lane, page.pdf_page, (), lane in text_only, present=False))
    return kept


def _competing(lane: LaneInput, by_lane: dict[str, LaneScore], names: list[str]) -> CompetingLane:
    score = by_lane[lane.lane]
    return CompetingLane(
        lane.lane,
        lane.engine,
        lane.role,
        lane.variant,
        lane.dewarp,
        lane.vram_mb,
        {name: _points(score, name) for name in names},
    )


def _points(score: LaneScore, name: str) -> list[PagePoint]:
    points = []
    for page in score.pages:
        if page.page_class != name:
            continue
        cer = page.character_error_rate
        value = None if cer.status != "scored" or cer.value is None else float(cer.value)
        points.append(
            PagePoint(
                page.pdf_page,
                value,
                sum(page.dangerous_substitutions.values()),
                page.reference_chars,
            )
        )
    return sorted(points, key=lambda point: point.pdf_page)


def _pipeline(
    language: str,
    selection: Selection,
    routings: dict[str, StageRouting],
    vram: dict[str, int],
) -> PipelineChoice:
    by_class = {choice.page_class: choice.variant.variant for choice in selection.classes}
    unique = set(by_class.values())
    if len(unique) == 1:
        default = unique.pop()
        overrides: dict[str, str] = {}
    else:
        default = "original"
        overrides = {name: variant for name, variant in by_class.items() if variant != "original"}
    return PipelineChoice(
        language,
        default,
        overrides,
        selection.dewarp_method,
        default != "original" or bool(overrides),
        routings,
        vram,
    )


def _vram(
    routings: Mapping[str, StageRouting],
    lanes: Sequence[LaneInput],
    baseline_engine: str,
    baseline_vram: int,
) -> dict[str, int]:
    needed: set[str] = set()
    for routing in routings.values():
        needed.add(routing.engine)
        needed.update(routing.by_page_class.values())
    known = {lane.engine: lane.vram_mb for lane in lanes}
    known[baseline_engine] = baseline_vram
    missing = sorted(engine for engine in needed if engine not in known)
    if missing:
        raise ValueError(f"no VRAM budget for {', '.join(missing)}")
    return {engine: known[engine] for engine in sorted(needed)}


def _project(
    names: Sequence[str],
    counts: Mapping[str, int],
    winners: Mapping[str, str],
    routings: Mapping[str, StageRouting],
    lanes: Sequence[LaneInput],
    baseline_engine: str,
    baseline_seconds: float | None,
) -> tuple[float | None, list[str]]:
    present = {name: counts[name] for name in names if name in counts}
    seconds = {
        lane.engine: lane.seconds_per_page
        for lane in lanes
        if _measured(lane) and lane.role == ROLE_PAGE and lane.seconds_per_page is not None
    }
    if baseline_seconds is not None:
        seconds[baseline_engine] = baseline_seconds
    pairs = {name: (routing.engine, routing.by_page_class) for name, routing in routings.items()}
    projected, missing_rates = projected_seconds(present, winners, pairs, seconds)
    unprojected = [name for name in names if name not in counts] + missing_rates
    if unprojected:
        return None, unprojected
    return projected, []


def _measured(lane: LaneInput) -> bool:
    return lane.status == STATUS_MEASURED


def _class_names(pages: list[RefPage]) -> list[str]:
    order = {name: index for index, name in enumerate(PAGE_CLASS_IDS)}
    names = list(dict.fromkeys(page.page_class for page in pages))
    return sorted(names, key=lambda name: (order.get(name, len(order)), name))


def _unique(pages: list[RefPage]) -> None:
    seen: set[int] = set()
    for page in pages:
        if page.pdf_page in seen:
            raise ValueError(f"pdf page {page.pdf_page} is repeated")
        seen.add(page.pdf_page)


def _require_formulas(pages: list[RefPage]) -> None:
    errors = [
        f"pdf page {page.pdf_page} block {block.block_id} latex does not parse"
        for page in pages
        for block in page.blocks
        if block.kind == "formula" and not latex_parses(block.latex or "")
    ]
    if errors:
        raise ValueError("; ".join(errors))
