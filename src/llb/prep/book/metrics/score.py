"""Score a reference transcription against a book-ocr run, the existing PDF lane, or both.

A run that has resolved pages adds the ``book-ocr/resolved`` lane beside the raw engine
lanes. Only reference pages are scored. A reference page with no submission stays in the
report and in the class totals.
"""

from pathlib import Path

from llb.prep.book.metrics.page_score import score_page
from llb.prep.book.metrics.reference import RefPage, load_reference, require_formulas
from llb.prep.book.metrics.report import (
    ClassScore,
    LaneScore,
    PageScore,
    ScoreReport,
    field_coverage,
    rollup,
)
from llb.prep.book.metrics.resolved import load_resolved
from llb.prep.book.metrics.submission import PageSubmission, load_citations, load_run


def score_reference(
    reference_dir: Path,
    run_dir: Path | None = None,
    citations: Path | None = None,
    engine: str | None = None,
) -> ScoreReport:
    """The report for ``llb book-ocr score``. Refuses a reference whose formulas do not parse."""
    pages = load_reference(reference_dir)
    require_formulas(pages)
    submissions: list[PageSubmission] = []
    if run_dir is not None:
        submissions.extend(load_run(run_dir, engine))
        submissions.extend(load_resolved(run_dir))
    if citations is not None:
        submissions.extend(load_citations(citations))
    if not submissions:
        raise ValueError("score needs a book-ocr run directory or a citations sidecar")
    return score_submissions(pages, submissions)


def score_submissions(pages: list[RefPage], submissions: list[PageSubmission]) -> ScoreReport:
    """Score every reference page on every lane the submissions name."""
    by_lane: dict[str, dict[int, PageSubmission]] = {}
    text_only: dict[str, bool] = {}
    for submission in submissions:
        by_lane.setdefault(submission.lane, {})[submission.pdf_page] = submission
        text_only[submission.lane] = submission.text_only
    lanes = [_lane(name, pages, found, text_only[name]) for name, found in sorted(by_lane.items())]
    return ScoreReport(lanes=lanes)


def summary_lines(report: ScoreReport) -> list[str]:
    """A short text report. Substitution counts are listed on their own; there is no total."""
    lines: list[str] = []
    for lane in report.lanes:
        lines.append(
            f"[book-ocr] score {lane.lane}: {len(lane.pages)} page(s) "
            f"missing {lane.missing_submissions}"
        )
        lines.extend(_page_line(page) for page in lane.pages)
        lines.extend(
            _class_line(name, rollup_score) for name, rollup_score in lane.by_page_class.items()
        )
    return lines


def _lane(
    name: str, pages: list[RefPage], found: dict[int, PageSubmission], text_only: bool
) -> LaneScore:
    scored = []
    for page in pages:
        submission = found.get(page.pdf_page)
        if submission is None:
            submission = PageSubmission(name, page.pdf_page, (), text_only, present=False)
        scored.append(score_page(page, submission))
    missing, unscorable, abstentions = field_coverage(scored)
    return LaneScore(
        lane=name,
        pages=scored,
        by_page_class=rollup(scored),
        missing_submissions=missing,
        unscorable_fields=unscorable,
        abstentions=abstentions,
    )


def _page_line(page: PageScore) -> str:
    detail = _hot(page.dangerous_substitutions) or "substitutions 0"
    abstain = _hot(page.abstentions)
    suffix = f" abstain {abstain}" if abstain else ""
    return (
        f"[book-ocr]   pdf {page.pdf_page} class {page.page_class} "
        f"cer {_fmt(page.character_error_rate.value)} {detail}{suffix}"
    )


def _class_line(name: str, rollup_score: ClassScore) -> str:
    return (
        f"[book-ocr]   class {name}: {rollup_score.pages} page(s) "
        f"cer {_fmt(rollup_score.character_error_rate.value)}{_class_extra(rollup_score)}"
    )


def _class_extra(rollup_score: ClassScore) -> str:
    parts: list[str] = []
    if rollup_score.missing_submissions:
        parts.append(f"missing {rollup_score.missing_submissions}")
    abstain = _hot(rollup_score.abstentions)
    if abstain:
        parts.append(f"abstain {abstain}")
    unscored = _hot(rollup_score.unscorable_fields)
    if unscored:
        parts.append(f"unscorable {unscored}")
    if not parts:
        return ""
    return f" {' '.join(parts)}"


def _hot(counts: dict[str, int]) -> str:
    return " ".join(f"{name} {count}" for name, count in counts.items() if count)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
