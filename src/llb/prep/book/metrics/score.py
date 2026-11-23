"""Score a reference transcription against a book-ocr run, the existing PDF lane, or both."""

from pathlib import Path

from llb.prep.book.metrics.latex import latex_parses
from llb.prep.book.metrics.page_score import score_page
from llb.prep.book.metrics.reference import RefPage, load_reference
from llb.prep.book.metrics.report import LaneScore, ScoreReport, rollup
from llb.prep.book.metrics.submission import PageSubmission, load_citations, load_run


def score_reference(
    reference_dir: Path,
    run_dir: Path | None = None,
    citations: Path | None = None,
    engine: str | None = None,
) -> ScoreReport:
    """The report for ``llb book-ocr score``. Refuses a reference whose formulas do not parse."""
    pages = load_reference(reference_dir)
    _require_formulas(pages)
    submissions: list[PageSubmission] = []
    if run_dir is not None:
        submissions.extend(load_run(run_dir, engine))
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
        lines.append(f"[book-ocr] score {lane.lane}: {len(lane.pages)} page(s)")
        for page in lane.pages:
            hot = [
                f"{name} {count}" for name, count in page.dangerous_substitutions.items() if count
            ]
            detail = " ".join(hot) if hot else "substitutions 0"
            lines.append(
                f"[book-ocr]   pdf {page.pdf_page} class {page.page_class} "
                f"cer {_fmt(page.character_error_rate.value)} {detail}"
            )
        for name, rollup_score in lane.by_page_class.items():
            lines.append(
                f"[book-ocr]   class {name}: {rollup_score.pages} page(s) "
                f"cer {_fmt(rollup_score.character_error_rate.value)}"
            )
    return lines


def _lane(
    name: str, pages: list[RefPage], found: dict[int, PageSubmission], text_only: bool
) -> LaneScore:
    scored = []
    for page in pages:
        submission = found.get(page.pdf_page)
        if submission is None:
            submission = PageSubmission(name, page.pdf_page, (), text_only)
        scored.append(score_page(page, submission))
    return LaneScore(lane=name, pages=scored, by_page_class=rollup(scored))


def _require_formulas(pages: list[RefPage]) -> None:
    errors = [
        f"pdf page {page.pdf_page} block {block.block_id} latex does not parse"
        for page in pages
        for block in page.blocks
        if block.kind == "formula" and not latex_parses(block.latex or "")
    ]
    if errors:
        raise ValueError("; ".join(errors))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
