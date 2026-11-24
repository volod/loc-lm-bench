"""Validate one reading vault and write its processing report."""

import logging
from pathlib import Path

from llb.prep.book.validate.ledger import scan_run
from llb.prep.book.validate.links import anchor_index, link_findings, repeated_anchor_findings
from llb.prep.book.validate.load import load_context
from llb.prep.book.validate.markers import marker_check
from llb.prep.book.validate.nav import nav_findings
from llb.prep.book.validate.pages import page_findings
from llb.prep.book.validate.references import reference_warnings
from llb.prep.book.validate.report import write_report
from llb.prep.book.validate.types import BookContext, Finding, PageRow, RunFacts, ValidationReport

_LOG = logging.getLogger(__name__)


def validate_book(book_dir: Path, run_dir: Path | None = None) -> ValidationReport:
    """Check the vault and write `metadata/processing_report.md` and `validation.json`.

    A failed check is recorded and the function still returns. The caller chooses
    the exit code. Recovered text is not logged.
    """
    context = load_context(book_dir, run_dir)
    facts = scan_run(context.run_dir, context.config)
    pages, findings = _collect(context, facts)
    open_marks, marker_findings, warnings = marker_check(context, facts)
    findings.extend(marker_findings)
    warnings.extend(reference_warnings(context.book_dir))
    warnings.extend(_structure_warnings(context))
    warnings.extend(_manifest_warnings(context, pages))
    unique = _unique(findings)
    report = ValidationReport(
        ok=not unique,
        findings=tuple(unique),
        pages=tuple(pages),
        warnings=tuple(warnings),
        facts=facts,
        context=context,
    )
    write_report(report, open_marks)
    _LOG.info("[book-validate] %s finding(s), %s warning(s)", len(unique), len(warnings))
    return report


def _collect(context: BookContext, facts: RunFacts) -> tuple[list[PageRow], list[Finding]]:
    noted = [doc for doc in context.docs if not doc.rel.startswith("metadata/")]
    findings = list(facts.findings)
    findings.extend(repeated_anchor_findings(noted))
    findings.extend(link_findings(context, anchor_index(noted)))
    findings.extend(nav_findings(context))
    pages, page_problems = page_findings(context)
    findings.extend(page_problems)
    return pages, findings


def _structure_warnings(context: BookContext) -> list[str]:
    warnings: list[str] = []
    conflicts = len(context.structure.conflicts)
    if conflicts:
        warnings.append(f"{conflicts} structure conflict(s) remain")
    fallback = sum(1 for page in context.structure.pages if page.fallback)
    if fallback:
        warnings.append(f"{fallback} printed page(s) used a fallback mapping")
    return warnings


def _manifest_warnings(context: BookContext, pages: list[PageRow]) -> list[str]:
    manifest = context.manifest
    if manifest is None:
        return []
    failed = {
        page.pdf_page
        for page in manifest.pages
        if any(stage.status == "failed" for stage in page.stages)
    }
    completed = {row.pdf_page for row in pages if row.status == "completed"}
    both = sorted(failed & completed)
    if not both:
        return []
    shown = ", ".join(str(number) for number in both[:8])
    extra = f" and {len(both) - 8} more" if len(both) > 8 else ""
    return [f"{len(both)} completed page(s) still have a failed stage: {shown}{extra}"]


def _unique(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, str, str]] = set()
    ordered: list[Finding] = []
    for item in sorted(findings, key=lambda finding: (finding.file, finding.check, finding.detail)):
        key = (item.check, item.file, item.detail)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered
