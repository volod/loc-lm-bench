"""Text summary of a comparison. Substitution examples stay on their own lines."""

from llb.prep.book.compare.report import (
    ClassDecision,
    ClassLaneReport,
    ComparisonReport,
    SubstitutionExample,
)


def summary_lines(report: ComparisonReport) -> list[str]:
    """One block per class, then the candidates that won nothing and the projection."""
    lines = [_header(report)]
    for row in report.classes:
        lines.extend(_class_lines(row))
    lines.extend(_tail(report))
    return lines


def _header(report: ComparisonReport) -> str:
    return (
        f"[book-ocr] compare language {report.language} baseline {report.baseline_lane} "
        "selection-set"
    )


def _class_lines(row: ClassDecision) -> list[str]:
    lines = [
        f"[book-ocr] class {row.page_class}: {row.pages} page(s) "
        f"winner {row.winner_lane} engine {row.configured_engine} variant {row.variant}"
    ]
    for lane in row.lanes:
        lines.extend(_lane_lines(lane))
    return lines


def _lane_lines(lane: ClassLaneReport) -> list[str]:
    if lane.margin is None:
        return []
    blocked = " blocked-dangerous" if lane.blocked_dangerous else ""
    margin = lane.margin
    lines = [
        f"[book-ocr]   {lane.lane} margin {margin.point:.3f} "
        f"[{margin.low:.3f}, {margin.high:.3f}] {margin.status}{blocked}"
    ]
    if lane.predraft_character_agreement is not None:
        lines.append(
            f"[book-ocr]   {lane.lane} predraft-agreement {lane.predraft_character_agreement:.3f}"
        )
    if lane.dangerous_margin is not None:
        danger = lane.dangerous_margin
        lines.append(
            f"[book-ocr]   {lane.lane} dangerous-margin {danger.point:.3f} "
            f"[{danger.low:.3f}, {danger.high:.3f}] {danger.status}"
        )
    return lines


def _tail(report: ComparisonReport) -> list[str]:
    lines: list[str] = []
    if report.won_no_class:
        lines.append("[book-ocr] won-no-class " + ", ".join(report.won_no_class))
    lines.extend(
        f"[book-ocr] not-compared {skipped.lane} {skipped.reason} {skipped.detail}"
        for skipped in report.not_compared
    )
    lines.extend(_example_line(example) for example in report.examples)
    lines.append(_projection(report))
    if report.peak_vram_engine is not None:
        lines.append(f"[book-ocr] peak vram {report.peak_vram_engine} {report.peak_vram_mb} MB")
    if report.negative:
        lines.append("[book-ocr] no candidate beat the existing lane on any class")
    return lines


def _example_line(example: SubstitutionExample) -> str:
    if not example.found:
        return f"[book-ocr] example {example.confusion_class} none"
    reference = _ascii(example.reference or "")
    hypothesis = _ascii(example.hypothesis or "")
    return (
        f"[book-ocr] example {example.confusion_class} pdf {example.pdf_page} "
        f"{reference} -> {hypothesis}"
    )


def _projection(report: ComparisonReport) -> str:
    if report.projected_seconds is None:
        return "[book-ocr] projected wall time unavailable selection-set"
    return f"[book-ocr] projected wall {report.projected_seconds:.1f} s selection-set"


def _ascii(text: str) -> str:
    return "".join(char if 32 <= ord(char) < 127 else f"\\u{ord(char):04x}" for char in text)
