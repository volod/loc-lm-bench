"""Noise-floor sections for fusion evidence reports."""

from llb.rag.fusion_evidence.models import FusionEvidenceReport
from llb.rag.noise_floor.report import render_noise_floor_markdown


def floor_section(report: FusionEvidenceReport) -> list[str]:
    """Render overall and focus-slice measurement floors when present."""
    floor = report.get("noise_floor")
    if floor is None:
        return []
    lines = render_noise_floor_markdown(floor, scored="every item")
    focus = report.get("noise_floor_focus")
    if focus is not None:
        lines += render_noise_floor_markdown(
            focus,
            title=f"Measurement floor: {report['focus_slice']}",
            scored=f"{report['focus_slice']} items",
        )
    return lines
