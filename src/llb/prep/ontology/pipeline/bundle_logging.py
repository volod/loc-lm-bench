"""Focused bundle logging implementation."""

import logging
from typing import TYPE_CHECKING
from pathlib import Path
from llb.prep.ontology.artifacts.report import required_gate_names
from llb.prep.ontology.constants import (
    PDF_ONTOLOGY_REPORT_FILENAME,
)

if TYPE_CHECKING:
    pass
_LOG = logging.getLogger(__name__)


def _log_calibration_gates(report: dict[str, object] | None, out_dir: Path) -> None:
    """Surface the calibration roll-up so `prepare-goldset-draft` (and the quickstart wrapper) act
    on the gate, not just record it. A failing gate is a WARNING, never fatal: the bundle is always
    written for inspection, and the human verification gate remains the real block on scoring."""
    gates = report.get("gates") if isinstance(report, dict) else None
    if not isinstance(gates, dict):
        return
    if gates.get("passed"):
        _LOG.info(
            "[ontology] calibration gates passed -> %s", out_dir / PDF_ONTOLOGY_REPORT_FILENAME
        )
        return
    # name only the REQUIRED gates that blocked the roll-up (informational gates like
    # nonzero_grounded_facts, and the needle gate on a non-PDF corpus, never appear here)
    required = required_gate_names(bool(gates.get("pdf_citation_gate_applicable")))
    failed = [name for name in required if not gates.get(name)]
    _LOG.warning(
        "[ontology] calibration gates NOT passed (%s); inspect %s before accepting this bundle",
        ", ".join(failed) or "see report",
        out_dir / PDF_ONTOLOGY_REPORT_FILENAME,
    )
