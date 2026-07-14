"""Resume, validation, verification-sample, and gate helpers for ontology drafting."""

from pathlib import Path
from typing import Any, Optional

import typer

from llb.cli.helpers import cli_error


def _validate_draft_inputs(
    drop_nonretrievable_needles: bool,
    retrieval_index_dir: Optional[Path],
    graph_dir: Optional[Path],
    rejection_feedback: Optional[Path],
) -> None:
    """Fail fast (exit 2) on option combinations and paths that cannot work."""
    if drop_nonretrievable_needles and retrieval_index_dir is None:
        cli_error("--drop-nonretrievable-needles requires --retrieval-index-dir")
    if retrieval_index_dir is not None and not retrieval_index_dir.is_dir():
        cli_error(f"retrieval index dir not found: {retrieval_index_dir}")
    if graph_dir is not None and not graph_dir.is_dir():
        cli_error(f"graph store dir not found: {graph_dir}")
    if rejection_feedback is not None and not rejection_feedback.is_file():
        cli_error(f"rejection feedback file not found: {rejection_feedback}")


def _extraction_adapter(extractor: str, spacy_model: str) -> Any:
    """The opt-in spaCy extraction adapter, or None for the default LLM extractor."""
    if extractor != "spacy":
        return None
    from llb.prep.ontology.spacy_adapter import SpacyExtractionAdapter

    return SpacyExtractionAdapter(model=spacy_model)


def _split_dir_list(value: Optional[str]) -> Optional[list[Path | str]]:
    """Comma-separated dirs -> Path list (None when the option is unset/empty)."""
    if not value:
        return None
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def _write_verification_sample(out_dir: Path, n: int, seed: int) -> None:
    from llb.goldset.verify_sampling.worksheet import build_sample_worksheet

    worksheet = out_dir / "verify_sample.csv"
    sample_size, _strata = build_sample_worksheet(out_dir, worksheet, n=n, seed=seed)
    typer.echo(f"[prepare-goldset-draft] verification sample: {sample_size} rows -> {worksheet}")


def _enforce_calibration_gates(calibration_report: Any, out_dir: Path) -> None:
    """Exit 1 when the required ontology calibration gates failed (--require-passed-gates)."""
    from llb.prep.ontology.artifacts.report import required_gate_names
    from llb.prep.ontology.constants import PDF_ONTOLOGY_REPORT_FILENAME

    gates = calibration_report.get("gates") if isinstance(calibration_report, dict) else None
    if isinstance(gates, dict) and bool(gates.get("passed")):
        return
    failed: list[str] = []
    if isinstance(gates, dict):
        required = required_gate_names(bool(gates.get("pdf_citation_gate_applicable")))
        failed = [name for name in required if not gates.get(name)]
    detail = ", ".join(failed) if failed else "see report"
    cli_error(
        "ontology calibration gates not passed "
        f"({detail}); inspect {out_dir / PDF_ONTOLOGY_REPORT_FILENAME}",
        code=1,
    )
