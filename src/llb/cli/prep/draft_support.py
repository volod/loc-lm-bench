"""Resume, validation, verification-sample, and gate helpers for ontology drafting."""

from pathlib import Path
from typing import Any, Optional

import typer

from llb.cli.helpers import cli_error


def _validate_optional_path(path: Optional[Path], *, kind: str, label: str) -> None:
    if path is None:
        return
    exists = path.is_dir() if kind == "directory" else path.is_file()
    if not exists:
        cli_error(f"{label} not found: {path}")


def _validate_extraction_bundle(bundle: Optional[Path]) -> None:
    if bundle is None:
        return
    if not (bundle / "extraction.jsonl").is_file():
        cli_error(f"reuse extraction bundle has no extraction.jsonl: {bundle}")


def _validate_dedup_bundles(bundles: Optional[list[Path | str]]) -> None:
    for bundle in bundles or []:
        if not (Path(bundle) / "goldset.jsonl").is_file():
            cli_error(f"dedup bundle has no goldset.jsonl: {bundle}")


def _validate_draft_inputs(
    drop_nonretrievable_needles: bool,
    retrieval_index_dir: Optional[Path],
    graph_dir: Optional[Path],
    rejection_feedback: Optional[Path],
    reuse_extraction_bundle: Optional[Path],
    multi_hop: bool,
    multi_hop_only: bool,
    carry_forward_multi_hop: bool,
    dedup_against: Optional[list[Path | str]],
    dedup_linkage_shadow: bool = False,
) -> None:
    """Fail fast (exit 2) on option combinations and paths that cannot work."""
    if drop_nonretrievable_needles and retrieval_index_dir is None:
        cli_error("--drop-nonretrievable-needles requires --retrieval-index-dir")
    _validate_optional_path(retrieval_index_dir, kind="directory", label="retrieval index dir")
    _validate_optional_path(graph_dir, kind="directory", label="graph store dir")
    _validate_optional_path(rejection_feedback, kind="file", label="rejection feedback file")
    _validate_extraction_bundle(reuse_extraction_bundle)
    if multi_hop_only and not multi_hop:
        cli_error("--multi-hop-only requires --multi-hop")
    if carry_forward_multi_hop and not dedup_against:
        cli_error("--carry-forward-multi-hop requires --dedup-against")
    if dedup_linkage_shadow and not dedup_against:
        cli_error("--dedup-linkage-shadow requires --dedup-against")
    _validate_dedup_bundles(dedup_against)


def _extraction_adapter(extractor: str, spacy_model: str) -> Any:
    """The opt-in spaCy extraction adapter, or None for the default LLM extractor."""
    if extractor != "spacy":
        return None
    from llb.prep.ontology.endpoints.spacy_adapter import SpacyExtractionAdapter

    return SpacyExtractionAdapter(model=spacy_model)


def _split_dir_list(value: Optional[str]) -> Optional[list[Path | str]]:
    """Comma-separated dirs -> Path list (None when the option is unset/empty)."""
    if not value:
        return None
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def _write_verification_sample(
    out_dir: Path,
    n: int | None,
    seed: int,
    *,
    confidence: float,
    precision: float,
) -> None:
    from llb.goldset.verify_sampling.worksheet import build_sample_worksheet

    worksheet = out_dir / "verify_sample.csv"
    sample_size, _strata = build_sample_worksheet(
        out_dir,
        worksheet,
        n=n,
        seed=seed,
        confidence=confidence,
        precision=precision,
    )
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
