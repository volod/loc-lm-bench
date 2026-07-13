"""Resume, validation, verification-sample, and gate helpers for ontology drafting."""

from pathlib import Path
from typing import Any, Optional, cast

import typer

from llb.cli.helpers import cli_error


def _resume_overrides(
    resume: Path,
    corpus_root: Optional[Path],
    model: Optional[str],
    endpoint: str,
    backend: str,
    frontier_stage: str,
    local_model: Optional[str],
    max_usd: Optional[float],
    max_calls: Optional[int],
    out_dir: Optional[Path],
) -> tuple[Optional[Path], str, str, str, str, Optional[str], Optional[float], Optional[int], Path]:
    """Fill unset CLI values from the resumed bundle's journal meta.

    The bundle's journal meta is authoritative for the corpus and endpoint identity; the
    extraction/seed/retrieval settings are re-read inside draft_goldset(resume=True). The
    base URL is intentionally NOT restored so a vLLM resume relaunches a fresh server.
    """
    from llb.prep.ontology.pipeline.journaling import load_journal_meta

    try:
        meta = load_journal_meta(resume)
    except ValueError as exc:
        cli_error(str(exc))
    ep_meta = cast(dict[str, Any], meta.get("endpoint") or {})
    stages = cast(dict[str, Any], ep_meta.get("stages") or {})
    extraction = cast(dict[str, Any], stages.get("extraction") or {})
    drafting = cast(dict[str, Any], stages.get("drafting") or {})
    if corpus_root is None:
        corpus_root = Path(str(meta.get("corpus_root")))
    frontier_phases = [
        phase
        for phase, config in (("extraction", extraction), ("drafting", drafting))
        if config.get("kind") == "frontier"
    ]
    if frontier_phases:
        endpoint = "frontier"
        frontier_stage = "both" if len(frontier_phases) == 2 else frontier_phases[0]
        frontier_config = extraction if extraction.get("kind") == "frontier" else drafting
        local_config = drafting if frontier_stage == "extraction" else extraction
        model = model or str(frontier_config.get("model") or "")
        if max_usd is None and frontier_config.get("max_usd") is not None:
            max_usd = float(frontier_config["max_usd"])
        if max_calls is None and frontier_config.get("max_calls") is not None:
            max_calls = int(frontier_config["max_calls"])
        local_model = local_model or str(local_config.get("model") or "") or None
        backend = str(local_config.get("backend") or backend)
    else:
        endpoint = "local"
        model = model or str(extraction.get("model") or "")
        backend = str(extraction.get("backend") or backend)
    return (
        corpus_root,
        model,
        endpoint,
        backend,
        frontier_stage,
        local_model,
        max_usd,
        max_calls,
        out_dir or resume,
    )


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
    from llb.goldset.verify import build_sample_worksheet

    worksheet = out_dir / "verify_sample.csv"
    sample_size, _strata = build_sample_worksheet(out_dir, worksheet, n=n, seed=seed)
    typer.echo(f"[prepare-goldset-draft] verification sample: {sample_size} rows -> {worksheet}")


def _enforce_calibration_gates(calibration_report: Any, out_dir: Path) -> None:
    """Exit 1 when the required ontology calibration gates failed (--require-passed-gates)."""
    from llb.prep.ontology.artifacts import required_gate_names
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
