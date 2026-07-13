"""Bundle directory + extraction-journal setup for a resumable draft run.

A fresh run pins its determinism-critical settings and endpoint identity to a meta sidecar (written
before any model call) and opens the extraction journal; `--resume` reads that sidecar back so the
run reproduces exactly, reusing journaled extraction windows instead of re-calling the model.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from llb.core.paths import resolve_data_dir
from llb.prep.ontology.constants import (
    EXTRACTION_JOURNAL_FILENAME,
    EXTRACTION_JOURNAL_META_FILENAME,
    EXTRACTION_JOURNAL_META_KIND,
    METHOD_DIR,
)
from llb.prep.ontology.endpoint_config import EndpointPlan
from llb.prep.ontology.journal import ExtractionJournal
from llb.prep.ontology.pipeline.settings import DraftSettings

_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime(_TIMESTAMP_FORMAT)


def default_out_dir() -> Path:
    return resolve_data_dir() / METHOD_DIR / _timestamp()


def _journal_meta_path(out_dir: Path) -> Path:
    return out_dir / EXTRACTION_JOURNAL_META_FILENAME


def _clear_fresh_extraction_journal(out_dir: Path) -> None:
    """Drop prior resumability state when the caller starts a fresh run in an existing bundle dir."""
    for name in (EXTRACTION_JOURNAL_FILENAME, EXTRACTION_JOURNAL_META_FILENAME):
        path = out_dir / name
        if path.exists():
            path.unlink()


def _write_journal_meta(out_dir: Path, pinned: dict[str, object], endpoints: EndpointPlan) -> None:
    """Record the determinism-critical settings + endpoint identity so a resume reproduces the run.

    Written once at the start of a fresh run (before any model call) so the sidecar survives a kill
    at any point during extraction.
    """
    payload = {
        "kind": EXTRACTION_JOURNAL_META_KIND,
        "endpoint": endpoints.config_provenance(),
        **pinned,
    }
    _journal_meta_path(out_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_journal_meta(out_dir: Path | str) -> dict[str, object]:
    """Read the journal meta sidecar for `--resume`. Raises a clear error when it is absent."""
    path = _journal_meta_path(Path(out_dir))
    if not path.is_file():
        raise ValueError(
            f"cannot resume: no {EXTRACTION_JOURNAL_META_FILENAME} in {out_dir} "
            "(a resumable draft writes it at the start of extraction)"
        )
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"malformed journal meta: {path}")
    return meta


def _prepare_bundle_dir(
    resolved_out: Path, settings: DraftSettings, endpoints: EndpointPlan, resume: bool
) -> ExtractionJournal:
    """Create the bundle dir, pin settings (fresh run only), and open the extraction journal."""
    resolved_out.mkdir(parents=True, exist_ok=True)
    if not resume:
        _clear_fresh_extraction_journal(resolved_out)
        _write_journal_meta(resolved_out, settings.pinned_payload(), endpoints)
    journal = ExtractionJournal(resolved_out / EXTRACTION_JOURNAL_FILENAME)
    journal.load()
    return journal
