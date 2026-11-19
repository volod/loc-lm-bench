"""The source PDF: its identity, the policy a run holds it under, and the checks around it.

The source is identified by the PDF lane's own SHA-256 fingerprint and is only ever opened for
reading. A run refuses to write inside the directory that holds the source, a copied
source is made read-only, and a resume re-hashes whatever it reads and refuses a mismatch -- so a
changed, truncated, or swapped PDF can never feed a ledger built from another one.
"""

import os
import shutil
import stat
from pathlib import Path

from llb.core.contracts.data_prep.book import BookSourceRecord, SourcePolicy
from llb.prep.book.constants import (
    SOURCE_DIRNAME,
    SOURCE_POLICY_COPY,
    SOURCE_POLICY_REFERENCE,
    SOURCE_POLICY_SYMLINK,
)
from llb.prep.pdf.reuse import _sha256_file

_READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


class SourceError(ValueError):
    """The source PDF cannot be used, or no longer matches the run that recorded it."""


def sha256_file(path: Path) -> str:
    """The PDF lane's source fingerprint: hex SHA-256 of the file bytes."""
    return _sha256_file(path)


def refuse_source_overlap(source: Path, run_dir: Path) -> None:
    """Refuse a run directory inside the source's own directory, or one holding the source.

    The directory that holds the source is the operator's library, not a workspace: a run never
    writes beside the scan, and never into a tree whose cleanup could reach it.
    """
    source_real = source.resolve()
    run_real = run_dir.resolve()
    library = source_real.parent
    if run_real == library or library in run_real.parents or run_real in source_real.parents:
        raise SourceError(
            f"run directory {run_dir} overlaps {library}, the directory holding the source; a "
            "run never writes into the source's tree -- choose another --out-dir"
        )


def stored_source_path(run_dir: Path, record: BookSourceRecord) -> Path:
    """Where the run reads its source from under the recorded policy."""
    if record.policy == SOURCE_POLICY_REFERENCE:
        return Path(record.stored_path)
    return run_dir / record.stored_path


def place_source(
    source: Path, run_dir: Path, policy: SourcePolicy, sha256: str, page_count: int
) -> BookSourceRecord:
    """Hold the source in the run under `policy` and verify what was placed hashes to `sha256`."""
    source = source.resolve()
    refuse_source_overlap(source, run_dir)
    if policy == SOURCE_POLICY_REFERENCE:
        stored = str(source)
    else:
        rel = Path(SOURCE_DIRNAME) / source.name
        target = run_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if policy == SOURCE_POLICY_COPY:
            shutil.copyfile(source, target)
            os.chmod(target, _READ_ONLY)
        elif policy == SOURCE_POLICY_SYMLINK:
            target.symlink_to(source)
        else:
            raise SourceError(f"unknown source policy {policy!r}")
        stored = rel.as_posix()
    record = BookSourceRecord(
        original_path=str(source),
        sha256=sha256,
        size_bytes=source.stat().st_size,
        page_count=page_count,
        policy=policy,
        stored_path=stored,
    )
    verify_source(run_dir, record)
    return record


def verify_source(run_dir: Path, record: BookSourceRecord, given: Path | None = None) -> Path:
    """Re-hash the source a resume would read (and `given`, if passed); refuse any mismatch."""
    stored = stored_source_path(run_dir, record)
    candidates = [stored] if given is None else [given, stored]
    for path in candidates:
        if not path.is_file():
            raise SourceError(f"source PDF {path} is missing (recorded sha256 {record.sha256})")
        actual = sha256_file(path)
        if actual != record.sha256:
            raise SourceError(
                f"source PDF {path} has sha256 {actual}, but this run was built from "
                f"{record.sha256}; refusing to mix pages of two documents"
            )
    return stored if given is None else given
