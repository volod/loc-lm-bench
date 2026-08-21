"""Write the edition-linkage lane into the audit bundle's `linkage/` subdirectory.

The seam's own bundle is written unchanged, so a linkage run inside an audit is read by exactly the
tools that read a standalone one -- including `--replay-from`, which re-scores the same documents
from the same model without re-fitting. Three files join it: the record table the fit read, the
edition groups it proposed, and the comparison against the thresholds the audit still decides by.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from llb.conflicts.linkage.constants import (
    EDITIONS_FILE,
    LINKAGE_MODE,
    RECORDS_FILE,
    SUMMARY_FILE,
)
from llb.conflicts.linkage.run import EditionLinkageRun
from llb.core.contracts.common import JsonObject
from llb.core.fsutil import atomic_write_text
from llb.linkage.artifacts import linkage_dir, write_linkage_artifacts


def _write_json(path: Path, payload: JsonObject) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Sequence[JsonObject]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def write_edition_linkage(out_dir: Path, run: EditionLinkageRun) -> dict[str, Path]:
    """Persist the lane; a declined run still writes its summary, because a reason is a result."""
    bundle = linkage_dir(Path(out_dir))
    bundle.mkdir(parents=True, exist_ok=True)
    _write_json(bundle / SUMMARY_FILE, run.summary)
    paths = {"edition_summary": bundle / SUMMARY_FILE}
    if run.declined or run.result is None:
        return paths
    write_linkage_artifacts(
        run.result,
        Path(out_dir),
        {
            "mode": LINKAGE_MODE,
            "n_documents": run.summary.get("n_documents"),
            "edition_cut": run.summary.get("cut"),
            "policy": (
                "a ranking and a proposed grouping only -- this run changes no finding, no "
                "relation, and no threshold"
            ),
        },
    )
    _write_jsonl(bundle / RECORDS_FILE, _record_rows(run))
    _write_jsonl(bundle / EDITIONS_FILE, [group.payload() for group in run.groups])
    paths["records"] = bundle / RECORDS_FILE
    paths["editions"] = bundle / EDITIONS_FILE
    return paths


def _record_rows(run: EditionLinkageRun) -> list[JsonObject]:
    """The record table with the two shingle arrays replaced by their sizes.

    The arrays are the fit's input and can run to hundreds of thousands of elements per document;
    what a reader of the bundle needs is which document carried how many, and the model beside it
    re-derives the rest from the corpus.
    """
    rows: list[JsonObject] = []
    for record in run.records:
        rows.append(
            {
                key: (len(value) if isinstance(value, list) else value)
                for key, value in record.items()
            }
        )
    return rows
