"""Write the resolution bundle under `$DATA_DIR/graph-entity-resolution/<run>/`.

Split from `run.py` along the obvious seam: that half decides what the reading IS, this half
decides where it lands. The linkage seam's own bundle nests inside this one, so a proposed merge
can be traced from the lane number back to the pair and the level agreements behind it.

The pre-merge graph is copied in rather than referenced, because the reading is only redoable
without the overlay for as long as the graph it was taken over still exists in that state.
"""

import json
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.core.fsutil import atomic_write_text
from llb.graph.model import KnowledgeGraph
from llb.graph.resolution.constants import (
    COMPARISON_FILE,
    METHOD,
    OVERLAY_FILE_TEMPLATE,
    OVERLAYS_DIR,
    PRE_MERGE_DIR,
    RECORDS_FILE,
    REPORT_FILE,
    RESOLUTION_MODE,
    SUMMARY_FILE,
)
from llb.graph.resolution.overlay import NodeOverlay, write_overlay
from llb.graph.resolution.report import format_resolution_report
from llb.linkage.model import LinkageResult
from llb.rag.comparison.models import ComparisonReport

_LOG = logging.getLogger(__name__)


def bundle_dir(data_dir: Path, run: str | None = None) -> Path:
    """`$DATA_DIR/graph-entity-resolution/<run>/` -- a fresh timestamped run unless one is named."""
    from llb.bench.common import new_run_timestamp

    return Path(data_dir) / METHOD / (run or new_run_timestamp()[1])


def write_declined(out_dir: Path, summary: JsonObject) -> Path:
    """A declined run still publishes: an absent artifact and a stated reason are not the same."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / SUMMARY_FILE, summary)
    return out_dir


def write_resolution_artifacts(
    out_dir: Path,
    summary: JsonObject,
    reports: dict[str, ComparisonReport],
    result: LinkageResult,
    overlays: Sequence[NodeOverlay],
    graph: KnowledgeGraph,
    records: Sequence[JsonObject],
    graph_dir: Path | None,
) -> Path:
    """Write the linkage bundle, the record table, every overlay, and the paired reading."""
    from llb.linkage.artifacts import write_linkage_artifacts

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_linkage_artifacts(
        result,
        out_dir,
        {
            "mode": RESOLUTION_MODE,
            "n_nodes": len(graph.nodes),
            "policy": "proposed overlay only -- this run rewrites no stored graph",
        },
    )
    write_jsonl(out_dir / RECORDS_FILE, records)
    _write_overlays(out_dir / OVERLAYS_DIR, overlays, graph)
    write_json(out_dir / COMPARISON_FILE, {"strategies": reports})
    write_json(out_dir / SUMMARY_FILE, summary)
    atomic_write_text(out_dir / REPORT_FILE, format_resolution_report(summary))
    retain_pre_merge_graph(graph_dir, out_dir / PRE_MERGE_DIR)
    _LOG.info("[graph-resolution] wrote bundle -> %s", out_dir)
    return out_dir


def _write_overlays(
    overlay_dir: Path, overlays: Sequence[NodeOverlay], graph: KnowledgeGraph
) -> None:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for overlay in overlays:
        write_overlay(
            overlay,
            overlay_dir / OVERLAY_FILE_TEMPLATE.format(threshold=f"{overlay.threshold:g}"),
            graph,
        )


def retain_pre_merge_graph(graph_dir: Path | None, destination: Path) -> None:
    """Copy the graph the reading was taken over, so it can be redone without the overlay."""
    if graph_dir is None or not Path(graph_dir).is_dir():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(Path(graph_dir).glob("*")):
        if source.is_file():
            shutil.copy2(source, destination / source.name)


def write_json(path: Path, payload: JsonObject) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Sequence[JsonObject]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
