"""Production wiring: fit or replay an identity decision and publish it into a run bundle."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from llb.bench.common import new_run_timestamp
from llb.core.contracts.common import JsonObject
from llb.linkage.artifacts import (
    read_saved_model,
    read_saved_spec,
    read_saved_summary,
    write_linkage_artifacts,
)
from llb.linkage.constants import METHOD
from llb.linkage.engine import replay_linkage, run_linkage
from llb.linkage.model import LinkageResult
from llb.linkage.records import read_labels, read_records
from llb.linkage.spec import LinkageSpec, load_spec

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinkageRun:
    """One published identity decision: what it decided and where it landed."""

    result: LinkageResult
    out_dir: Path
    paths: dict[str, str]


def read_spec_file(path: Path) -> LinkageSpec:
    """Load a specification from a JSON file, validated."""
    return load_spec(json.loads(Path(path).read_text(encoding="utf-8")))


def bundle_dir(data_dir: Path, method: str = METHOD, run: str | None = None) -> Path:
    """`$DATA_DIR/<method>/<run>/` -- a fresh timestamped run unless one is named."""
    stamp = run or new_run_timestamp()[1]
    return Path(data_dir) / method / stamp


def link_records(
    records_path: Path,
    spec_path: Path,
    data_dir: Path,
    *,
    labels_path: Path | None = None,
    method: str = METHOD,
    run: str | None = None,
) -> LinkageRun:
    """Fit a linkage model over a JSONL record table and publish the bundle."""
    spec = read_spec_file(spec_path)
    records = read_records(records_path)
    labels = read_labels(labels_path, spec) if labels_path is not None else None
    result = run_linkage(records, spec, labels)
    return _publish(
        result,
        bundle_dir(data_dir, method, run),
        {
            "records": str(records_path),
            "specification": str(spec_path),
            "labels": str(labels_path) if labels_path is not None else None,
            "mode": "fit",
        },
    )


def replay_records(
    records_path: Path,
    source_bundle: Path,
    data_dir: Path,
    *,
    method: str = METHOD,
    run: str | None = None,
) -> LinkageRun:
    """Re-score a record table from a bundle's saved model, without re-fitting anything."""
    spec = read_saved_spec(source_bundle)
    model = read_saved_model(source_bundle)
    records = read_records(records_path)
    result = replay_linkage(
        records,
        spec,
        model,
        trained_from_labels=bool(read_saved_summary(source_bundle).get("trained_from_labels")),
    )
    return _publish(
        result,
        bundle_dir(data_dir, method, run),
        {"records": str(records_path), "replayed_from": str(source_bundle), "mode": "replay"},
    )


def _publish(result: LinkageResult, out_dir: Path, metadata: JsonObject) -> LinkageRun:
    paths = write_linkage_artifacts(result, out_dir, metadata)
    _LOG.info("[linkage] wrote bundle -> %s", out_dir)
    return LinkageRun(result=result, out_dir=out_dir, paths=paths)


def format_summary(result: LinkageResult, examples: int = 5) -> str:
    """A short operator-readable report: the counts, then the strongest proposed merges."""
    summary = result.summary()
    lines = [
        f"[linkage] records={summary['n_records']} scored_pairs={summary['n_scored_pairs']} "
        f"matched={summary['n_matched_pairs']} (threshold {summary['match_threshold']})",
        f"[linkage] clusters={summary['n_clusters']} "
        f"multi_record={summary['n_multi_record_clusters']} "
        f"largest={summary['largest_cluster']}",
        "[linkage] blocking rules:",
    ]
    lines.extend(
        f"  - {count.rule}: {count.post_filter} comparisons to score "
        f"({count.pre_filter} before filters)"
        for count in result.blocking_counts
    )
    if result.untrained_levels:
        lines.append(
            f"[linkage] {len(result.untrained_levels)} comparison level(s) had no estimate and "
            "used a default: "
            + ", ".join(f"{p.comparison}/{p.level}" for p in result.untrained_levels)
        )
    lines.append("[linkage] proposed identities (NOT applied):")
    lines.extend(_cluster_lines(result, examples))
    lines.append(
        "[linkage] these are same-record probabilities, not contradiction verdicts; adopt a "
        "merge on retrieval or review evidence, never on the probability alone"
    )
    return "\n".join(lines)


def _cluster_lines(result: LinkageResult, examples: int) -> list[str]:
    clusters = result.multi_record_clusters[:examples]
    if not clusters:
        return ["  - none at this threshold"]
    return [f"  - {c.cluster_id}: {', '.join(c.record_ids)}" for c in clusters]


def format_accuracy(result: LinkageResult) -> str:
    """The labelled precision/recall curve and the operating point it supports."""
    if not result.accuracy:
        return (
            "[linkage] no reviewer labels supplied: the output is a RANKED CANDIDATE LIST with no "
            "operating point, on the same terms as an unlabelled conflict tier"
        )
    lines = ["[linkage] labelled accuracy curve (threshold: precision / recall / f1):"]
    lines.extend(
        f"  - {point.threshold:.6f}: {point.precision:.3f} / {point.recall:.3f} / "
        f"{point.f1:.3f}  (tp={point.true_positives} fp={point.false_positives} "
        f"fn={point.false_negatives})"
        for point in result.accuracy
    )
    best = result.best_accuracy()
    if best is not None:
        lines.append(
            f"[linkage] best f1 on the curve is at threshold {best.threshold:.6f} "
            f"(precision {best.precision:.3f}, recall {best.recall:.3f})"
        )
    lines.extend(_operating_lines(result))
    return "\n".join(lines)


def _operating_lines(result: LinkageResult) -> list[str]:
    """What this run's own cut scored on the labels, pairwise and after clustering."""
    labelled = (
        ("pairwise", result.pair_operating_point),
        ("after clustering", result.cluster_operating_point),
    )
    return [
        f"[linkage] this run cut at {point.threshold}, {name}: precision {point.precision:.3f}, "
        f"recall {point.recall:.3f} (tp={point.true_positives} fp={point.false_positives} "
        f"fn={point.false_negatives} tn={point.true_negatives})"
        for name, point in labelled
        if point is not None
    ]


def format_pairs(result: LinkageResult, examples: int = 5) -> str:
    """The top scored pairs with the per-level agreement that produced each probability."""
    lines = ["[linkage] strongest pairs:"]
    for pair in result.pairs[:examples]:
        agreement = ", ".join(f"{k}={v}" for k, v in sorted(pair.agreement.items()))
        lines.append(
            f"  - {pair.left_id} ~ {pair.right_id}: p={pair.match_probability:.4f} "
            f"weight={pair.match_weight:.2f} [{agreement}]"
        )
    if len(result.pairs) == 0:
        lines.append("  - none: no blocking rule generated a scored comparison")
    return "\n".join(lines)
