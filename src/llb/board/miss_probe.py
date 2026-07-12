"""Bounded retrieval-depth probe for the miss analysis (probe mode).

`llb analyze-misses --probe-top-k 3,8` re-runs ONLY the analyzed run's miss subset at the
requested alternative retrieval depths, so the retrieval hypothesis ("the gold span never
entered the context; a different `top_k` would fix it") is confirmed or rejected with measured
numbers instead of guessed. Each probe depth is a normal durable `run_eval` bundle (same model,
backend, and RAG config; only `top_k` and `run_name` differ), so a probe campaign inherits the
durable-eval-runner's journal + resume behavior:

- a finalized probe bundle (matched by its deterministic `run_name` + case count) is REUSED,
  never re-run;
- an interrupted probe's staged journal is found by its pinned config + goldset digests and
  resumed via `run_eval(..., resume=...)` instead of re-spending model calls;
- only then does a fresh probe run start.

`run_eval_fn` is injectable, so probe orchestration (reuse, resume, outcome math) is fully
unit-testable with fakes -- no backend or GPU.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llb.board.miss_analysis import MISS_RETRIEVAL, MissRecord
from llb.core.config import RUN_EVAL_METHOD, RunConfig
from llb.core.contracts import JsonObject
from llb.executor import durability
from llb.goldset.schema import GoldItem

_LOG = logging.getLogger(__name__)

PROBE_RUN_NAME_PREFIX = "miss-probe"


def probe_run_name(run_id: str, k: int) -> str:
    """Deterministic probe run name: ties every probe bundle to its source run and depth, so a
    re-invoked analysis finds and reuses finalized probes instead of re-running them."""
    return f"{PROBE_RUN_NAME_PREFIX}-{run_id}-k{k}"


def probe_config(manifest: JsonObject, k: int) -> RunConfig:
    """The probe's RunConfig: the analyzed run's recorded config with only the retrieval depth
    and run name changed (judge + telemetry off -- the probe measures retrieval, not quality
    blending or throughput)."""
    config = manifest.get("config") or {}
    fields = {key: value for key, value in config.items() if key in RunConfig.model_fields}
    fields.update(
        top_k=k,
        run_name=probe_run_name(str(manifest.get("run_id", "run")), k),
        judge_model=None,
        measure_telemetry=False,
    )
    return RunConfig.model_validate(fields)


def _read_jsonl(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _find_finalized(run_root: Path, name: str, n_items: int) -> Path | None:
    """A published probe bundle for this run name + subset size, newest first."""
    for manifest_path in sorted(run_root.glob("*/manifest.json"), reverse=True):
        if manifest_path.parent.name.startswith("."):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("run_name") == name and int(manifest.get("n_cases", -1)) == n_items:
            return manifest_path.parent
    return None


def _find_resumable(
    run_root: Path, config_fingerprint: JsonObject, items: list[GoldItem], split: str
) -> Path | None:
    """The canonical run dir of an interrupted probe whose pinned journal meta matches this
    probe's config + goldset digests -- the durable-eval-runner can resume it."""
    wanted_config = durability.config_digest(config_fingerprint)
    wanted_goldset = durability.goldset_digest(items)
    for meta_path in run_root.glob(f".*.tmp/{durability.JOURNAL_META_NAME}"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            meta.get("config_digest") == wanted_config
            and meta.get("goldset_digest") == wanted_goldset
            and meta.get("split") == split
        ):
            staging_name = meta_path.parent.name  # ".<timestamp>.tmp"
            return run_root / staging_name[1 : -len(".tmp")]
    return None


def _probe_outcome(
    probe_dir: Path,
    k: int,
    retrieval_miss_ids: set[str],
    base_mean_objective: float,
    *,
    reused: bool,
    resumed: bool,
) -> JsonObject:
    """Measured probe evidence from the probe bundle's scores: subset mean objective at depth k
    plus how many of the source run's retrieval misses the deeper/shallower context recovered."""
    rows = _read_jsonl(probe_dir / "scores.jsonl")
    objectives = [float(row.get("objective_score", 0.0)) for row in rows]
    recovered = sum(
        1
        for row in rows
        if str(row.get("item_id")) in retrieval_miss_ids
        and float(row.get("retrieval_hit", 0.0) or 0.0) > 0.0
    )
    return {
        "top_k": k,
        "n_items": len(rows),
        "mean_objective": sum(objectives) / len(objectives) if objectives else 0.0,
        "base_mean_objective": base_mean_objective,
        "recovered_retrieval": recovered,
        "n_retrieval_misses": len(retrieval_miss_ids),
        "run_dir": str(probe_dir),
        "reused": reused,
        "resumed": resumed,
    }


def run_probes(
    manifest: JsonObject,
    misses: list[MissRecord],
    items: list[GoldItem],
    ks: list[int],
    *,
    run_eval_fn: Callable[..., Any] | None = None,
) -> list[JsonObject]:
    """Execute (or reuse/resume) one probe run per requested depth over the miss subset.

    Returns one outcome record per depth, ready to attach to `MissAnalysis.probes`. Depths
    equal to the analyzed run's `top_k` are skipped (they would re-measure the baseline).
    """
    if not misses:
        return []
    if run_eval_fn is None:
        from llb.executor.runner import run_eval

        run_eval_fn = run_eval
    items_by_id = {item.id: item for item in items}
    subset = sorted(
        (items_by_id[m.item_id] for m in misses if m.item_id in items_by_id),
        key=lambda item: item.id,
    )
    if not subset:
        raise SystemExit(
            "[analyze-misses] none of the miss item ids exist in the goldset; "
            "pass the goldset the run actually scored (--goldset)"
        )
    retrieval_miss_ids = {m.item_id for m in misses if m.miss_class == MISS_RETRIEVAL}
    base_mean = sum(m.objective_score for m in misses) / len(misses)
    split = str(manifest.get("split", "final"))
    source_top_k = int((manifest.get("config") or {}).get("top_k", 0))

    outcomes: list[JsonObject] = []
    for k in ks:
        if k == source_top_k:
            _LOG.info("[analyze-misses] probe top_k=%d equals the run's top_k; skipping", k)
            continue
        probe_dir, reused, resumed = _probe_run_dir(manifest, k, subset, split, run_eval_fn)
        outcomes.append(
            _probe_outcome(
                probe_dir, k, retrieval_miss_ids, base_mean, reused=reused, resumed=resumed
            )
        )
    return outcomes


def _probe_run_dir(
    manifest: JsonObject,
    k: int,
    subset: list[GoldItem],
    split: str,
    run_eval_fn: Callable[..., Any],
) -> tuple[Path, bool, bool]:
    """`(probe run dir, reused, resumed)`: reuse a finalized run, else resume/launch one."""
    cfg = probe_config(manifest, k)
    run_root = cfg.data_dir / RUN_EVAL_METHOD
    probe_dir = _find_finalized(run_root, cfg.run_name, len(subset))
    if probe_dir is not None:
        _LOG.info("[analyze-misses] probe top_k=%d reuses %s", k, probe_dir)
        return probe_dir, True, False
    resume_dir = _find_resumable(run_root, cfg.fingerprint(), subset, split)
    if resume_dir is not None:
        _LOG.info("[analyze-misses] probe top_k=%d resumes %s", k, resume_dir)
    result = run_eval_fn(cfg, items=subset, split=split, resume=resume_dir, emit=False)
    return run_root / result["run_timestamp"], False, resume_dir is not None


def parse_probe_depths(spec: str) -> list[int]:
    """Parse the `--probe-top-k 3,8` depth list (positive ints, deduplicated, sorted)."""
    depths: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            raise SystemExit(f"[analyze-misses] --probe-top-k expects integers, got '{token}'")
        if value < 1:
            raise SystemExit("[analyze-misses] --probe-top-k depths must be >= 1")
        depths.add(value)
    if not depths:
        raise SystemExit("[analyze-misses] --probe-top-k needs at least one depth (e.g. 3,8)")
    return sorted(depths)
