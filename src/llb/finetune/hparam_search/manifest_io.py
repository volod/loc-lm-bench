"""Write and read the hparams manifest: `_finish` finalizes one study into
`hparams_manifest.json` + `HparamSearchResult`; `latest_hparams_manifest` / `load_hparams_manifest`
/ `trainer_defaults` read the recorded best config back for self-improvement and campaign rounds.
"""

import json
import logging
from pathlib import Path
from typing import Any

from llb.bench.common import new_run_timestamp
from llb.core.contracts import JsonObject
from llb.core.fsutil import atomic_write_text
from llb.finetune.hparam_search.model import (
    HPARAMS_MANIFEST,
    HPARAMS_METHOD,
    STATE_COMPLETE,
    STATE_PRUNED,
    STUDY_DB,
    HparamSearchResult,
    TrialRecord,
)
from llb.finetune.naming import model_slug

_LOG = logging.getLogger(__name__)


def _finish(
    *,
    root: Path,
    model: str,
    dataset_dir: Path,
    dataset_manifest: JsonObject,
    dev_slice: Any,
    study: Any,
    trials: list[TrialRecord],
    seed: int,
    max_trials: int,
    max_hours: float | None,
    budget_exhausted: bool,
) -> HparamSearchResult:
    complete = [trial for trial in trials if trial.state == STATE_COMPLETE]
    best = max(complete, key=lambda trial: trial.objective or 0.0) if complete else None
    manifest: JsonObject = {
        "kind": "llb.finetune.hparams",
        "model": model,
        "dataset_dir": str(dataset_dir),
        "dataset_digest": dataset_manifest["dataset_digest"],
        "study_name": study.study_name,
        "study_seed": seed,
        "storage": f"sqlite:///{root / STUDY_DB}",
        "dev_slice": dev_slice.as_dict(),
        "max_trials": max_trials,
        "max_hours": max_hours,
        "budget_exhausted": budget_exhausted,
        "n_trials": len(trials),
        "n_complete": len(complete),
        "n_pruned": sum(1 for trial in trials if trial.state == STATE_PRUNED),
        "best_trial": best.number if best else None,
        "best_objective": best.objective if best else None,
        "best_hyperparameters": best.hyperparameters if best else None,
        "trials": [trial.as_dict() for trial in trials],
        "created_at": new_run_timestamp()[1],
    }
    manifest_path = root / HPARAMS_MANIFEST
    atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return HparamSearchResult(
        out_dir=root,
        manifest_path=manifest_path,
        dev_slice=dev_slice,
        trials=trials,
        best_trial=best.number if best else None,
        best_objective=best.objective if best else None,
        best_hyperparameters=best.hyperparameters if best else None,
        budget_exhausted=budget_exhausted,
    )


def latest_hparams_manifest(data_dir: Path | str, model: str) -> Path | None:
    """Newest `$DATA_DIR/finetune-hparams/<model>/*/hparams_manifest.json`, or None."""
    root = Path(data_dir) / HPARAMS_METHOD / model_slug(model)
    if not root.is_dir():
        return None
    for candidate in sorted(root.iterdir(), reverse=True):
        manifest = candidate / HPARAMS_MANIFEST
        if manifest.is_file():
            return manifest
    return None


def load_hparams_manifest(path: Path | str) -> JsonObject:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"hparams manifest must be a JSON object: {path}")
    return data


def trainer_defaults(data_dir: Path | str, model: str) -> JsonObject:
    """`train_adapter` kwargs from this model's recorded search, or `{}` when none exists.

    A recorded best config becomes the model's default for self-improvement and campaign rounds;
    the manifest path travels into `adapter_manifest.json` so a tuned row names the search that
    chose its hyperparameters.
    """
    path = latest_hparams_manifest(data_dir, model)
    if path is None:
        return {}
    try:
        manifest = load_hparams_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _LOG.warning("[finetune-hparams] ignoring unreadable manifest %s: %s", path, exc)
        return {}
    best = manifest.get("best_hyperparameters")
    if not isinstance(best, dict) or not best:
        return {}
    return {"hyperparameters": dict(best), "hparams_manifest": str(path)}
