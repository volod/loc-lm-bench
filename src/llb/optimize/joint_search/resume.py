"""Resume markers for joint-search: screen cells + finalist tune results."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, cast

from llb.core.contracts.runs import EvalResult
from llb.optimize.joint_search.constants import (
    FINALIST_RESULT_FILE,
    MARKER_STATUS_DONE,
    OPTUNA_METHOD,
    PICKS_DIR,
    SCREEN_DIR,
)
from llb.optimize.joint_search.halving import ScreenScore
from llb.optimize.joint_search.hooks import slug
from llb.optimize.joint_search.models import FinalistTuneResult

_LOG = logging.getLogger(__name__)

_SCREEN_REQUIRED = frozenset({"status", "name", "quality", "round_index", "case_limit"})
_FINALIST_REQUIRED = frozenset(
    {"status", "name", "backend", "source", "study_name", "overrides_by_pick", "finals"}
)
_PICK_REQUIRED = frozenset({"status", "goal", "result"})


def screen_marker_path(run_dir: Path, name: str, round_index: int) -> Path:
    """``$DATA_DIR/joint-search/<run>/screen/<slug>-r<round>.json``."""
    return run_dir / SCREEN_DIR / f"{slug(name)}-r{round_index}.json"


def write_screen_marker(
    run_dir: Path,
    score: ScreenScore,
    *,
    round_index: int,
    case_limit: int,
) -> Path:
    """Atomically persist a completed screen cell so a resume skips re-evaluation."""
    path = screen_marker_path(run_dir, score.name, round_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": MARKER_STATUS_DONE,
        "name": score.name,
        "quality": score.quality,
        "latency_s": score.latency_s,
        "backend": score.backend,
        "source": score.source,
        "round_index": round_index,
        "case_limit": case_limit,
    }
    _atomic_write_json(path, payload)
    return path


def read_screen_marker(run_dir: Path, name: str, round_index: int) -> ScreenScore | None:
    """Load a completed screen marker, or None if missing / truncated / invalid."""
    path = screen_marker_path(run_dir, name, round_index)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("[joint-search] ignore unreadable screen marker %s: %s", path, exc)
        return None
    if (
        not isinstance(value, dict)
        or value.get("status") != MARKER_STATUS_DONE
        or not _SCREEN_REQUIRED.issubset(value)
        or value.get("name") != name
        or value.get("round_index") != round_index
    ):
        _LOG.warning("[joint-search] ignore invalid screen marker %s", path)
        return None
    return ScreenScore(
        name=str(value["name"]),
        quality=float(value["quality"]),
        latency_s=float(value.get("latency_s") or 0.0),
        backend=str(value.get("backend") or ""),
        source=str(value.get("source") or ""),
    )


def finalist_result_path(cell_dir: Path) -> Path:
    """``$DATA_DIR/joint-search/<run>/finalists/<slug>/result.json``."""
    return cell_dir / FINALIST_RESULT_FILE


def write_finalist_result(cell_dir: Path, result: FinalistTuneResult) -> Path:
    """Atomically persist a finished finalist tune (study id + final-split picks)."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    path = finalist_result_path(cell_dir)
    payload = {
        "status": MARKER_STATUS_DONE,
        "name": result.name,
        "backend": result.backend,
        "source": result.source,
        "study_name": result.study_name,
        "overrides_by_pick": result.overrides_by_pick,
        "finals": result.finals,
    }
    _atomic_write_json(path, payload)
    return path


def read_finalist_result(cell_dir: Path) -> FinalistTuneResult | None:
    """Load a finished finalist result, or None if missing / truncated / invalid."""
    path = finalist_result_path(cell_dir)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("[joint-search] ignore unreadable finalist result %s: %s", path, exc)
        return None
    if (
        not isinstance(value, dict)
        or value.get("status") != MARKER_STATUS_DONE
        or not _FINALIST_REQUIRED.issubset(value)
    ):
        _LOG.warning("[joint-search] ignore invalid finalist result %s", path)
        return None
    overrides = value["overrides_by_pick"]
    finals = value["finals"]
    if not isinstance(overrides, dict) or not isinstance(finals, dict):
        _LOG.warning("[joint-search] ignore invalid finalist result %s", path)
        return None
    return FinalistTuneResult(
        name=str(value["name"]),
        backend=str(value["backend"]),
        source=str(value["source"]),
        study_name=str(value["study_name"]),
        overrides_by_pick={str(k): dict(v) for k, v in overrides.items() if isinstance(v, dict)},
        finals=finals,
        report_dir=cell_dir,
    )


def remaining_optuna_trials(data_dir: Path, study_name: str, n_trials: int) -> int:
    """How many Optuna trials are still needed for ``study_name`` (0 when already complete)."""
    if n_trials < 1:
        return 0
    db_path = Path(data_dir) / OPTUNA_METHOD / f"{study_name}.db"
    if not db_path.is_file():
        return n_trials
    import optuna

    storage = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        return n_trials
    return max(0, n_trials - len(study.trials))


def study_name_for(run_id: str, model_name: str) -> str:
    """Stable Optuna study id for one finalist under a joint-search run."""
    return f"joint-{run_id}-{slug(model_name)}"


def pick_marker_path(cell_dir: Path, goal: str) -> Path:
    """``$DATA_DIR/joint-search/<run>/finalists/<slug>/picks/<goal>.json``."""
    return cell_dir / PICKS_DIR / f"{slug(goal)}.json"


def write_pick_marker(cell_dir: Path, goal: str, result: EvalResult) -> Path:
    """Atomically persist one finished final-split pick eval."""
    path = pick_marker_path(cell_dir, goal)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": MARKER_STATUS_DONE, "goal": goal, "result": result}
    _atomic_write_json(path, payload)
    return path


def read_pick_marker(cell_dir: Path, goal: str) -> EvalResult | None:
    """Load a finished final-split pick marker, or None if missing / truncated / invalid."""
    path = pick_marker_path(cell_dir, goal)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("[joint-search] ignore unreadable pick marker %s: %s", path, exc)
        return None
    if (
        not isinstance(value, dict)
        or value.get("status") != MARKER_STATUS_DONE
        or not _PICK_REQUIRED.issubset(value)
        or value.get("goal") != goal
        or not isinstance(value.get("result"), dict)
    ):
        _LOG.warning("[joint-search] ignore invalid pick marker %s", path)
        return None
    return cast(EvalResult, value["result"])


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish JSON atomically so a kill cannot leave a false-complete marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp:
            json.dump(payload, temp, indent=2, sort_keys=True, default=str)
            temp.write("\n")
            temp_path = Path(temp.name)
        temp_path.replace(path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
