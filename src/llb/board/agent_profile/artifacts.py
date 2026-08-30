"""Small shared readers for the per-lane roots the profile composes from.

Every lane writes a timestamped run directory and a JSON payload inside it, so the profile needs
exactly three things from each: the newest directory, a readable payload, and the moment the
reading was taken. Freshness comes from the directory NAME first -- the bundle convention already
encodes the run instant, and a file copied between hosts keeps that name while losing its mtime.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from llb.core.contracts.common import JsonObject

_LOG = logging.getLogger(__name__)

# `20260828T081146.273571Z-51d163a555e5` and the older `20260724T073314Z` both appear on disk.
_DIR_TIMESTAMP_FORMATS = ("%Y%m%dT%H%M%S.%fZ", "%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> JsonObject | None:
    """A JSON object at `path`, or None when it is missing, unreadable, or not an object.

    An ABSENT file is silent: a lane root holds bundles from several studies, and scanning it for
    the one artifact this profile reads finds far more misses than hits. A file that exists but
    does not parse is a real problem and is logged.
    """
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("[agent-profile] unreadable artifact %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def newest_payload(root: Path, filename: str) -> tuple[Path, JsonObject] | None:
    """The newest `<root>/<run>/<filename>` that parses, or None when the lane never ran here.

    Newest by the run TIMESTAMP in the directory name -- the bundle convention encodes the run
    instant, and sorting on it keeps the answer stable when a lane root is copied between hosts. A
    hand-named directory beside the bundles (a kept legacy result) carries no run instant, so it
    sorts LAST: alphabetical order would otherwise let `cardgate-legacy` outrank every 2026 bundle.
    """
    if not root.is_dir():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    ordered = sorted(
        candidates,
        key=lambda run_dir: (_parse_dir_timestamp(run_dir.name) or "", run_dir.name),
        reverse=True,
    )
    for run_dir in ordered:
        payload = read_json(run_dir / filename)
        if payload is not None:
            return run_dir / filename, payload
    return None


def artifact_timestamp(path: Path) -> str | None:
    """When the reading at `path` was taken, as an ISO-8601 UTC string.

    The run directory's own name wins; a directory that does not follow the bundle convention
    falls back to the file's modification time. None when neither can be read.
    """
    for candidate in (path.parent.name, path.name):
        stamp = _parse_dir_timestamp(candidate)
        if stamp is not None:
            return stamp
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def _parse_dir_timestamp(name: str) -> str | None:
    head = name.split("-", 1)[0]
    for fmt in _DIR_TIMESTAMP_FORMATS:
        try:
            parsed = datetime.strptime(head, fmt)
        except ValueError:
            continue
        return parsed.replace(tzinfo=timezone.utc).isoformat()
    return None


def age_days(measured_at: str | None, now: datetime) -> float | None:
    """Whole-ish days between a recorded reading and `now`; None when the reading has no stamp."""
    if not measured_at:
        return None
    try:
        taken = datetime.fromisoformat(measured_at)
    except ValueError:
        return None
    if taken.tzinfo is None:
        taken = taken.replace(tzinfo=timezone.utc)
    return round((now - taken).total_seconds() / 86400.0, 2)
