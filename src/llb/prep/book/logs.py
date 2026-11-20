"""Structured JSON run log: every event as one JSON object per line in `logs/run.jsonl`.

Two loggers: `llb.book_ocr.events` carries the structured payload to the run's file handler and
never propagates, so the file holds every event at every level; `llb.book_ocr` carries a one-line
human summary to the console, where per-stage events sit at DEBUG (`LLB_LOG=debug` shows them) and
run and batch events at INFO, so a 1,000-page run does not print 1,000 lines per stage.
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llb.prep.book.constants import LOGS_DIRNAME, RUN_LOG_FILENAME

LOGGER_NAME = "llb.book_ocr"
_LOG = logging.getLogger(LOGGER_NAME)
_EVENTS = logging.getLogger(f"{LOGGER_NAME}.events")
_EVENTS.propagate = False


class _JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return str(record.msg)


def log_event(event: str, level: int = logging.DEBUG, **fields: Any) -> None:
    """Record one structured event; `fields` must be JSON-able."""
    payload = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event}
    payload.update(fields)
    if _EVENTS.handlers:
        _EVENTS.info(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))
    summary = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    _LOG.log(level, "[book-ocr] %s %s", event, summary)


@contextmanager
def run_log(run_dir: Path) -> Iterator[Path]:
    """Append this invocation's events to `logs/run.jsonl` while the block runs."""
    path = run_dir / LOGS_DIRNAME / RUN_LOG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(_JsonLinesFormatter())
    _EVENTS.addHandler(handler)
    _EVENTS.setLevel(logging.INFO)
    try:
        yield path
    finally:
        _EVENTS.removeHandler(handler)
        handler.close()
