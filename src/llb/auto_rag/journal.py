"""Atomic stage markers and append-only events for auto-RAG resume."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llb.core.fsutil import atomic_write_text

MANIFEST_FILE = "manifest.json"
JOURNAL_FILE = "journal.jsonl"
RESULT_FILE = "result.json"


def stable_digest(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


class AutoRagJournal:
    """Publish a stage only after all of its artifacts are durable."""

    def __init__(self, run_dir: Path, manifest: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.manifest = manifest
        self.fingerprint = stable_digest(manifest)

    def open(self) -> bool:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / MANIFEST_FILE
        if path.is_file():
            prior = json.loads(path.read_text(encoding="utf-8"))
            if prior.get("fingerprint") != self.fingerprint:
                raise ValueError(f"auto-RAG resume settings differ from {path}; use a new --run-id")
            return True
        payload = {"kind": "auto-rag", "fingerprint": self.fingerprint, **self.manifest}
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        self.event("run", "started")
        return False

    def result_path(self, stage: str) -> Path:
        return self.run_dir / "stages" / stage / RESULT_FILE

    def load(self, stage: str) -> dict[str, Any] | None:
        path = self.result_path(stage)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("status") != "completed" or payload.get("stage") != stage:
            return None
        result = payload.get("result")
        return result if isinstance(result, dict) else None

    def complete(self, stage: str, result: dict[str, Any]) -> None:
        payload = {"status": "completed", "stage": stage, "result": result}
        atomic_write_text(
            self.result_path(stage), json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
        self.event(stage, "completed", result_digest=stable_digest(result))

    def event(self, stage: str, status: str, **fields: object) -> None:
        path = self.run_dir / JOURNAL_FILE
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "status": status,
            **fields,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
