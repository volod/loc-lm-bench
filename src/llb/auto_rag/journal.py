"""Atomic stage markers and append-only events for auto-RAG resume."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llb.artifacts.errors import ArtifactContractError
from llb.artifacts.run_bundle.auto_rag import read_auto_rag_manifest, read_stage_result
from llb.core.contracts.run_bundle.auto_rag import (
    AutoRagJournalEvent,
    AutoRagManifestDocument,
    AutoRagStageResult,
)
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
            # Read the prior manifest through its contract: a resume must compare against a
            # record this build can still read, not against whatever keys the file happens to
            # hold, or a fingerprint would match on a manifest whose settings no longer parse.
            if read_auto_rag_manifest(path).fingerprint != self.fingerprint:
                raise ValueError(f"auto-RAG resume settings differ from {path}; use a new --run-id")
            return True
        payload = AutoRagManifestDocument.model_validate(
            {"fingerprint": self.fingerprint, **self.manifest}
        ).model_dump(mode="json")
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
            payload = read_stage_result(path)
        except (OSError, ArtifactContractError):
            return None
        return payload.result if payload.stage == stage else None

    def complete(self, stage: str, result: dict[str, Any]) -> None:
        payload = AutoRagStageResult(stage=stage, result=result).model_dump(mode="json")
        atomic_write_text(
            self.result_path(stage), json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
        self.event(stage, "completed", result_digest=stable_digest(result))

    def event(self, stage: str, status: str, **fields: object) -> None:
        path = self.run_dir / JOURNAL_FILE
        record = AutoRagJournalEvent.model_validate(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "status": status,
                **{key: str(value) for key, value in fields.items()},
            }
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
