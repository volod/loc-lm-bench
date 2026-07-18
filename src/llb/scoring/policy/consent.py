"""Upfront egress consent record for the frontier scorer lane."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llb.scoring.policy.errors import ScorerPolicyError

CONSENT_FILENAME = "consent.json"


@dataclass(frozen=True)
class ConsentRecord:
    """One explicit operator approval to send eval answers to a frontier judge."""

    model: str
    approved: bool
    max_usd: float | None
    max_calls: int | None
    recorded_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ConsentRecord":
        return cls(
            model=str(raw["model"]),
            approved=bool(raw["approved"]),
            max_usd=_optional_float(raw.get("max_usd")),
            max_calls=_optional_int(raw.get("max_calls")),
            recorded_at=str(raw["recorded_at"]),
        )


def scorer_dir(run_dir: Path) -> Path:
    """Artifact root for consent + cost ledger under one run."""
    return run_dir / "scorer"


def consent_path(run_dir: Path) -> Path:
    return scorer_dir(run_dir) / CONSENT_FILENAME


def record_consent(
    run_dir: Path,
    *,
    model: str,
    approved: bool,
    max_usd: float | None,
    max_calls: int | None,
) -> ConsentRecord:
    """Persist the consent decision; refuse unsigned frontier scoring."""
    if not approved:
        raise ScorerPolicyError("frontier scorer requires explicit egress consent")
    if max_usd is None and max_calls is None:
        raise ScorerPolicyError("frontier scorer requires max_usd or max_calls")
    if max_usd is not None and max_usd <= 0:
        raise ScorerPolicyError("max_usd must be > 0 when set")
    if max_calls is not None and max_calls < 1:
        raise ScorerPolicyError("max_calls must be >= 1 when set")
    record = ConsentRecord(
        model=model,
        approved=True,
        max_usd=max_usd,
        max_calls=max_calls,
        recorded_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    dest = consent_path(run_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8")
    return record


def load_consent(run_dir: Path) -> ConsentRecord | None:
    path = consent_path(run_dir)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ScorerPolicyError(f"consent file is not a mapping: {path}")
    return ConsentRecord.from_dict(raw)


def require_consent(run_dir: Path, *, model: str) -> ConsentRecord:
    """Load an existing approved consent or raise."""
    record = load_consent(run_dir)
    if record is None or not record.approved:
        raise ScorerPolicyError(
            f"frontier scorer missing approved consent under {consent_path(run_dir)}"
        )
    if record.model != model:
        raise ScorerPolicyError(
            f"consent model {record.model!r} does not match frontier judge {model!r}"
        )
    return record


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
