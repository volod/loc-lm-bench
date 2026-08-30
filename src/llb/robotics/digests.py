"""Canonical SHA-256 helpers for robotics contracts and fixtures."""

import hashlib
import json
from pathlib import Path
from typing import Any

DIGEST_PREFIX = "sha256:"


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON-compatible value in the canonical form used by every pin."""
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def value_digest(value: Any) -> str:
    return f"{DIGEST_PREFIX}{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def file_digest(path: Path) -> str:
    return f"{DIGEST_PREFIX}{hashlib.sha256(path.read_bytes()).hexdigest()}"
