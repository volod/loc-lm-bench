"""Load only actionable fields from one composed agent operating profile."""

import json
from pathlib import Path
from typing import Any

from llb.robotics.digests import file_digest


def latest_profile(data_dir: Path) -> Path:
    candidates = list((data_dir / "agent-profile").glob("*/agent_profile.json"))
    if not candidates:
        raise ValueError("no composed agent profile found; run `make recommend-agent-profile`")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_measured_profile(path: Path, *, model: str, backend: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = payload["fields"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{path}: invalid composed agent profile -- {exc}") from None
    if not isinstance(fields, dict):
        raise ValueError(f"{path}: composed agent profile fields must be an object")
    measured: dict[str, Any] = {
        name: field.get("value")
        for name, field in fields.items()
        if isinstance(field, dict) and field.get("state") == "measured"
    }
    for name, requested in (("model", model), ("backend", backend)):
        if name in measured and measured[name] != requested:
            raise ValueError(
                f"robotics benchmark {name} {requested!r} conflicts with measured profile value "
                f"{measured[name]!r}"
            )
    adapter = measured.get("adapter")
    if adapter not in (None, "none"):
        raise ValueError(
            "robotics benchmark has no adapter-serving lane; measured adapter must be none"
        )
    return {
        "path": str(path),
        "sha256": file_digest(path),
        "generated_at": payload.get("generated_at"),
        "measured_fields": measured,
        "excluded_field_states": {
            name: field.get("state")
            for name, field in fields.items()
            if isinstance(field, dict) and field.get("state") != "measured"
        },
    }
