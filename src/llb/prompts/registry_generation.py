"""Focused registry generation implementation."""

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates"

REGISTRY_FILE = "registry.json"

DESCRIPTOR_SUFFIX = ".prompt.json"


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _required_str(payload: dict[str, Any], key: str, *, source: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: missing string field {key!r}")
    return value


def _template_ref(root: Path, descriptor: Path, value: str) -> dict[str, str]:
    path = (descriptor.parent / value).resolve()
    try:
        rel = path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{descriptor}: template escapes template root: {value}") from exc
    if not path.is_file():
        raise ValueError(f"{descriptor}: missing template file {value}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": rel.as_posix(), "sha256": digest}


def _text_entry(root: Path, descriptor: Path, payload: dict[str, Any]) -> dict[str, Any]:
    template = _required_str(payload, "template", source=descriptor)
    return {"template": _template_ref(root, descriptor, template)}


def _text_list_entry(root: Path, descriptor: Path, payload: dict[str, Any]) -> dict[str, Any]:
    values = payload.get("templates")
    if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
        raise ValueError(f"{descriptor}: field 'templates' must be a non-empty string list")
    return {"templates": [_template_ref(root, descriptor, value) for value in values]}


def _text_map_entry(root: Path, descriptor: Path, payload: dict[str, Any]) -> dict[str, Any]:
    values = payload.get("templates")
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{descriptor}: field 'templates' must be a non-empty object")
    refs: dict[str, dict[str, str]] = {}
    for key, value in values.items():
        if not isinstance(value, str):
            raise ValueError(f"{descriptor}: text_map value for {key!r} must be a string")
        refs[str(key)] = _template_ref(root, descriptor, value)
    return {"templates": refs}


def _chat_entry(root: Path, descriptor: Path, payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{descriptor}: field 'messages' must be a non-empty list")
    refs: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"{descriptor}: every message must be an object")
        role = _required_str(message, "role", source=descriptor)
        template = _required_str(message, "template", source=descriptor)
        refs.append({"role": role, "template": _template_ref(root, descriptor, template)})
    return {"messages": refs}


def generate_registry(root: Path | str = DEFAULT_TEMPLATE_ROOT) -> dict[str, Any]:
    """Scan `*.prompt.json` descriptors and return the generated registry payload."""
    root = Path(root)
    entries: dict[str, dict[str, Any]] = {}
    for descriptor in sorted(root.rglob(f"*{DESCRIPTOR_SUFFIX}")):
        payload = _load_json_object(descriptor)
        template_id = _required_str(payload, "id", source=descriptor)
        kind = _required_str(payload, "kind", source=descriptor)
        if template_id in entries:
            raise ValueError(f"duplicate prompt template id: {template_id}")
        entry: dict[str, Any] = {
            "kind": kind,
            "descriptor": descriptor.relative_to(root).as_posix(),
        }
        if kind == "text":
            entry.update(_text_entry(root, descriptor, payload))
        elif kind == "text_list":
            entry.update(_text_list_entry(root, descriptor, payload))
        elif kind == "text_map":
            entry.update(_text_map_entry(root, descriptor, payload))
        elif kind == "chat":
            entry.update(_chat_entry(root, descriptor, payload))
        else:
            raise ValueError(f"{descriptor}: unsupported prompt kind {kind!r}")
        entries[template_id] = entry
    return {"version": 1, "templates": entries}


def write_registry(
    root: Path | str = DEFAULT_TEMPLATE_ROOT, out_path: Path | str | None = None
) -> Path:
    """Generate and write `registry.json` for a template directory."""
    root = Path(root)
    out = Path(out_path) if out_path is not None else root / REGISTRY_FILE
    payload = generate_registry(root)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out
