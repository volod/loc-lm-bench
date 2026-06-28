"""Directory-backed prompt registry and generator."""

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llb.contracts import ChatMessage
from llb.prompts.engine import PromptAugmentation, render_template

_LOG = logging.getLogger(__name__)

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


@dataclass(slots=True)
class PromptRegistry:
    """Runtime renderer backed by a generated prompt-template registry."""

    root: Path
    entries: dict[str, dict[str, Any]]
    _cache: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path | str = DEFAULT_TEMPLATE_ROOT) -> "PromptRegistry":
        root = Path(root)
        path = root / REGISTRY_FILE
        payload = _load_json_object(path) if path.is_file() else generate_registry(root)
        templates = payload.get("templates")
        if not isinstance(templates, dict):
            raise ValueError(f"{path}: expected a templates object")
        entries = {
            str(key): dict(value) for key, value in templates.items() if isinstance(value, dict)
        }
        return cls(root=root, entries=entries)

    def _entry(self, template_id: str, kind: str) -> dict[str, Any]:
        entry = self.entries.get(template_id)
        if entry is None:
            raise KeyError(f"unknown prompt template id: {template_id}")
        if entry.get("kind") != kind:
            raise TypeError(f"{template_id}: expected kind {kind!r}, got {entry.get('kind')!r}")
        return entry

    def _read(self, spec: dict[str, str]) -> str:
        rel = spec["path"]
        cached = self._cache.get(rel)
        if cached is not None:
            return cached
        text = (self.root / rel).read_text(encoding="utf-8")
        self._cache[rel] = text
        return text

    def render_text(self, template_id: str, values: dict[str, Any] | None = None) -> str:
        entry = self._entry(template_id, "text")
        spec = entry.get("template")
        if not isinstance(spec, dict):
            raise ValueError(f"{template_id}: malformed text template entry")
        return render_template(self._read(spec), values)

    def render_text_list(self, template_id: str, values: dict[str, Any] | None = None) -> list[str]:
        entry = self._entry(template_id, "text_list")
        specs = entry.get("templates")
        if not isinstance(specs, list):
            raise ValueError(f"{template_id}: malformed text_list template entry")
        return [
            render_template(self._read(spec), values) for spec in specs if isinstance(spec, dict)
        ]

    def render_text_map(
        self, template_id: str, values: dict[str, Any] | None = None
    ) -> dict[str, str]:
        entry = self._entry(template_id, "text_map")
        specs = entry.get("templates")
        if not isinstance(specs, dict):
            raise ValueError(f"{template_id}: malformed text_map template entry")
        return {
            str(key): render_template(self._read(spec), values)
            for key, spec in specs.items()
            if isinstance(spec, dict)
        }

    def render_chat(
        self,
        template_id: str,
        values: dict[str, Any] | None = None,
        augmentation: PromptAugmentation | None = None,
    ) -> list[ChatMessage]:
        entry = self._entry(template_id, "chat")
        specs = entry.get("messages")
        if not isinstance(specs, list):
            raise ValueError(f"{template_id}: malformed chat template entry")
        messages: list[ChatMessage] = []
        for spec in specs:
            if not isinstance(spec, dict) or not isinstance(spec.get("template"), dict):
                raise ValueError(f"{template_id}: malformed chat message entry")
            role = str(spec.get("role", "user"))
            content = render_template(self._read(spec["template"]), values)
            if augmentation is not None and role == "system":
                content = augmentation.apply_system(content)
            elif augmentation is not None and role == "user":
                content = augmentation.apply_user(content)
            messages.append({"role": role, "content": content})
        return messages


_DEFAULT_REGISTRY: PromptRegistry | None = None


def default_registry() -> PromptRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = PromptRegistry.load()
    return _DEFAULT_REGISTRY


def render_text(template_id: str, values: dict[str, Any] | None = None) -> str:
    return default_registry().render_text(template_id, values)


def render_text_list(template_id: str, values: dict[str, Any] | None = None) -> list[str]:
    return default_registry().render_text_list(template_id, values)


def render_text_map(template_id: str, values: dict[str, Any] | None = None) -> dict[str, str]:
    return default_registry().render_text_map(template_id, values)


def render_chat(
    template_id: str,
    values: dict[str, Any] | None = None,
    augmentation: PromptAugmentation | None = None,
) -> list[ChatMessage]:
    return default_registry().render_chat(template_id, values, augmentation)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the llb prompt-template registry.")
    parser.add_argument("--root", type=Path, default=DEFAULT_TEMPLATE_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = write_registry(args.root, args.out)
    _LOG.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
