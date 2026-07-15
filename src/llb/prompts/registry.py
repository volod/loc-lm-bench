"""Directory-backed prompt registry and generator."""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llb.core.contracts.common import ChatMessage
from llb.prompts.engine import PromptAugmentation, render_template
from llb.prompts.registry_generation import (
    DEFAULT_TEMPLATE_ROOT,
    REGISTRY_FILE,
    _load_json_object,
    generate_registry,
    write_registry,
)

_LOG = logging.getLogger(__name__)


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
