"""Shared prompt-template rendering API."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from llb.contracts import ChatMessage
from llb.prompts.engine import PromptAugmentation, render_template

if TYPE_CHECKING:
    from llb.prompts.registry import PromptRegistry


def __getattr__(name: str) -> Any:
    if name == "PromptRegistry":
        from llb.prompts.registry import PromptRegistry

        return PromptRegistry
    raise AttributeError(name)


def default_registry() -> "PromptRegistry":
    from llb.prompts.registry import default_registry as load_default

    return load_default()


def generate_registry(root: Path | str | None = None) -> dict[str, Any]:
    from llb.prompts.registry import DEFAULT_TEMPLATE_ROOT, generate_registry as generate

    return generate(DEFAULT_TEMPLATE_ROOT if root is None else root)


def write_registry(root: Path | str | None = None, out_path: Path | str | None = None) -> Path:
    from llb.prompts.registry import DEFAULT_TEMPLATE_ROOT, write_registry as write

    return write(DEFAULT_TEMPLATE_ROOT if root is None else root, out_path)


def render_text(template_id: str, values: dict[str, Any] | None = None) -> str:
    from llb.prompts.registry import render_text as render

    return render(template_id, values)


def render_text_list(template_id: str, values: dict[str, Any] | None = None) -> list[str]:
    from llb.prompts.registry import render_text_list as render

    return render(template_id, values)


def render_text_map(template_id: str, values: dict[str, Any] | None = None) -> dict[str, str]:
    from llb.prompts.registry import render_text_map as render

    return render(template_id, values)


def render_chat(
    template_id: str,
    values: dict[str, Any] | None = None,
    augmentation: PromptAugmentation | None = None,
) -> list[ChatMessage]:
    from llb.prompts.registry import render_chat as render

    return render(template_id, values, augmentation)


__all__ = [
    "PromptAugmentation",
    "PromptRegistry",
    "default_registry",
    "generate_registry",
    "render_chat",
    "render_template",
    "render_text",
    "render_text_list",
    "render_text_map",
    "write_registry",
]
