"""Small prompt-template renderer shared by benchmark, prep, and eval prompts."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_PLACEHOLDER = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*}}")


def _lookup(values: Mapping[str, Any], name: str) -> Any:
    current: Any = values
    for part in name.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                raise KeyError(name)
            current = current[part]
        else:
            if not hasattr(current, part):
                raise KeyError(name)
            current = getattr(current, part)
    return current


def render_template(template: str, values: Mapping[str, Any] | None = None) -> str:
    """Render `{{ name }}` placeholders from a mapping.

    The syntax is deliberately small: dotted lookup is supported, and missing values fail fast.
    This keeps prompt text reviewable in files without adding a runtime template dependency.
    """
    values = values or {}

    def replace(match: re.Match[str]) -> str:
        return str(_lookup(values, match.group(1)))

    return _PLACEHOLDER.sub(replace, template).removesuffix("\n")


@dataclass(frozen=True, slots=True)
class PromptAugmentation:
    """Optional prompt text added around rendered chat messages."""

    system_prefix: str = ""
    system_suffix: str = ""
    user_prefix: str = ""
    user_suffix: str = ""

    def apply_system(self, content: str) -> str:
        return _join_nonempty(self.system_prefix, content, self.system_suffix)

    def apply_user(self, content: str) -> str:
        return _join_nonempty(self.user_prefix, content, self.user_suffix)


def _join_nonempty(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())
