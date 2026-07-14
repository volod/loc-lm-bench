"""Focused tooling protocol implementation."""

import json
from collections.abc import Callable
from typing import Any
from llb.bench.common import (
    LLMComplete,
)
from llb.core.contracts import ToolDef
from llb.prompts.registry import render_text
from llb.scoring import tool_calls

TOOL_PROTOCOL_TEXT = "text"  # catalog-in-prompt JSON protocol (works on every backend)

TOOL_PROTOCOL_NATIVE = "native"  # native OpenAI tools= function-calling (tool-capable backends)

ToolCaller = Callable[[str, dict[str, ToolDef]], "tool_calls.ToolCall | None"]


def openai_tools(catalog: dict[str, ToolDef]) -> list[dict[str, Any]]:
    """Convert the project's tool catalog into the OpenAI `tools=` function-calling schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        }
        for tool in catalog.values()
    ]


def text_tool_caller(complete: LLMComplete) -> ToolCaller:
    """The universal text transport: embed the catalog in the prompt, parse the JSON call back."""

    def call(instruction: str, catalog: dict[str, ToolDef]) -> "tool_calls.ToolCall | None":
        return tool_calls.parse_tool_call(complete(text_tool_prompt(instruction, catalog)))

    return call


def native_tool_caller(
    client: Any,
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 120.0,
) -> ToolCaller:
    """The native OpenAI `tools=` transport over a tool-capable endpoint. `parse_tool_call` already
    normalizes the native `tool_calls` message, so the SAME scorer runs; `client` is injectable so a
    fake proves the wiring with no server. Transport errors -> no call attempted (None)."""
    import openai

    tool_specs_cache: list[dict[str, Any]] | None = None

    def call(instruction: str, catalog: dict[str, ToolDef]) -> "tool_calls.ToolCall | None":
        nonlocal tool_specs_cache
        if tool_specs_cache is None:
            tool_specs_cache = openai_tools(catalog)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": instruction}],
                tools=tool_specs_cache,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except openai.APIError:
            return None
        message = resp.choices[0].message if resp.choices else None
        return tool_calls.parse_tool_call(message)

    return call


def text_tool_prompt(instruction: str, catalog: dict[str, ToolDef]) -> str:
    """A backend-agnostic tool-calling prompt: the catalog as JSON + the user instruction, asking
    for a single JSON tool call (or a null call when no tool is needed)."""
    tools_json = json.dumps(list(catalog.values()), ensure_ascii=False, indent=2)
    return render_text(
        "bench.tooling.text_tool_prompt",
        {"instruction": instruction, "tools_json": tools_json},
    )
