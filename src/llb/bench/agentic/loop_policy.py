"""Measured policy knobs for the framework-free agent controller loop."""

import json
from dataclasses import dataclass
from typing import Any

from llb.core.contracts.benchmarks import ToolDef
from llb.prompts.registry import render_text

MALFORMED_ANSWER = "answer"
MALFORMED_REPAIR_ONCE = "repair_once"
MALFORMED_STRICT = "strict"
MALFORMED_POLICIES = (
    MALFORMED_ANSWER,
    MALFORMED_REPAIR_ONCE,
    MALFORMED_STRICT,
)

REPEATED_ALLOW = "allow"
REPEATED_NOOP = "noop"
REPEATED_CALL_POLICIES = (REPEATED_ALLOW, REPEATED_NOOP)

DEFAULT_MALFORMED_POLICY = MALFORMED_ANSWER
DEFAULT_REPEATED_CALL_POLICY = REPEATED_ALLOW

REPEATED_NOOP_OBSERVATION = (
    "[loop] identical consecutive tool call was not executed; revise the call or finish"
)
STRICT_MALFORMED_FEEDBACK = (
    "[loop] malformed tool call was not executed: {error}. Return one valid JSON tool call."
)

REPAIR_TEMPLATE = "bench.agentic.tool_call_repair"


@dataclass(frozen=True, slots=True)
class LoopPolicy:
    """Episode-decision policy; defaults reproduce the original loop behavior."""

    malformed_call: str = DEFAULT_MALFORMED_POLICY
    repeated_call: str = DEFAULT_REPEATED_CALL_POLICY

    def __post_init__(self) -> None:
        if self.malformed_call not in MALFORMED_POLICIES:
            raise ValueError(
                f"unknown malformed-call policy: {self.malformed_call!r}; "
                f"choose from {MALFORMED_POLICIES}"
            )
        if self.repeated_call not in REPEATED_CALL_POLICIES:
            raise ValueError(
                f"unknown repeated-call policy: {self.repeated_call!r}; "
                f"choose from {REPEATED_CALL_POLICIES}"
            )


def call_key(name: str, arguments: dict[str, Any]) -> str:
    """Stable identity for consecutive-call detection."""
    return f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"


def repair_prompt(
    original_prompt: str,
    catalog: dict[str, ToolDef],
    malformed_output: str,
    error: str,
) -> str:
    """One bounded repair request carrying the schema, rejected output, and parse error."""
    return render_text(
        REPAIR_TEMPLATE,
        {
            "original_prompt": original_prompt,
            "tools_json": json.dumps(list(catalog.values()), ensure_ascii=False, indent=2),
            "malformed_output": malformed_output,
            "parse_error": error,
        },
    )


def strict_feedback(error: str) -> str:
    return STRICT_MALFORMED_FEEDBACK.format(error=error)
