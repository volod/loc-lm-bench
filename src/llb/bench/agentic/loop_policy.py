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

REPEAT_FEEDBACK_CURRENT = "current"
REPEAT_FEEDBACK_UK = "uk"
REPEAT_FEEDBACK_BILINGUAL = "bilingual"
REPEAT_FEEDBACK_AYA_DIRECT = "aya_direct"
REPEAT_FEEDBACK_MISTRAL_USE = "mistral_use"
REPEAT_FEEDBACK_GEMMA_CHOICE = "gemma_choice"
REPEAT_FEEDBACK_GEMMA_PROGRESS = "gemma_progress"
REPEAT_FEEDBACK_GEMMA_AUTHORITY = "gemma_authority"
REPEAT_FEEDBACK_VARIANTS = (
    REPEAT_FEEDBACK_CURRENT,
    REPEAT_FEEDBACK_UK,
    REPEAT_FEEDBACK_BILINGUAL,
    REPEAT_FEEDBACK_AYA_DIRECT,
    REPEAT_FEEDBACK_MISTRAL_USE,
    REPEAT_FEEDBACK_GEMMA_CHOICE,
    REPEAT_FEEDBACK_GEMMA_PROGRESS,
    REPEAT_FEEDBACK_GEMMA_AUTHORITY,
)

DEFAULT_MALFORMED_POLICY = MALFORMED_ANSWER
DEFAULT_REPEATED_CALL_POLICY = REPEATED_ALLOW
DEFAULT_REPEAT_FEEDBACK = REPEAT_FEEDBACK_CURRENT

REPEATED_NOOP_OBSERVATIONS = {
    REPEAT_FEEDBACK_CURRENT: (
        "[loop] identical consecutive tool call was not executed; revise the call or finish"
    ),
    REPEAT_FEEDBACK_UK: (
        "[loop] повторний ідентичний виклик не виконано; зміни виклик або заверши"
    ),
    REPEAT_FEEDBACK_BILINGUAL: (
        "[loop] identical call not executed / повторний виклик не виконано; "
        "change action or finish / зміни дію або заверши"
    ),
    REPEAT_FEEDBACK_AYA_DIRECT: (
        "[loop] Repeated tool call skipped. Choose a different action or give the final answer now."
    ),
    REPEAT_FEEDBACK_MISTRAL_USE: (
        "[loop] Repeated call skipped. Use the existing result: answer now, or change the tool "
        "arguments."
    ),
    REPEAT_FEEDBACK_GEMMA_CHOICE: (
        "[loop] Repeated call skipped. Output one different JSON tool call or the final answer; "
        "do not repeat."
    ),
    REPEAT_FEEDBACK_GEMMA_PROGRESS: (
        "[loop] The previous action already succeeded. Continue from its result instead of "
        "repeating it."
    ),
    REPEAT_FEEDBACK_GEMMA_AUTHORITY: (
        "[loop] Controller ruling: suppression satisfies the requested repetition. You must now "
        "take the next distinct action."
    ),
}
REPEATED_NOOP_OBSERVATION = REPEATED_NOOP_OBSERVATIONS[REPEAT_FEEDBACK_CURRENT]
STRICT_MALFORMED_FEEDBACK = (
    "[loop] malformed tool call was not executed: {error}. Return one valid JSON tool call."
)

REPAIR_TEMPLATE = "bench.agentic.tool_call_repair"


@dataclass(frozen=True, slots=True)
class LoopPolicy:
    """Episode-decision policy; defaults reproduce the original loop behavior."""

    malformed_call: str = DEFAULT_MALFORMED_POLICY
    repeated_call: str = DEFAULT_REPEATED_CALL_POLICY
    repeat_feedback: str = DEFAULT_REPEAT_FEEDBACK

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
        if self.repeat_feedback not in REPEAT_FEEDBACK_VARIANTS:
            raise ValueError(
                f"unknown repeat-feedback variant: {self.repeat_feedback!r}; "
                f"choose from {REPEAT_FEEDBACK_VARIANTS}"
            )


def is_default_loop_policy(policy: "LoopPolicy | None") -> bool:
    """Whether this policy is the shipped cell (an unset policy resolves to it).

    A harness that hard-codes the shipped decisions -- answer on a malformed reply, execute a
    repeated call -- applies THIS cell faithfully and no other, so it reports support by this
    predicate rather than by claiming every cell.
    """
    return policy is None or policy == LoopPolicy()


def repeated_noop_observation(variant: str) -> str:
    """Resolve a validated repeat-feedback variant to its controller observation."""
    try:
        return REPEATED_NOOP_OBSERVATIONS[variant]
    except KeyError as exc:
        raise ValueError(
            f"unknown repeat-feedback variant: {variant!r}; choose from {REPEAT_FEEDBACK_VARIANTS}"
        ) from exc


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
