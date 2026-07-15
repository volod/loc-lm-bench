"""Primitive contracts shared across multiple domains."""

from typing import Any, TypeAlias

from typing_extensions import TypedDict

JsonObject: TypeAlias = dict[str, Any]


class ChatMessage(TypedDict):
    role: str
    content: str


class UsageRecord(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    tokens_per_s: float


class ValidationReport(TypedDict):
    n: int
    splits: dict[str, int]
    errors: list[str]
