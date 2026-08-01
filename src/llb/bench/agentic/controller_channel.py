"""Backend-neutral controller-message roles for agent-loop authority studies."""

from dataclasses import dataclass
from typing import Literal

from llb.core.contracts.common import ChatMessage

CHANNEL_OBSERVATION = "observation"
CHANNEL_CONTROLLER = "controller"
ControllerChannel = Literal["observation", "controller"]
CONTROLLER_CHANNELS = (CHANNEL_OBSERVATION, CHANNEL_CONTROLLER)

# Both supported chat transports accept the same role labels. Keeping the map explicit makes a
# prospective study pin the wire representation instead of relying on an adapter default.
DEFAULT_ROLE_SERIALIZATION: dict[str, dict[str, str]] = {
    "ollama": {CHANNEL_OBSERVATION: "user", CHANNEL_CONTROLLER: "system"},
    "openai_compatible": {CHANNEL_OBSERVATION: "user", CHANNEL_CONTROLLER: "system"},
}


@dataclass(frozen=True, slots=True)
class ControllerFeedback:
    """One controller-generated notice and the logical channel that carries it."""

    content: str
    channel: ControllerChannel


def backend_serialization_name(backend: str) -> str:
    """Map configured backends onto the two predeclared chat-wire contracts."""
    return "ollama" if backend == "ollama" else "openai_compatible"


def serialize_controller_transcript(
    prompt: str,
    feedback: list[ControllerFeedback],
    *,
    backend: str,
    role_serialization: dict[str, dict[str, str]] | None = None,
) -> list[ChatMessage]:
    """Serialize a task prompt plus feedback; feedback text/order stay fixed across channels."""
    serialization = role_serialization or DEFAULT_ROLE_SERIALIZATION
    backend_name = backend_serialization_name(backend)
    try:
        roles = serialization[backend_name]
    except KeyError as exc:
        raise ValueError(
            f"no controller-message serialization for backend {backend_name!r}"
        ) from exc
    messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
    for item in feedback:
        try:
            role = roles[item.channel]
        except KeyError as exc:
            raise ValueError(
                f"no {item.channel!r} role serialization for backend {backend_name!r}"
            ) from exc
        messages.append({"role": role, "content": item.content})
    return messages


def transcript_chars(messages: list[ChatMessage]) -> int:
    """Match the existing prompt-budget approximation over every message content."""
    return sum(len(message["content"]) for message in messages)
