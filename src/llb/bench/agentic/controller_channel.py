"""Backend-neutral controller-message roles for agent-loop authority studies."""

from dataclasses import dataclass
from typing import Literal

from llb.core.contracts.common import ChatMessage

CHANNEL_OBSERVATION = "observation"
CHANNEL_CONTROLLER = "controller"
CHANNEL_PREAMBLE = "preamble"
ControllerChannel = Literal["observation", "controller", "preamble"]
CONTROLLER_CHANNELS = (CHANNEL_OBSERVATION, CHANNEL_CONTROLLER, CHANNEL_PREAMBLE)

# Both supported chat transports accept the same role labels. Keeping the map explicit makes a
# prospective study pin the wire representation instead of relying on an adapter default.
DEFAULT_ROLE_SERIALIZATION: dict[str, dict[str, str]] = {
    "ollama": {CHANNEL_OBSERVATION: "user", CHANNEL_CONTROLLER: "system"},
    "openai_compatible": {CHANNEL_OBSERVATION: "user", CHANNEL_CONTROLLER: "system"},
}

# A transform names message sources rather than only roles because a native preamble changes both
# the authority role and its position. An ``authority`` step expands every accumulated controller
# notice at that exact point; a ``prompt`` step emits the current task/transcript prompt once.
DEFAULT_PREAMBLE_SERIALIZATION: dict[str, dict[str, list[dict[str, str]]]] = {
    backend: {
        CHANNEL_OBSERVATION: [
            {"source": "prompt", "role": "user"},
            {"source": "authority", "role": "user"},
        ],
        CHANNEL_PREAMBLE: [
            {"source": "authority", "role": "system"},
            {"source": "prompt", "role": "user"},
        ],
    }
    for backend in ("ollama", "openai_compatible")
}


@dataclass(frozen=True, slots=True)
class ControllerFeedback:
    """One controller-generated notice and the logical channel that carries it."""

    content: str
    channel: ControllerChannel


def backend_serialization_name(backend: str) -> str:
    """Map configured backends onto the two predeclared chat-wire contracts."""
    return "ollama" if backend == "ollama" else "openai_compatible"


def _transform_for(
    serializer_transforms: dict[str, dict[str, list[dict[str, str]]]],
    feedback: list[ControllerFeedback],
    backend_name: str,
) -> list[dict[str, str]]:
    """The declared message plan for this feedback's placement, refusing a mixed placement.

    Mixed placements are refused rather than merged: the study this seam exists for compares ONE
    placement against another, and a transcript carrying both is neither arm.
    """
    channels = {item.channel for item in feedback}
    if len(channels) != 1:
        raise ValueError("controller feedback cannot mix serializer placements")
    channel = next(iter(channels))
    try:
        return serializer_transforms[backend_name][channel]
    except KeyError as exc:
        raise ValueError(
            f"no {channel!r} transcript transform for backend {backend_name!r}"
        ) from exc


def _transformed_messages(
    transform: list[dict[str, str]],
    prompt: str,
    feedback: list[ControllerFeedback],
    backend_name: str,
) -> list[ChatMessage]:
    """Render the declared plan: each step places either the prompt or the authority text."""
    channel = feedback[0].channel
    messages: list[ChatMessage] = []
    for step in transform:
        source = step.get("source")
        role = step.get("role")
        if source == "prompt" and role:
            messages.append({"role": role, "content": prompt})
        elif source == "authority" and role:
            messages.extend({"role": role, "content": item.content} for item in feedback)
        else:
            raise ValueError(
                f"invalid {channel!r} transcript transform for backend {backend_name!r}"
            )
    return messages


def _role_messages(
    roles: dict[str, str], prompt: str, feedback: list[ControllerFeedback], backend_name: str
) -> list[ChatMessage]:
    """The ordinary path: the prompt as a user turn, each feedback item in its channel's role."""
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


def serialize_controller_transcript(
    prompt: str,
    feedback: list[ControllerFeedback],
    *,
    backend: str,
    role_serialization: dict[str, dict[str, str]] | None = None,
    serializer_transforms: dict[str, dict[str, list[dict[str, str]]]] | None = None,
) -> list[ChatMessage]:
    """Serialize a task prompt plus feedback; feedback text/order stay fixed across channels."""
    backend_name = backend_serialization_name(backend)
    if serializer_transforms is not None and feedback:
        transform = _transform_for(serializer_transforms, feedback, backend_name)
        return _transformed_messages(transform, prompt, feedback, backend_name)
    serialization = role_serialization or DEFAULT_ROLE_SERIALIZATION
    try:
        roles = serialization[backend_name]
    except KeyError as exc:
        raise ValueError(
            f"no controller-message serialization for backend {backend_name!r}"
        ) from exc
    return _role_messages(roles, prompt, feedback, backend_name)


def transcript_chars(messages: list[ChatMessage]) -> int:
    """Match the existing prompt-budget approximation over every message content."""
    return sum(len(message["content"]) for message in messages)
