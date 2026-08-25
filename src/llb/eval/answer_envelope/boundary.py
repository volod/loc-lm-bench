"""The ONE place a completion becomes a typed answer (typed-rag-answer-envelope).

Every answer-side signal downstream of this function reads a declared field, so the parse and the
validation happen once, here, and a completion that does not satisfy the contract ends in a typed
status instead of being scored as a wrong answer:

  - `malformed`   -- the completion is not a JSON object at all;
  - `schema_invalid` -- it IS JSON, but does not satisfy `AnswerEnvelope`.

The split matters because the two call for different fixes (a decoding/prompt problem versus a
field the model got wrong), and one number cannot say which happened.

`complete_envelope` adds the bounded repair policy around that parse: exactly ONE retry, carrying
the validation error back to the model, in the same shape the agent loop-policy lane measures for
tool calls. It is bounded on purpose -- an unbounded repair loop converts a formatting failure into
an unmeasured token cost, and the whole point of the two statuses is to MEASURE that failure.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from llb.backends.base import ChatResult
from llb.core.contracts.common import ChatMessage
from llb.eval import common as eval_common
from llb.eval.answer_envelope.models import AnswerEnvelope, envelope_prompt_values
from llb.prompts.registry import render_text
from llb.scoring.structured.schema import parse_output

# The repair reprompt asks for the same contract again, quoting the validator's own complaint.
REPAIR_TEMPLATE = "eval.rag.envelope_repair"

# How many validator complaints the repair prompt carries. All of them would let one badly-shaped
# completion fill the reprompt with error text; the first few name the fields that actually broke.
MAX_REPORTED_ERRORS = 5

# One chat turn: the callable the graph's generate node already holds, narrowed to what the
# repair policy needs.
ChatFn = Callable[[list[ChatMessage]], ChatResult]


@dataclass(frozen=True, slots=True)
class EnvelopeParse:
    """The outcome of validating one completion against the answer contract."""

    envelope: AnswerEnvelope | None
    status: str
    error: str

    @property
    def ok(self) -> bool:
        return self.envelope is not None


def format_validation_errors(exc: ValidationError) -> str:
    """The validator's complaint as compact `field: message` lines, for the repair prompt."""
    lines = []
    for entry in exc.errors()[:MAX_REPORTED_ERRORS]:
        location = ".".join(str(part) for part in entry.get("loc", ())) or "<root>"
        lines.append(f"{location}: {entry.get('msg', 'invalid')}")
    return "; ".join(lines)


def parse_envelope(text: str) -> EnvelopeParse:
    """Parse and validate one completion into an `AnswerEnvelope`, or a typed failure.

    The JSON extraction is the structured-output lane's own `parse_output`, so a fenced or
    prose-wrapped object is recovered here exactly as it is in the BENCHMARK lane.
    """
    data: dict[str, Any] | None = parse_output(text)
    if data is None:
        return EnvelopeParse(None, eval_common.MALFORMED, "response is not a JSON object")
    try:
        envelope = AnswerEnvelope.model_validate(data)
    except ValidationError as exc:
        return EnvelopeParse(None, eval_common.SCHEMA_INVALID, format_validation_errors(exc))
    return EnvelopeParse(envelope, eval_common.OK, "")


def repair_prompt(error: str) -> ChatMessage:
    """The single bounded reprompt: the validator's complaint plus the contract, once more.

    The rejected completion itself is not quoted here -- it is already the assistant turn the
    reprompt follows, so quoting it again would only spend context on text the model just wrote.
    """
    content = render_text(REPAIR_TEMPLATE, {"error": error, **envelope_prompt_values()})
    return {"role": "user", "content": content}


@dataclass(frozen=True, slots=True)
class EnvelopeCompletion:
    """One envelope generation: its final parse, both chat results, and whether it was repaired.

    `repaired` means a repair reprompt was ISSUED -- the FIRST completion did not validate --
    whatever the retry then produced. That makes first-attempt conformance readable off the
    recorded columns as `1 - repair_rate`, and the repair's contribution readable as the gap
    between first-attempt and final conformance.
    """

    parse: EnvelopeParse
    results: tuple[ChatResult, ...]
    repaired: bool


def complete_envelope(
    chat: ChatFn, messages: list[ChatMessage], result: ChatResult
) -> EnvelopeCompletion:
    """Validate `result`, and on a failure spend exactly one repair reprompt.

    A transport failure is never repaired: the run's retry policy owns that, and reprompting a
    timed-out backend would spend the repair budget on a case that never answered.
    """
    parse = parse_envelope(result.text or "")
    if parse.ok or result.error:
        return EnvelopeCompletion(parse, (result,), repaired=False)
    rejected: ChatMessage = {"role": "assistant", "content": result.text or ""}
    retry_messages: list[ChatMessage] = [
        *messages,
        rejected,
        repair_prompt(parse.error),
    ]
    retry = chat(retry_messages)
    if retry.error:  # the repair itself failed in transport; keep the first, typed verdict
        return EnvelopeCompletion(parse, (result, retry), repaired=True)
    return EnvelopeCompletion(parse_envelope(retry.text or ""), (result, retry), repaired=True)
