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

The SEMANTIC step of the two-step gate (ontology-validated-answer-gate) extends this same boundary
rather than adding a second one: an optional `validate` callable reads the accepted envelope and
returns the axioms its declared triples broke. Its repair is bounded exactly like the schema one --
at most one reprompt, and the retry is accepted ONLY if it both parses and passes, so the semantic
repair can rescue an answer and can never damage one. The two budgets are separate because the two
columns must stay separate: `repaired` is what makes first-attempt CONFORMANCE readable as
`1 - repair_rate`, and folding a semantic reprompt into it would silently redefine that number.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from pydantic import ValidationError

from llb.backends.base import ChatResult
from llb.core.contracts.common import ChatMessage
from llb.eval import common as eval_common
from llb.eval.answer_envelope.models import AnswerEnvelope, envelope_prompt_values
from llb.eval.answer_validation.models import GateVerdict
from llb.prompts.registry import render_text
from llb.scoring.structured.schema import parse_output

# The repair reprompt asks for the same contract again, quoting the validator's own complaint.
REPAIR_TEMPLATE = "eval.rag.envelope_repair"
# The semantic repair names the broken constraints instead of a field error, and offers the
# declared abstention as the honest way out of a claim the context does not support.
ONTOLOGY_REPAIR_TEMPLATE = "eval.rag.envelope_ontology_repair"

# How many validator complaints the repair prompt carries. All of them would let one badly-shaped
# completion fill the reprompt with error text; the first few name the fields that actually broke.
MAX_REPORTED_ERRORS = 5

# One chat turn: the callable the graph's generate node already holds, narrowed to what the
# repair policy needs.
ChatFn = Callable[[list[ChatMessage]], ChatResult]
# Step two of the gate, bound to this case's retrieved chunks by the caller.
Validator = Callable[[AnswerEnvelope], GateVerdict]


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


def ontology_repair_prompt(violations: str) -> ChatMessage:
    """The semantic reprompt: which accepted constraints the declared triples broke.

    It names the constraints rather than dictating a replacement answer -- rewriting the answer on
    the model's behalf is outside this gate -- and points at the declared abstention as the honest
    outcome when the context does not carry the claim.
    """
    content = render_text(
        ONTOLOGY_REPAIR_TEMPLATE, {"violations": violations, **envelope_prompt_values()}
    )
    return {"role": "user", "content": content}


@dataclass(frozen=True, slots=True)
class EnvelopeCompletion:
    """One envelope generation: its final parse, every chat result, and what was repaired.

    `repaired` means a SCHEMA repair reprompt was ISSUED -- the FIRST completion did not validate
    -- whatever the retry then produced. That makes first-attempt conformance readable off the
    recorded columns as `1 - repair_rate`, and the repair's contribution readable as the gap
    between first-attempt and final conformance.

    `validation` is step two's verdict (None when the gate did not run), and `validation_repaired`
    says its own bounded reprompt was spent. They are separate fields for the same reason they are
    separate columns: a formatting failure and a semantic one call for different fixes.
    """

    parse: EnvelopeParse
    results: tuple[ChatResult, ...]
    repaired: bool
    validation: GateVerdict | None = None
    validation_repaired: bool = False


def complete_envelope(
    chat: ChatFn,
    messages: list[ChatMessage],
    result: ChatResult,
    validate: Validator | None = None,
) -> EnvelopeCompletion:
    """Validate `result`, spend at most one repair reprompt, then run step two on what survived.

    A transport failure is never repaired: the run's retry policy owns that, and reprompting a
    timed-out backend would spend the repair budget on a case that never answered.
    """
    completion = _schema_completion(chat, messages, result)
    if validate is None or completion.parse.envelope is None:
        return completion
    return _validated(chat, messages, completion, validate)


def _schema_completion(
    chat: ChatFn, messages: list[ChatMessage], result: ChatResult
) -> EnvelopeCompletion:
    """Step one alone: the parse plus its single bounded formatting repair."""
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


def _validated(
    chat: ChatFn,
    messages: list[ChatMessage],
    completion: EnvelopeCompletion,
    validate: Validator,
) -> EnvelopeCompletion:
    """Step two: check the accepted envelope, and on a violation spend one semantic reprompt.

    The retry REPLACES the answer only when it both parses and passes -- an unparseable or still
    violating retry leaves the first envelope and its verdict standing, so the case ends in
    `ontology_violation` on the answer the model actually gave. Its tokens are still charged: the
    round trip cost what it cost whether or not it rescued anything.
    """
    assert completion.parse.envelope is not None  # guarded by the caller
    verdict = validate(completion.parse.envelope)
    if verdict.ok:
        return replace(completion, validation=verdict)
    last = completion.results[-1]
    retry_messages: list[ChatMessage] = [
        *messages,
        {"role": "assistant", "content": last.text or ""},
        ontology_repair_prompt(verdict.detail()),
    ]
    retry = chat(retry_messages)
    results = (*completion.results, retry)
    rescued = _rescued(retry, validate)
    if rescued is None:  # transport failure, unparseable retry, or still violating
        return replace(completion, results=results, validation=verdict, validation_repaired=True)
    repaired_parse, repaired_verdict = rescued
    return EnvelopeCompletion(
        repaired_parse,
        results,
        repaired=completion.repaired,
        validation=repaired_verdict,
        validation_repaired=True,
    )


def _rescued(retry: ChatResult, validate: Validator) -> tuple[EnvelopeParse, "GateVerdict"] | None:
    """The retry's parse and verdict when it BOTH parses and passes, else None."""
    if retry.error:
        return None
    parse = parse_envelope(retry.text or "")
    if parse.envelope is None:
        return None
    verdict = validate(parse.envelope)
    return None if not verdict.ok else (parse, verdict)
