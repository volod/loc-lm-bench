"""The generate-node half of the envelope lane: one completion -> one typed state update.

The free-text path records `answer` + `status` and lets the scorers infer the rest. This lane
records the same two fields -- `answer` is the envelope's own `answer` string, so the objective
scores exactly as it would have from free text -- plus the declaration itself, so the answer-side
scorers can read fields instead of re-deriving them.

`envelope` is stored as a plain dict rather than the model instance: the durability journal
serializes the state to JSON, and a state key the journal cannot carry is a score column a RESUMED
case would silently lose.
"""

from llb.backends.base import ChatResult
from llb.core.contracts.common import UsageRecord
from llb.eval import common as eval_common
from llb.eval.answer_envelope.boundary import EnvelopeCompletion
from llb.eval.graph_contracts import RagState

# Answer format of the generation lane: free prose, or the declared answer contract.
FREE_TEXT = "free_text"
ENVELOPE = "envelope"
ANSWER_FORMATS = (FREE_TEXT, ENVELOPE)

# The statuses a violation may override. A refusal or an empty answer asserted nothing, and a
# transport failure produced nothing -- turning either into `ontology_violation` would credit the
# gate with a rejection it did not make.
_GATEABLE_STATUSES = (eval_common.OK,)


def merged_usage(results: tuple[ChatResult, ...]) -> UsageRecord:
    """Token + latency accounting across every call one case spent, the repair reprompt included.

    A repaired case really did cost two generations; charging it for one would make the envelope
    lane look cheaper than it is, which is precisely the tradeoff the study has to weigh. A case
    whose last call failed in transport reports no rate at all, exactly as `ChatResult` does, so a
    failed case can never be credited with throughput.
    """
    completion_tokens = sum(result.completion_tokens for result in results)
    latency_s = sum(result.latency_s for result in results)
    measurable = latency_s > 0 and completion_tokens > 0 and not results[-1].error
    return {
        "prompt_tokens": sum(result.prompt_tokens for result in results),
        "completion_tokens": completion_tokens,
        "latency_s": latency_s,
        "tokens_per_s": (completion_tokens / latency_s) if measurable else 0.0,
    }


def envelope_state(completion: EnvelopeCompletion) -> RagState:
    """The terminal state update for one envelope case.

    A valid envelope is classified on its DECLARED answer text, so a refusal or an empty answer
    reaches the same status it would have from free text; an envelope that never validated keeps
    its typed parse status (`malformed` / `schema_invalid`) and contributes an empty answer.

    Step two of the gate (ontology-validated-answer-gate) is applied LAST and only to an envelope
    that already parsed: an answer whose declared triples break an accepted axiom ends in
    `ontology_violation` rather than being scored as a fluent answer. The answer text is still
    recorded -- the study reads what the gate refused, and a rejection nobody can inspect is not
    evidence.
    """
    last = completion.results[-1]
    parse = completion.parse
    update: RagState = {
        "answer": parse.envelope.answer if parse.envelope is not None else "",
        "usage": merged_usage(completion.results),
        "error": last.error,
        "envelope_status": parse.status,
        "envelope_repaired": completion.repaired,
    }
    if parse.envelope is None:
        update["status"] = last.error or parse.status
        update["envelope_error"] = parse.error
        return update
    update["status"] = eval_common.classify_response(parse.envelope.answer, last.error)
    update["envelope"] = parse.envelope.model_dump()
    return {**update, **_validation_state(completion, update["status"])}


def _validation_state(completion: EnvelopeCompletion, status: str) -> RagState:
    """The step-two columns, and the status override a violation earns.

    A case that already failed in transport keeps its transport status: the gate has an opinion
    about the ANSWER, and there is no answer to have an opinion about.
    """
    verdict = completion.validation
    if verdict is None:
        return {}
    update: RagState = {
        "validation_checked_triples": verdict.checked_triples,
        "validation_repaired": completion.validation_repaired,
        "validation_violations": len(verdict.violations),
        "validation_classes": verdict.classes,
        "validation_axioms": verdict.axiom_ids,
    }
    if verdict.violations and status in _GATEABLE_STATUSES:
        update["status"] = eval_common.ONTOLOGY_VIOLATION
        update["validation_error"] = verdict.detail()
    return update
