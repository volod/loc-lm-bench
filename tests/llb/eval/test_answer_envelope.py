"""typed-rag-answer-envelope -- the contract, the boundary, the repair, and the graph seam.

Everything here is fixture-driven: the boundary is a pure function over completion text, and the
generate node runs against a scripted fake launcher, so parse / validate / repair and every
terminal status are covered with no backend and no GPU.
"""

import json

import pytest

from llb.backends.base import ERR_BACKEND, ERR_TIMEOUT, ChatResult
from llb.eval import common
from llb.eval import graph
from llb.eval.answer_envelope import boundary, lane
from llb.eval.answer_envelope.models import (
    AnswerEnvelope,
    EnvelopeClaim,
    envelope_schema_block,
)
from llb.executor.durability_journal import _JOURNALED_STATE_KEYS

VALID = {
    "answer": "Конвенцію підписано 1994 року.",
    "abstained": False,
    "claims": [{"text": "Конвенцію підписано 1994 року", "citations": [1]}],
}


def _completion(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


# --- the contract -----------------------------------------------------------------------------


def test_schema_block_is_the_contract_itself_not_a_copy_of_it():
    # The prompt's worked example is a MODEL INSTANCE, so a field added to the contract cannot
    # silently fail to reach the model (or vice versa).
    block = json.loads(envelope_schema_block())
    assert set(block) == set(AnswerEnvelope.model_fields)
    assert set(block["claims"][0]) == set(EnvelopeClaim.model_fields)


def test_entity_types_normalize_into_the_closed_vocabulary():
    payload = dict(VALID)
    payload["claims"] = [
        {
            "text": "Конвенцію підписано 1994 року",
            "citations": [1],
            "triple": {
                "subject": "Конвенція",
                "relation": "підписана",
                "object": "Женева",
                "subject_type": "treaty",  # synonym -> LAW
                "object_type": "не-існує",  # out of vocabulary -> MISC
            },
        }
    ]
    triple = boundary.parse_envelope(_completion(payload)).envelope.claims[0].triple
    assert (triple.subject_type, triple.object_type) == ("LAW", "MISC")


def test_extra_keys_are_ignored_rather_than_rejected():
    parsed = boundary.parse_envelope(_completion({**VALID, "confidence": 0.9}))
    assert parsed.status == common.OK and parsed.envelope.answer == VALID["answer"]


# --- the boundary -----------------------------------------------------------------------------


def test_fenced_json_is_recovered_exactly_as_the_structured_lane_recovers_it():
    parsed = boundary.parse_envelope(f"Ось відповідь:\n```json\n{_completion(VALID)}\n```")
    assert parsed.status == common.OK


def test_prose_is_malformed_and_wrong_shaped_json_is_schema_invalid():
    # The split is the point: "did not emit JSON" and "emitted JSON of the wrong shape" are
    # different failures with different fixes, and today's single number cannot say which.
    assert boundary.parse_envelope("Конвенцію підписано 1994.").status == common.MALFORMED
    invalid = boundary.parse_envelope('{"answer": "Так"}')
    assert invalid.status == common.SCHEMA_INVALID
    assert "abstained" in invalid.error and "claims" in invalid.error


def test_abstention_is_a_declared_field_not_a_refusal_stem():
    parsed = boundary.parse_envelope(_completion({"answer": "", "abstained": True, "claims": []}))
    assert parsed.envelope.abstained is True
    # ...and the same declaration carries no first-person refusal text at all.
    assert common.is_abstention(parsed.envelope.answer) is False


# --- the bounded repair -----------------------------------------------------------------------


def _scripted(*results):
    queue = list(results)
    calls: list[list[dict]] = []

    def chat(messages):
        calls.append(messages)
        return queue.pop(0)

    return chat, calls


def test_repair_is_spent_once_and_carries_the_validation_error():
    chat, calls = _scripted(ChatResult(text=_completion(VALID), completion_tokens=30))
    first = ChatResult(text='{"answer": "Так"}', completion_tokens=5)
    done = boundary.complete_envelope(chat, [{"role": "user", "content": "q"}], first)
    assert done.repaired is True and done.parse.status == common.OK
    assert len(calls) == 1  # exactly ONE reprompt, never a loop
    assert "abstained" in calls[0][-1]["content"]  # the validator's own complaint reaches the model
    assert calls[0][-2]["role"] == "assistant"  # the rejected completion stays in the turn


def test_a_conformant_first_completion_spends_no_repair():
    chat, calls = _scripted()
    done = boundary.complete_envelope(
        chat, [], ChatResult(text=_completion(VALID), completion_tokens=30)
    )
    assert done.repaired is False and calls == []


def test_a_repair_that_also_fails_ends_in_the_typed_status():
    chat, _ = _scripted(ChatResult(text="все ще проза", completion_tokens=4))
    done = boundary.complete_envelope(chat, [], ChatResult(text='{"answer": "Так"}'))
    assert done.repaired is True and done.parse.status == common.MALFORMED


def test_a_transport_failure_is_never_repaired():
    chat, calls = _scripted()
    done = boundary.complete_envelope(chat, [], ChatResult(text="", error=ERR_TIMEOUT))
    assert done.repaired is False and calls == []


def test_a_repaired_case_is_charged_for_both_generations():
    usage = lane.merged_usage(
        (
            ChatResult(text="x", prompt_tokens=100, completion_tokens=5, latency_s=1.0),
            ChatResult(text="y", prompt_tokens=140, completion_tokens=30, latency_s=3.0),
        )
    )
    assert usage["prompt_tokens"] == 240
    assert usage["completion_tokens"] == 35
    assert usage["latency_s"] == 4.0


# --- the graph seam ---------------------------------------------------------------------------


class FakeLauncher:
    def __init__(self, *results):
        self._queue = list(results)
        self.messages: list[list[dict]] = []

    def chat(self, messages, max_tokens, temperature, timeout):
        self.messages.append(messages)
        return self._queue.pop(0)


def _generate(launcher, answer_format=lane.ENVELOPE):
    node = graph.make_generate_node(launcher, 96, 0.0, 30.0, answer_format=answer_format)
    return node({"question": "Коли підписано конвенцію?", "context": "[1] текст"})


TERMINAL_CASES = [
    (ChatResult(text=_completion(VALID)), common.OK),
    (ChatResult(text=_completion({"answer": "", "abstained": True, "claims": []})), common.EMPTY),
    (ChatResult(text="проза без JSON"), common.MALFORMED),
    (ChatResult(text='{"answer": "Так"}'), common.SCHEMA_INVALID),
    (ChatResult(text="", error=ERR_TIMEOUT), ERR_TIMEOUT),
    (ChatResult(text="", error=ERR_BACKEND), ERR_BACKEND),
]


@pytest.mark.parametrize("result,expected", TERMINAL_CASES)
def test_every_envelope_case_ends_in_exactly_one_terminal_status(result, expected):
    # A non-conformant completion ends in a TYPED status rather than being scored as a wrong
    # answer, so the two failure modes never collapse into one low correctness number.
    launcher = FakeLauncher(result, result)  # second result covers the repair attempt
    assert _generate(launcher)["status"] == expected


def test_a_declared_refusal_still_classifies_as_a_refusal():
    refusal = _completion(
        {"answer": "Вибачте, але я не можу відповісти.", "abstained": True, "claims": []}
    )
    assert _generate(FakeLauncher(ChatResult(text=refusal)))["status"] == common.REFUSAL


def test_the_envelope_prompt_is_used_only_for_the_envelope_format():
    assert graph.generation_template(answer_format=lane.ENVELOPE) == graph.ENVELOPE_TEMPLATE
    # The declared contract supersedes the [i]-in-prose instruction rather than stacking with it.
    assert (
        graph.generation_template(cited=True, answer_format=lane.ENVELOPE)
        == graph.ENVELOPE_TEMPLATE
    )
    assert graph.generation_template(cited=True) == graph.CITED_ANSWER_TEMPLATE
    assert graph.generation_template() == graph.CHAT_TEMPLATE


def test_with_the_envelope_off_the_generate_node_records_exactly_what_it_always_did():
    launcher = FakeLauncher(ChatResult(text="Київ", completion_tokens=2, latency_s=0.5))
    update = _generate(launcher, answer_format=lane.FREE_TEXT)
    assert set(update) == {"answer", "status", "error", "usage"}
    assert update["answer"] == "Київ" and update["status"] == common.OK
    assert "Поверни ЛИШЕ один JSON" not in launcher.messages[0][0]["content"]


def test_every_envelope_state_key_survives_the_durability_journal():
    # A state key the journal drops is a score column a RESUMED case silently loses.
    update = _generate(FakeLauncher(ChatResult(text=_completion(VALID))))
    carried = set(_JOURNALED_STATE_KEYS)
    assert {key for key in update if key.startswith("envelope")} <= carried
