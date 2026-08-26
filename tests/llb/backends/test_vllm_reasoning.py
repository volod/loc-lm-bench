"""vLLM reasoning-control body + the verified-once probe, driven by fakes (no vLLM/server)."""

import json

from llb.backends import vllm_reasoning as vr


def test_body_is_empty_when_no_thinking_control_is_asked_for():
    """`think=None` is what keeps a shipped vLLM request byte-identical: no extras at all."""
    assert vr.reasoning_extra_body(None) == {}
    assert vr.reasoning_extra_body(False, fields=vr.FIELDS_NONE) == {}


def test_full_body_carries_the_template_kwarg_and_vllms_own_request_fields():
    body = vr.reasoning_extra_body(False)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["include_reasoning"] is False
    assert body["reasoning_effort"] == vr.REASONING_EFFORT_NONE


def test_template_only_body_drops_the_fields_an_older_server_may_reject():
    assert vr.reasoning_extra_body(False, fields=vr.FIELDS_TEMPLATE_ONLY) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_enabling_thinking_never_sends_the_suppression_fields():
    """`enable_thinking=true` asks for reasoning; `reasoning_effort=none` would contradict it."""
    assert vr.reasoning_extra_body(True) == {"chat_template_kwargs": {"enable_thinking": True}}


def _recording_send(reject: tuple[str, ...] = ()):
    """A probe send that fails whenever the body carries one of the `reject` keys."""
    seen: list[dict] = []

    def send(extra_body):
        seen.append(extra_body)
        if any(key in extra_body for key in reject):
            return "backend_error"
        return None

    return send, seen


def test_probe_stops_at_the_first_body_the_server_accepts():
    send, seen = _recording_send()
    verdict = vr.probe_thinking_fields(send, vllm_version="0.23.0")
    assert verdict is not None and verdict["fields"] == vr.FIELDS_FULL
    assert verdict["vllm_version"] == "0.23.0" and len(seen) == 1


def test_probe_falls_back_to_the_template_kwarg_when_the_request_fields_are_rejected():
    send, seen = _recording_send(reject=("include_reasoning",))
    verdict = vr.probe_thinking_fields(send, vllm_version="0.9.0")
    assert verdict is not None and verdict["fields"] == vr.FIELDS_TEMPLATE_ONLY
    assert len(seen) == 2


def test_probe_records_none_when_every_body_is_rejected_but_a_plain_request_works():
    send, seen = _recording_send(reject=("include_reasoning", "chat_template_kwargs"))
    verdict = vr.probe_thinking_fields(send, vllm_version="0.1.0")
    assert verdict is not None and verdict["fields"] == vr.FIELDS_NONE
    assert seen[-1] == {}  # the control request that proves the SERVER is fine


def test_probe_is_inconclusive_when_the_control_request_fails_too():
    """A dead or timing-out server must not pin `unsupported` for the whole vLLM version."""
    assert vr.probe_thinking_fields(lambda body: "timeout", vllm_version="0.23.0") is None


def test_verdict_round_trips_and_is_invalidated_by_a_vllm_upgrade(tmp_path):
    verdict = vr.probe_thinking_fields(_recording_send()[0], vllm_version="0.23.0")
    assert verdict is not None
    vr.save_verdict(verdict, tmp_path)
    loaded = vr.load_verdict(tmp_path)
    assert loaded == verdict
    assert vr.verdict_is_current(loaded, "0.23.0")
    assert not vr.verdict_is_current(loaded, "0.24.0")
    assert not vr.verdict_is_current(None, "0.23.0")


def test_load_verdict_ignores_a_missing_or_malformed_file(tmp_path):
    assert vr.load_verdict(tmp_path) is None
    path = vr.verdict_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert vr.load_verdict(tmp_path) is None
    path.write_text(json.dumps({"fields": "bogus"}), encoding="utf-8")
    assert vr.load_verdict(tmp_path) is None


def test_resolve_probes_once_then_reuses_the_cached_verdict(tmp_path):
    send, seen = _recording_send()
    first = vr.resolve_thinking_fields(send, data_dir=tmp_path, installed="0.23.0")
    second = vr.resolve_thinking_fields(send, data_dir=tmp_path, installed="0.23.0")
    assert first["fields"] == second["fields"] == vr.FIELDS_FULL
    assert len(seen) == 1  # the second launch never touched the server


def test_resolve_reprobes_after_a_vllm_upgrade(tmp_path):
    send, seen = _recording_send()
    vr.resolve_thinking_fields(send, data_dir=tmp_path, installed="0.23.0")
    vr.resolve_thinking_fields(send, data_dir=tmp_path, installed="0.24.0")
    assert len(seen) == 2
    assert vr.load_verdict(tmp_path)["vllm_version"] == "0.24.0"


def test_resolve_sends_nothing_and_persists_nothing_when_the_probe_is_inconclusive(tmp_path):
    verdict = vr.resolve_thinking_fields(
        lambda body: "timeout", data_dir=tmp_path, installed="0.23.0"
    )
    assert verdict["fields"] == vr.FIELDS_NONE
    assert vr.load_verdict(tmp_path) is None  # nothing cached, so the next launch retries
