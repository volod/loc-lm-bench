from llb.backends.base import ERR_TIMEOUT, ChatResult
from llb.eval import map_reduce as mr
from llb.eval.common import EMPTY, OK


class SeqLauncher:
    """Returns canned ChatResults in order; records the messages it was called with."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def chat(self, messages, max_tokens, temperature, timeout):
        self.calls.append(messages)
        return self._results.pop(0)


def test_split_document_overlapping_windows():
    segs = mr.split_document("abcdefghij", max_chars=4, overlap=1)
    assert segs == ["abcd", "defg", "ghij", "j"]


def test_split_document_empty():
    assert mr.split_document("   ", max_chars=10, overlap=0) == []


def test_split_node_flags_empty_document():
    node = mr.make_split_node(max_chars=100, overlap=10)
    update = node({"document": ""})
    assert update["status"] == EMPTY
    assert update["segments"] == []


def test_split_node_counts_segments():
    node = mr.make_split_node(max_chars=4, overlap=0)
    update = node({"document": "abcdef"})
    assert update["n_segments"] == 2
    assert "status" not in update


def test_is_no_info():
    assert mr.is_no_info(mr.NO_INFO_MARKER)
    assert mr.is_no_info("   ")
    assert not mr.is_no_info("Київ")


def test_map_node_collects_nonempty_partials():
    launcher = SeqLauncher(
        [
            ChatResult(text="Київ - столиця", completion_tokens=4),
            ChatResult(text=mr.NO_INFO_MARKER, completion_tokens=2),
        ]
    )
    node = mr.make_map_node(launcher, max_tokens=64, temperature=0.0, timeout=10)
    update = node({"question": "q", "segments": ["s1", "s2"]})
    assert update["partials"] == ["Київ - столиця"]
    assert update["n_model_calls"] == 2
    assert update["total_completion_tokens"] == 6


def test_map_node_short_circuits_on_empty():
    launcher = SeqLauncher([ChatResult(text="unused")])
    node = mr.make_map_node(launcher, max_tokens=64, temperature=0.0, timeout=10)
    update = node({"question": "q", "status": EMPTY})
    assert update["partials"] == []
    assert launcher.calls == []


def test_map_node_propagates_transport_error_when_all_fail():
    launcher = SeqLauncher([ChatResult(text="", error=ERR_TIMEOUT)])
    node = mr.make_map_node(launcher, max_tokens=64, temperature=0.0, timeout=10)
    update = node({"question": "q", "segments": ["s1"]})
    assert update["partials"] == []
    assert update["status"] == ERR_TIMEOUT
    assert update["error"] == ERR_TIMEOUT


def test_reduce_node_ok_path_accumulates_usage():
    launcher = SeqLauncher([ChatResult(text="Зведено", completion_tokens=3, latency_s=0.5)])
    node = mr.make_reduce_node(launcher, max_tokens=64, temperature=0.0, timeout=10)
    update = node(
        {"question": "q", "partials": ["A", "B"], "n_model_calls": 2, "total_completion_tokens": 4}
    )
    assert update["status"] == OK
    assert update["answer"] == "Зведено"
    assert update["n_model_calls"] == 3
    assert update["total_completion_tokens"] == 7
    assert update["usage"]["completion_tokens"] == 7
    assert update["usage"]["tokens_per_s"] > 0


def test_reduce_node_empty_when_no_partials():
    launcher = SeqLauncher([ChatResult(text="unused")])
    node = mr.make_reduce_node(launcher, max_tokens=64, temperature=0.0, timeout=10)
    update = node({"question": "q", "partials": []})
    assert update["status"] == EMPTY
    assert update["answer"] == ""
    assert launcher.calls == []


def test_reduce_node_short_circuits_on_terminal_status():
    launcher = SeqLauncher([ChatResult(text="unused")])
    node = mr.make_reduce_node(launcher, max_tokens=64, temperature=0.0, timeout=10)
    update = node({"question": "q", "partials": ["A"], "status": ERR_TIMEOUT})
    assert update["answer"] == ""
    assert launcher.calls == []


# --- text-prompt driver (the M5 `complete` substrate; M5.4) --------------------------------


def test_run_map_reduce_text_splits_maps_reduces():
    doc = "Розділ один. " * 50 + "Ключовий факт: бюджет зріс."
    calls = []

    def complete(prompt):
        calls.append(prompt)
        if "Зведена відповідь" in prompt:  # the reduce prompt
            return "Бюджет зріс."
        return "Бюджет зріс."  # each map partial

    answer = mr.run_map_reduce_text(
        complete, "Що сталося з бюджетом?", doc, max_chars=200, overlap=20
    )
    assert answer == "Бюджет зріс."
    assert len(calls) > 2  # multiple map calls + one reduce


def test_run_map_reduce_text_single_segment_no_reduce():
    answer = mr.run_map_reduce_text(lambda _p: "коротка відповідь", "питання?", "малий документ")
    assert answer == "коротка відповідь"


def test_run_map_reduce_text_all_no_info_returns_empty():
    answer = mr.run_map_reduce_text(
        lambda _p: mr.NO_INFO_MARKER, "q", "doc " * 100, max_chars=100, overlap=0
    )
    assert answer == ""
