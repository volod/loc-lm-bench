from llb.backends.base import ERR_TIMEOUT, ChatResult
from llb.eval import multi_hop as mh
from llb.eval.common import OK, RETRIEVAL_MISS


class SeqStore:
    """Returns canned chunk lists in order, one per retrieve call."""

    def __init__(self, results):
        self._results = list(results)
        self.queries = []

    def retrieve(self, question, k):
        self.queries.append(question)
        return self._results.pop(0)


class SeqLauncher:
    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def chat(self, messages, max_tokens, temperature, timeout):
        self.calls.append(messages)
        return self._results.pop(0)


def _chunk(doc, start, end, text):
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": text}


def test_parse_controller_done():
    assert mh.parse_controller("ГОТОВО") == (mh.STOP, "")


def test_parse_controller_next():
    assert mh.parse_controller("ДАЛІ: хто був президентом") == (mh.CONTINUE, "хто був президентом")


def test_parse_controller_empty_is_stop():
    assert mh.parse_controller("") == (mh.STOP, "")
    assert mh.parse_controller("   ") == (mh.STOP, "")


def test_parse_controller_unparseable_is_stop():
    assert mh.parse_controller("незрозуміла відповідь без маркера") == (mh.STOP, "")


def test_parse_controller_next_without_query_is_stop():
    assert mh.parse_controller("ДАЛІ:") == (mh.STOP, "")


def test_retrieve_node_merges_and_dedupes_across_hops():
    c1, c2, c3 = _chunk("a", 0, 5, "alpha"), _chunk("a", 5, 10, "betaa"), _chunk("b", 0, 5, "gamma")
    store = SeqStore([[c1, c2], [c2, c3]])
    node = mh.make_retrieve_node(store, k=2)

    first = node({"question": "q"})
    assert len(first["gathered"]) == 2
    assert first["hop"] == 1 and first["n_hops"] == 1

    second = node({"question": "q", "subquery": "more", **first})
    assert [c["text"] for c in second["gathered"]] == ["alpha", "betaa", "gamma"]  # c2 deduped
    assert second["hop"] == 2
    assert store.queries == ["q", "more"]  # hop 0 uses the question; later hops the sub-query


def test_route_after_controller_continues_within_budget():
    state = {"decision": mh.CONTINUE, "hop": 1, "max_hops": 3}
    assert mh.route_after_controller(state) == "retrieve"


def test_route_after_controller_stops_when_budget_exhausted():
    state = {"decision": mh.CONTINUE, "hop": 3, "max_hops": 3}
    assert mh.route_after_controller(state) == "answer"


def test_route_after_controller_stops_on_stop_decision():
    state = {"decision": mh.STOP, "hop": 1, "max_hops": 3}
    assert mh.route_after_controller(state) == "answer"


def test_controller_node_parses_and_counts():
    launcher = SeqLauncher([ChatResult(text="ДАЛІ: наступне питання", completion_tokens=5)])
    node = mh.make_controller_node(launcher, max_tokens=32, temperature=0.0, timeout=10)
    update = node({"question": "q", "gathered": []})
    assert update["decision"] == mh.CONTINUE
    assert update["subquery"] == "наступне питання"
    assert update["n_model_calls"] == 1
    assert update["total_completion_tokens"] == 5


def test_controller_node_transport_error_stops():
    launcher = SeqLauncher([ChatResult(text="", error=ERR_TIMEOUT)])
    node = mh.make_controller_node(launcher, max_tokens=32, temperature=0.0, timeout=10)
    update = node({"question": "q", "gathered": []})
    assert update["decision"] == mh.STOP
    assert update["subquery"] == ""
    assert update["n_model_calls"] == 1


def test_answer_node_retrieval_miss_when_nothing_gathered():
    launcher = SeqLauncher([ChatResult(text="unused")])
    node = mh.make_answer_node(launcher, max_tokens=64, temperature=0.0, timeout=10)
    update = node({"question": "q", "gathered": []})
    assert update["status"] == RETRIEVAL_MISS
    assert update["answer"] == ""
    assert launcher.calls == []


def test_answer_node_ok_path():
    launcher = SeqLauncher([ChatResult(text="Відповідь", completion_tokens=4, latency_s=0.4)])
    node = mh.make_answer_node(launcher, max_tokens=64, temperature=0.0, timeout=10)
    update = node(
        {
            "question": "q",
            "gathered": [_chunk("a", 0, 5, "alpha")],
            "n_model_calls": 2,
            "total_completion_tokens": 6,
        }
    )
    assert update["status"] == OK
    assert update["answer"] == "Відповідь"
    assert update["n_model_calls"] == 3
    assert update["total_completion_tokens"] == 10
    assert update["usage"]["tokens_per_s"] > 0
