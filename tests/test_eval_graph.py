from llb.backends.base import ERR_BACKEND, ERR_TIMEOUT, ChatResult
from llb.eval import common
from llb.eval import graph


class FakeStore:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, question, k):
        return self._chunks[:k]


class FakeLauncher:
    def __init__(self, result):
        self._result = result

    def chat(self, messages, max_tokens, temperature, timeout):
        self.last_messages = messages
        return self._result


def test_classify_ok():
    assert common.classify_response("Київ - столиця", None) == common.OK


def test_classify_empty():
    assert common.classify_response("   ", None) == common.EMPTY


def test_classify_refusal():
    assert common.classify_response("Вибачте, але я не можу відповісти.", None) == common.REFUSAL


def test_classify_passes_through_transport_errors():
    assert common.classify_response("", ERR_TIMEOUT) == ERR_TIMEOUT
    assert common.classify_response("anything", ERR_BACKEND) == ERR_BACKEND


def test_classify_malformed_json():
    assert common.classify_response("{not json", None, expect_json=True) == common.MALFORMED
    assert common.classify_response('{"a": 1}', None, expect_json=True) == common.OK


def test_retrieve_node_flags_miss_on_empty():
    node = graph.make_retrieve_node(FakeStore([]), k=5)
    update = node({"question": "q"})
    assert update["status"] == common.RETRIEVAL_MISS
    assert update["retrieved"] == []


def test_retrieve_node_builds_context():
    chunks = [{"doc_id": "a.txt", "text": "Київ столиця", "char_start": 0, "char_end": 12}]
    node = graph.make_retrieve_node(FakeStore(chunks), k=5)
    update = node({"question": "q"})
    assert "Київ столиця" in update["context"]
    assert "status" not in update  # no miss


def test_generate_node_ok_path():
    launcher = FakeLauncher(ChatResult(text="Київ", completion_tokens=3, latency_s=0.5))
    node = graph.make_generate_node(launcher, max_tokens=64, temperature=0.0, timeout=10)
    update = node({"question": "q", "context": "ctx"})
    assert update["status"] == common.OK
    assert update["answer"] == "Київ"
    assert update["usage"]["tokens_per_s"] > 0


def test_generate_node_short_circuits_on_retrieval_miss():
    launcher = FakeLauncher(ChatResult(text="should not be called"))
    node = graph.make_generate_node(launcher, max_tokens=64, temperature=0.0, timeout=10)
    update = node({"question": "q", "status": common.RETRIEVAL_MISS})
    assert update["answer"] == ""


def test_generate_node_timeout_status():
    launcher = FakeLauncher(ChatResult(text="", error=ERR_TIMEOUT))
    node = graph.make_generate_node(launcher, max_tokens=64, temperature=0.0, timeout=10)
    update = node({"question": "q", "context": "ctx"})
    assert update["status"] == ERR_TIMEOUT
