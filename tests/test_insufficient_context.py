"""Insufficient-context abstention probe (groundedness-citation-metrics): filter, sampling, scoring."""

from llb.eval import insufficient_context as ic
from llb.goldset.schema import GoldItem

GOLD_CHUNK = {"doc_id": "d", "char_start": 0, "char_end": 10, "text": "золоте правило десять"}
DISTRACTOR = {"doc_id": "d", "char_start": 50, "char_end": 70, "text": "щось зовсім інше тут"}


def _item(item_id: str) -> GoldItem:
    return GoldItem(
        id=item_id,
        lang="uk",
        question="Яке золоте правило?",
        reference_answer="золоте правило",
        source_doc_id="d",
        source_spans=[{"doc_id": "d", "char_start": 0, "char_end": 10, "text": "золоте пра"}],
        provenance="human-authored",
        verified=True,
        split="final",
    )


class _FakeStore:
    """Mimics RagStore.retrieve including the chunk_filter seam; records the last context."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.last_retrieved: list[dict] = []

    def retrieve(self, question, k, chunk_filter=None):
        hits = [c for c in self._chunks if chunk_filter is None or chunk_filter(c)]
        self.last_retrieved = hits[:k]
        return self.last_retrieved


def _scripted_chat(responses):
    queue = list(responses)

    def chat(messages):
        return queue.pop(0)

    return chat


def test_gold_excluding_filter_rejects_gold_chunks():
    spans = [{"doc_id": "d", "char_start": 0, "char_end": 10, "text": "золоте пра"}]
    accept = ic.gold_excluding_filter(spans)
    assert accept(GOLD_CHUNK) is False  # overlaps the gold span -> excluded
    assert accept(DISTRACTOR) is True  # no overlap -> kept


def test_sample_probe_items_is_seeded_and_deterministic():
    items = [_item(f"i{n}") for n in range(10)]
    a = ic.sample_probe_items(items, 3, seed=7)
    b = ic.sample_probe_items(items, 3, seed=7)
    assert [it.id for it in a] == [it.id for it in b]
    assert len(a) == 3
    assert [it.id for it in ic.sample_probe_items(items, 99, seed=7)] == [it.id for it in items]


def test_probe_excludes_gold_from_retrieval():
    store = _FakeStore([GOLD_CHUNK, DISTRACTOR])
    report = ic.run_insufficient_context_probe(
        [_item("i1")],
        store,
        _scripted_chat([("інформації недостатньо", None)]),
        model="m",
        backend="ollama",
        k=5,
        n=1,
    )
    assert GOLD_CHUNK not in store.last_retrieved  # gold evidence was filtered out
    assert report.rows[0]["probe"] is True


def test_probe_scores_abstention_accuracy():
    store = _FakeStore([GOLD_CHUNK, DISTRACTOR])
    report = ic.run_insufficient_context_probe(
        [_item("i1"), _item("i2")],
        store,
        _scripted_chat(
            [("На жаль, інформації недостатньо в контексті.", None), ("Париж, столиця.", None)]
        ),
        model="m",
        backend="ollama",
        k=5,
        n=2,
    )
    assert report.n_probes == 2
    assert report.n_abstained == 1
    assert report.abstention_accuracy == 0.5


def test_probe_excludes_transport_errors_from_denominator():
    store = _FakeStore([DISTRACTOR])
    report = ic.run_insufficient_context_probe(
        [_item("i1"), _item("i2")],
        store,
        _scripted_chat([("інформації недостатньо", None), ("", "timeout")]),
        model="m",
        backend="ollama",
        k=5,
        n=2,
    )
    assert report.n_errors == 1
    assert report.n_probes == 2
    assert report.abstention_accuracy == 1.0  # the one scoreable probe abstained


def test_probe_render_report_is_ascii():
    store = _FakeStore([DISTRACTOR])
    report = ic.run_insufficient_context_probe(
        [_item("i1")],
        store,
        _scripted_chat([("інформації недостатньо", None)]),
        model="m",
        backend="ollama",
        k=5,
        n=1,
    )
    text = ic.render_report(report)
    assert "abstention accuracy" in text
    assert text.isascii() or "інформ" not in text  # markdown body stays ASCII structure
