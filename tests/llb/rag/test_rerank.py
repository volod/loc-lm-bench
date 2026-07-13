"""rerank-context-order: cross-encoder rerank seam, context-order policy, stage latency.

Pure: an injected fake scorer + fake stores drive the candidate flow, the kept set, and the
exact prompt ordering per policy -- no sentence-transformers, no FAISS, no GPU.
"""

import pytest

from llb.core.config import RunConfig
from llb.core.contracts import ChunkRecord
from llb.eval import common as eval_common
from llb.eval.graph import make_retrieve_node
from llb.executor.cases import score_case
from llb.executor.runner_metrics import _stage_latency
from llb.rag.rerank import RerankingRetriever, maybe_wrap_reranker, rerank_chunks


def _chunk(text: str, rank: int) -> ChunkRecord:
    return {
        "doc_id": "d1",
        "char_start": rank * 100,
        "char_end": rank * 100 + 10,
        "text": text,
        "rank": rank,
    }


def keyword_scorer(question: str, texts: list[str]) -> list[float]:
    """Deterministic fake cross-encoder: question-token overlap count per candidate."""
    tokens = set(question.split())
    return [float(len(tokens & set(text.split()))) for text in texts]


class FakeStore:
    """Records the requested depth; returns fixed candidates truncated to it."""

    def __init__(self, hits: list[ChunkRecord]) -> None:
        self.hits = hits
        self.requested: list[int] = []
        self.meta = {"mode": "flat"}

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        self.requested.append(k)
        return self.hits[:k]


# --- rerank_chunks: candidate flow + rank bookkeeping ---


def test_rerank_keeps_best_scored_and_renumbers():
    candidates = [_chunk("nothing here", 1), _chunk("kyiv capital", 2), _chunk("kyiv", 3)]
    kept = rerank_chunks("kyiv capital", candidates, keep_k=2, scorer=keyword_scorer)
    assert [c["text"] for c in kept] == ["kyiv capital", "kyiv"]
    assert [c["rank"] for c in kept] == [1, 2]
    assert [c["pre_rerank_rank"] for c in kept] == [2, 3]
    assert kept[0]["rerank_score"] == 2.0 and kept[1]["rerank_score"] == 1.0
    assert candidates[1]["rank"] == 2  # inputs never mutated


def test_rerank_all_equal_scores_is_an_ordering_noop():
    candidates = [_chunk("a", 1), _chunk("b", 2), _chunk("c", 3)]
    kept = rerank_chunks("q", candidates, keep_k=3, scorer=lambda q, t: [0.5] * len(t))
    assert [c["text"] for c in kept] == ["a", "b", "c"]  # stable sort keeps retrieval order


def test_rerank_rejects_bad_inputs():
    with pytest.raises(ValueError, match="keep_k"):
        rerank_chunks("q", [], 0, keyword_scorer)
    with pytest.raises(ValueError, match="scores"):
        rerank_chunks("q", [_chunk("a", 1)], 1, lambda q, t: [])


# --- RerankingRetriever: wrapper seam ---


def test_wrapper_retrieves_candidate_depth_and_keeps_top_k():
    hits = [_chunk("noise", 1), _chunk("kyiv capital", 2), _chunk("kyiv", 3), _chunk("x", 4)]
    store = FakeStore(hits)
    wrapper = RerankingRetriever(store, keyword_scorer, candidates=4)
    kept = wrapper.retrieve("kyiv capital", 2)
    assert store.requested == [4]  # pool depth, not k
    assert [c["text"] for c in kept] == ["kyiv capital", "kyiv"]
    assert len(wrapper.last_candidates) == 4  # pre-rerank pool exposed
    assert set(wrapper.stage_latency) == {"retrieve_s", "rerank_s"}
    mean = wrapper.mean_stage_latency()
    assert wrapper.n_queries == 1 and mean["retrieve_s"] >= 0.0


def test_wrapper_depth_never_below_k():
    store = FakeStore([_chunk(f"c{i}", i) for i in range(1, 9)])
    RerankingRetriever(store, keyword_scorer, candidates=2).retrieve("q", 5)
    assert store.requested == [5]


def test_wrapper_delegates_unknown_attributes():
    store = FakeStore([])
    wrapper = RerankingRetriever(store, keyword_scorer)
    assert wrapper.meta == {"mode": "flat"}  # delegated to the wrapped store


def test_maybe_wrap_reranker_is_identity_when_off():
    store = FakeStore([])
    config = RunConfig()
    assert maybe_wrap_reranker(store, config) is store
    wrapped = maybe_wrap_reranker(store, config.with_overrides(reranker="org/ce"))
    assert isinstance(wrapped, RerankingRetriever)
    assert wrapped.candidates == config.rerank_candidates


# --- context-order policy ---


def test_order_chunks_policies():
    chunks = [_chunk("a", 1), _chunk("b", 2)]
    assert eval_common.order_chunks(chunks, eval_common.ORDER_RANK) == chunks
    assert eval_common.order_chunks(chunks, eval_common.ORDER_REVERSE_RANK) == chunks[::-1]
    with pytest.raises(ValueError, match="context order"):
        eval_common.order_chunks(chunks, "sideways")


def test_format_context_reverse_rank_lays_best_last():
    chunks = [_chunk("best", 1), _chunk("worst", 2)]
    assert eval_common.format_context(chunks) == "[1] (d1)\nbest\n\n[2] (d1)\nworst"
    assert (
        eval_common.format_context(chunks, order=eval_common.ORDER_REVERSE_RANK)
        == "[1] (d1)\nworst\n\n[2] (d1)\nbest"
    )


def test_retrieve_node_applies_order_but_keeps_rank_order_in_state():
    store = FakeStore([_chunk("best", 1), _chunk("worst", 2)])
    node = make_retrieve_node(store, 2, context_order=eval_common.ORDER_REVERSE_RANK)
    state = node({"question": "q"})
    assert [c["text"] for c in state["retrieved"]] == ["best", "worst"]  # metrics unaffected
    assert state["context"].startswith("[1] (d1)\nworst")
    assert state["retrieve_latency_s"] >= 0.0
    assert "rerank_latency_s" not in state  # no reranking store wired


def test_retrieve_node_records_wrapper_stage_latency():
    store = FakeStore([_chunk("kyiv capital", 1), _chunk("noise", 2), _chunk("kyiv", 3)])
    ticks = iter(range(100))
    wrapper = RerankingRetriever(store, keyword_scorer, candidates=3, clock=lambda: next(ticks))
    state = make_retrieve_node(wrapper, 2)({"question": "kyiv capital"})
    assert state["retrieve_latency_s"] == 1.0 and state["rerank_latency_s"] == 1.0
    assert [c["text"] for c in state["retrieved"]] == ["kyiv capital", "kyiv"]


# --- RunConfig knobs + manifest fields ---


def test_config_rerank_fields_default_off_and_fingerprint():
    config = RunConfig()
    assert config.reranker is None and config.context_order == "rank"
    fingerprint = config.fingerprint()
    assert fingerprint["reranker"] is None
    assert fingerprint["rerank_candidates"] == 30
    assert fingerprint["context_order"] == "rank"


def test_config_rejects_bad_rerank_knobs():
    with pytest.raises(ValueError):
        RunConfig(context_order="sideways")
    with pytest.raises(ValueError):
        RunConfig(rerank_candidates=0)


def test_score_case_carries_stage_latency_and_aggregate_means():
    from tests.llb.executor.test_runner import gold_item

    item = gold_item("q1", "Яка столиця України?", "Київ", "Київ є столицею")
    state = {
        "answer": "Київ",
        "status": "ok",
        "retrieved": [],
        "usage": {"latency_s": 2.0, "tokens_per_s": 10.0, "completion_tokens": 4},
        "retrieve_latency_s": 0.25,
        "rerank_latency_s": 0.5,
    }
    row = score_case(item, state)
    assert row["retrieve_latency_s"] == 0.25 and row["rerank_latency_s"] == 0.5
    stage = _stage_latency([row])
    assert stage == {"retrieve_s": 0.25, "rerank_s": 0.5, "generate_s": 2.0}
    assert _stage_latency([]) == {}  # nothing measured -> no manifest field
