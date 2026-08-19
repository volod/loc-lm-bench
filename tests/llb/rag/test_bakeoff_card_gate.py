"""The card gate inside both bake-off lanes: a candidate that runs and is wrong is not a row.

The lane-level statement the pure card modules cannot make on their own: a mismatching candidate
never reaches the ranking (no store is built for it, no pass is scored), it lands in `skipped` with
the diagnosis, and a candidate that clears the gate carries its verdict on its row -- so a reader
can tell a verified lead from an unverified one.
"""

from _embedding_bakeoff_helpers import _chunk, _FakeStore, _fixed_builder, _items
from _rerank_bakeoff_helpers import BASELINE, fake_loader, items, pools

from llb.rag.card_parity import (
    SKIP_CARD_PARITY,
    STATUS_REPRODUCED,
    STATUS_UNPUBLISHED,
    CardExpectation,
    compare_to_card,
    unpublished_result,
)
from llb.rag.embedding_bakeoff import run_bakeoff
from llb.rag.rerank_bakeoff.lane import run_rerank_bakeoff
from llb.rag.rerank_bakeoff.models import ROW_NO_RERANK

CARD = "https://huggingface.co/acme/encoder"
# A registered reranker id whose fake scorer the helpers already bind, standing in for a candidate
# that loads and then fails its card.
BROKEN = "mixedbread-ai/mxbai-rerank-base-v2"


def _reproduced(model: str):
    return compare_to_card(model, CARD, CardExpectation(values=(1.0,)), (1.0,))


def _mismatch(model: str):
    return compare_to_card(model, CARD, CardExpectation(values=(1.0,)), (0.2,))


def _built(builder, seen):
    def build(model):
        seen.append(model)
        return builder(model)

    return build


def test_a_mismatching_encoder_is_never_built_and_lands_in_skipped():
    builder = _fixed_builder(_FakeStore([_chunk("d1", 0, 10)]))
    seen: list[str] = []
    report = run_bakeoff(
        _items(),
        k=10,
        corpus_root="corpus",
        local_models=["e5-base", "broken-encoder"],
        build_local=_built(builder, seen),
        card_parity=lambda model: (
            _mismatch(model) if model == "broken-encoder" else _reproduced(model)
        ),
    )
    assert seen == ["e5-base"]  # the store build is the expensive half; it never ran
    assert [row["model"] for row in report["candidates"]] == ["e5-base"]
    skipped = report["skipped"]
    assert [row["model"] for row in skipped] == ["broken-encoder"]
    assert skipped[0]["reason"] == SKIP_CARD_PARITY and CARD in skipped[0]["detail"]


def test_a_scored_encoder_row_carries_the_verdict_that_let_it_in():
    report = run_bakeoff(
        _items(),
        k=10,
        corpus_root="corpus",
        local_models=["e5-base"],
        build_local=_fixed_builder(_FakeStore([_chunk("d1", 0, 10)])),
        card_parity=_reproduced,
    )
    assert report["candidates"][0]["card_parity"]["status"] == STATUS_REPRODUCED


def test_an_unbound_gate_marks_every_row_unchecked_rather_than_verified():
    report = run_bakeoff(
        _items(),
        k=10,
        corpus_root="corpus",
        local_models=["e5-base"],
        build_local=_fixed_builder(_FakeStore([_chunk("d1", 0, 10)])),
    )
    assert report["candidates"][0]["card_parity"]["status"] == STATUS_UNPUBLISHED
    assert "skipped" not in report


def test_a_mismatching_reranker_is_not_scored_and_the_rest_of_the_roster_still_ranks(monkeypatch):
    monkeypatch.setattr(
        "llb.rag.rerank_bakeoff.lane.check_rerank_card",
        lambda model, _scorer: _mismatch(model) if model == BROKEN else unpublished_result(model),
    )
    scored_items = items(4)
    report = run_rerank_bakeoff(
        scored_items,
        pools([1, 2, 3, 4]),
        k=3,
        corpus_root="corpus",
        embedding_model="e5",
        chunking="recursive@800/120",
        pool_depth=3,
        batch_size=8,
        candidates=[BASELINE, BROKEN],
        load_scorer=fake_loader(),
        baseline=None,
    )
    scored = [row["model"] for row in report["candidates"]]
    assert scored == [ROW_NO_RERANK, BASELINE]
    assert [row["reason"] for row in report["skipped"]] == [SKIP_CARD_PARITY]


def test_the_reranker_report_states_the_precision_every_candidate_was_held_at():
    scored_items = items(4)
    report = run_rerank_bakeoff(
        scored_items,
        pools([1, 2, 3, 4]),
        k=3,
        corpus_root="corpus",
        embedding_model="e5",
        chunking="recursive@800/120",
        pool_depth=3,
        batch_size=8,
        candidates=[BASELINE],
        load_scorer=fake_loader(),
        dtype="float32",
        baseline=None,
    )
    assert report["dtype"] == "float32"
