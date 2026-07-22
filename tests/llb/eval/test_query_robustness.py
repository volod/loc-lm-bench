"""Deterministic variants, fake end-to-end lanes, aggregation, and probe persistence."""

from pathlib import Path

import pytest

from llb.backends.base import ChatResult
from llb.core.config import RunConfig
from llb.eval import graph as eval_graph
from llb.eval.query_robustness import evaluate_query_robustness
from llb.eval.query_robustness_report import write_robustness_artifacts
from llb.board.io import read_case_rows
from llb.eval.query_robustness_run import make_query_executor
from llb.eval.query_robustness_variants import (
    KEYBOARD_TYPOS,
    MIXED_SCRIPT,
    TRANSLITERATION,
    generate_variant,
)
from llb.goldset.schema import GoldItem


def _item() -> GoldItem:
    return GoldItem(
        id="q1",
        lang="uk",
        question="Який закон?",
        reference_answer="відповідь",
        source_doc_id="doc",
        source_spans=[{"doc_id": "doc", "char_start": 0, "char_end": 9, "text": "відповідь"}],
        provenance="human-authored",
        verified=True,
        split="final",
    )


@pytest.mark.parametrize("variant_class", [TRANSLITERATION, MIXED_SCRIPT, KEYBOARD_TYPOS])
def test_variants_are_seeded_deterministic_and_non_identity(variant_class: str):
    kwargs = {"item_id": "q1", "seed": 17, "typo_rate": 0.1}
    first = generate_variant(_item().question, variant_class, **kwargs)
    assert first == generate_variant(_item().question, variant_class, **kwargs)
    assert first != _item().question


def test_variant_rate_validation():
    with pytest.raises(ValueError, match="between 0 and 1"):
        generate_variant("query", KEYBOARD_TYPOS, item_id="q", seed=1, typo_rate=1.1)


def test_clean_baseline_reads_canonical_case_rows_not_aggregate_rows(tmp_path: Path):
    scores = tmp_path / "scores.jsonl"
    scores.write_text('{"item_id":"q1","objective_score":1,"retrieval_hit":1}\n')
    assert read_case_rows(scores)[0]["item_id"] == "q1"
    scores.write_text('{"model":"aggregate"}\n')
    with pytest.raises(ValueError, match="per-case score row"):
        read_case_rows(scores)


class FakeStore:
    def __init__(self) -> None:
        self.chunk = {
            "doc_id": "doc",
            "char_start": 0,
            "char_end": 20,
            "text": "який закон відповідь",
            "rank": 1,
            "retrieval_score": 1.0,
        }
        self.chunks = [self.chunk]

    def retrieve(self, question: str, k: int) -> list[dict[str, object]]:
        return [self.chunk] if question.casefold() == _item().question.casefold() else []


class FakeEndpoint:
    def chat(self, messages, max_tokens, temperature, timeout) -> ChatResult:
        return ChatResult(text="відповідь", latency_s=0.01)


class FakeGraphApp:
    """Run the production graph's pure nodes without the optional LangGraph package."""

    def __init__(self, retrieve, generate) -> None:
        self.retrieve = retrieve
        self.generate = generate

    def invoke(self, state):
        retrieved = {**state, **self.retrieve(state)}
        return {**retrieved, **self.generate(retrieved)}


def build_fake_graph(
    store,
    launcher,
    k,
    max_tokens,
    temperature,
    timeout,
    prompt_package=None,
    context_order="rank",
    query_prep=None,
    chunk_filter=None,
    cited=False,
):
    retrieve = eval_graph.make_retrieve_node(store, k, context_order, query_prep, chunk_filter)
    generate = eval_graph.make_generate_node(
        launcher, max_tokens, temperature, timeout, prompt_package, cited
    )
    return FakeGraphApp(retrieve, generate)


def test_fake_store_endpoint_measure_mitigation_and_keep_probe_rows_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    item = _item()
    guard_loaded = []

    def load_guard():
        guard_loaded.append(True)
        return lambda _token: False

    monkeypatch.setattr("llb.rag.lexical.load_uk_word_probe", load_guard)
    monkeypatch.setattr("llb.eval.graph.build_rag_graph", build_fake_graph)
    executor = make_query_executor(RunConfig(top_k=1, max_tokens=16), FakeStore(), FakeEndpoint())
    clean_rows = [{"item_id": item.id, "objective_score": 1.0, "retrieval_hit": 1.0}]
    result = evaluate_query_robustness([item], clean_rows, executor, seed=13, typo_rate=0.1)

    assert len(result.rows) == 6
    assert guard_loaded == [True]
    assert all(row["probe"] is True for row in result.rows)
    raw = {lane.variant_class: lane for lane in result.lanes if not lane.mitigated}
    mitigated = {lane.variant_class: lane for lane in result.lanes if lane.mitigated}
    assert all(lane.recall_at_k == 0.0 for lane in raw.values())
    assert all(lane.recall_at_k == 1.0 for lane in mitigated.values())
    assert all(lane.recall_recovery == 1.0 for lane in mitigated.values())

    out = tmp_path / "query-robustness" / "run"
    paths = write_robustness_artifacts(
        result,
        out,
        {
            "model": "fake",
            "backend": "fake",
            "split": "final",
            "seed": 13,
            "typo_rate": 0.1,
            "clean_run_dir": "run-eval/clean",
        },
    )
    assert set(out.iterdir()) == {Path(paths["report"]), Path(paths["robustness"])}
    assert not (out / "scores.jsonl").exists()
    assert len((out / "robustness.jsonl").read_text(encoding="utf-8").splitlines()) == 6
