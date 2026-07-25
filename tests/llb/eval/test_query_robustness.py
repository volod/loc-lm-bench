"""Deterministic variants, fake end-to-end lanes, aggregation, and probe persistence."""

from pathlib import Path

import pytest

from llb.backends.base import ChatResult
from llb.core.config import RunConfig
from llb.eval import graph as eval_graph
from llb.eval.query_robustness import (
    LANE_NORMALIZE,
    LANE_NORMALIZE_TYPOS,
    LANE_OFF,
    MITIGATION_LANES,
    evaluate_query_robustness,
)
from llb.eval.query_robustness_report import write_robustness_artifacts
from llb.board.io import read_case_rows
from llb.eval.query_robustness_run import make_query_executor
from llb.eval.query_robustness_variants import (
    ALL_VARIANT_CLASSES,
    APOSTROPHE_MIXED_SCRIPT,
    APOSTROPHE_VARIANT,
    KEYBOARD_TYPOS,
    MIXED_SCRIPT,
    TRANSLITERATION,
    VARIANT_CLASSES,
    generate_variant,
    parse_variant_classes,
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


APOSTROPHE_QUESTION = "Який закон про пам'ять?"


@pytest.mark.parametrize("variant_class", ALL_VARIANT_CLASSES)
def test_variants_are_seeded_deterministic_and_non_identity(variant_class: str):
    kwargs = {"item_id": "q1", "seed": 17, "typo_rate": 0.1}
    first = generate_variant(APOSTROPHE_QUESTION, variant_class, **kwargs)
    assert first == generate_variant(APOSTROPHE_QUESTION, variant_class, **kwargs)
    assert first != APOSTROPHE_QUESTION


def test_variant_rate_validation():
    with pytest.raises(ValueError, match="between 0 and 1"):
        generate_variant("query", KEYBOARD_TYPOS, item_id="q", seed=1, typo_rate=1.1)


def test_split_classes_apply_one_mechanism_each_and_compose_into_the_combined_class():
    kwargs = {"item_id": "q1", "seed": 17, "typo_rate": 0.5}
    question = APOSTROPHE_QUESTION
    apostrophes = generate_variant(question, APOSTROPHE_VARIANT, **kwargs)
    homoglyphs = generate_variant(question, MIXED_SCRIPT, **kwargs)
    combined = generate_variant(question, APOSTROPHE_MIXED_SCRIPT, **kwargs)
    # the apostrophe class re-types the apostrophe and touches nothing else
    assert (
        apostrophes.replace("ʼ", "'").replace("’", "'").replace("‘", "'").replace("`", "'")
        == question
    )
    assert "'" not in apostrophes
    # the homoglyph class substitutes Latin look-alikes and leaves the apostrophe alone
    assert "'" in homoglyphs
    assert any(char in homoglyphs for char in "aeikmnoptuxc")
    # the combined class is exactly the composition of the two halves at the same seed
    assert combined == generate_variant(homoglyphs, APOSTROPHE_VARIANT, **kwargs)
    assert combined != apostrophes and combined != homoglyphs


def test_apostrophe_class_is_a_no_op_on_a_question_without_an_apostrophe():
    kwargs = {"item_id": "q1", "seed": 17, "typo_rate": 0.1}
    assert generate_variant("Який закон?", APOSTROPHE_VARIANT, **kwargs) == "Який закон?"


def test_class_selection_defaults_to_single_mechanism_classes_and_rejects_unknown_names():
    assert APOSTROPHE_MIXED_SCRIPT not in VARIANT_CLASSES
    assert parse_variant_classes(f"{MIXED_SCRIPT}, {APOSTROPHE_VARIANT} ,{MIXED_SCRIPT}") == (
        MIXED_SCRIPT,
        APOSTROPHE_VARIANT,
    )
    assert parse_variant_classes(APOSTROPHE_MIXED_SCRIPT) == (APOSTROPHE_MIXED_SCRIPT,)
    with pytest.raises(ValueError, match="unknown query robustness variant class"):
        parse_variant_classes("apostrophe")
    with pytest.raises(ValueError, match="at least one"):
        parse_variant_classes(" , ")


def test_clean_baseline_reads_canonical_case_rows_not_aggregate_rows(tmp_path: Path):
    scores = tmp_path / "scores.jsonl"
    scores.write_text('{"item_id":"q1","objective_score":1,"retrieval_hit":1}\n')
    assert read_case_rows(scores)[0]["item_id"] == "q1"
    scores.write_text('{"model":"aggregate"}\n')
    with pytest.raises(ValueError, match="per-case score row"):
        read_case_rows(scores)


class FakeStore:
    def __init__(self, *questions: str) -> None:
        self.questions = {question.casefold() for question in (questions or (_item().question,))}
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
        return [self.chunk] if question.casefold() in self.questions else []


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

    assert len(result.rows) == len(VARIANT_CLASSES) * len(MITIGATION_LANES)
    assert guard_loaded == [True]  # only the vocabulary-correction lane loads the morphology probe
    assert all(row["probe"] is True for row in result.rows)
    lanes = {(lane.variant_class, lane.mitigation): lane for lane in result.lanes}
    assert {lane.mitigation for lane in result.lanes} == {lane.id for lane in MITIGATION_LANES}
    for variant_class in (TRANSLITERATION, MIXED_SCRIPT, KEYBOARD_TYPOS):
        assert lanes[(variant_class, LANE_OFF.id)].recall_at_k == 0.0
        assert lanes[(variant_class, LANE_NORMALIZE_TYPOS.id)].recall_at_k == 1.0
        assert lanes[(variant_class, LANE_NORMALIZE_TYPOS.id)].recall_recovery == 1.0
    # the isolated lanes separate the two mechanisms: normalization alone inverts the noise it can
    # attribute (script and transliteration), and only keyboard typos need vocabulary correction
    assert lanes[(TRANSLITERATION, LANE_NORMALIZE.id)].recall_at_k == 1.0
    assert lanes[(MIXED_SCRIPT, LANE_NORMALIZE.id)].recall_at_k == 1.0
    assert lanes[(KEYBOARD_TYPOS, LANE_NORMALIZE.id)].recall_at_k == 0.0
    # each mechanism is its own class, so a recovery is attributable to one of them; the combined
    # class is opt-in, and a class with nothing to perturb reproduces the clean rows exactly
    assert APOSTROPHE_MIXED_SCRIPT not in {lane.variant_class for lane in result.lanes}
    assert lanes[(APOSTROPHE_VARIANT, LANE_OFF.id)].recall_at_k == 1.0
    assert lanes[(APOSTROPHE_VARIANT, LANE_OFF.id)].changed.n == 0

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
    assert len((out / "robustness.jsonl").read_text(encoding="utf-8").splitlines()) == len(
        result.rows
    )
    report = (out / "report.md").read_text(encoding="utf-8")
    assert f"| {APOSTROPHE_VARIANT} | `{LANE_NORMALIZE.id}` |" in report
    assert f"| {MIXED_SCRIPT} | `{LANE_NORMALIZE.id}` |" in report
    # the untouched apostrophe rows carry no measurement, so the subset table dashes them out
    assert f"| {APOSTROPHE_VARIANT} | `{LANE_OFF.id}` | 0 | - | - | - | - | - | - |" in report


def test_items_a_class_cannot_perturb_are_reported_apart_from_the_affected_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("llb.rag.lexical.load_uk_word_probe", lambda: lambda _token: False)
    monkeypatch.setattr("llb.eval.graph.build_rag_graph", build_fake_graph)
    plain_item = _item()
    apostrophe_item = GoldItem(
        **{**plain_item.model_dump(), "id": "q2", "question": APOSTROPHE_QUESTION}
    )
    store = FakeStore(apostrophe_item.question, plain_item.question)
    executor = make_query_executor(RunConfig(top_k=1, max_tokens=16), store, FakeEndpoint())
    clean_rows = [
        {"item_id": item.id, "objective_score": 1.0, "retrieval_hit": 1.0}
        for item in (apostrophe_item, plain_item)
    ]
    result = evaluate_query_robustness(
        [apostrophe_item, plain_item],
        clean_rows,
        executor,
        seed=13,
        typo_rate=0.1,
        variant_classes=[APOSTROPHE_VARIANT],
    )

    assert result.variant_classes == (APOSTROPHE_VARIANT,)
    lanes = {lane.mitigation: lane for lane in result.lanes}
    off, normalized = lanes[LANE_OFF.id], lanes[LANE_NORMALIZE.id]
    assert (off.n, off.changed.n) == (2, 1)
    # pooling the item the class cannot touch halves the visible loss; the subset states it whole
    assert (off.recall_at_k, off.recall_delta) == (0.5, -0.5)
    assert (off.changed.recall_at_k, off.changed.recall_delta) == (0.0, -1.0)
    assert normalized.recall_recovery == 0.5
    assert normalized.changed.recall_recovery == 1.0

    out = tmp_path / "query-robustness" / "run"
    write_robustness_artifacts(
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
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Affected items only" in report
    assert f"`{APOSTROPHE_VARIANT}` 1" in report
    assert (
        f"| {APOSTROPHE_VARIANT} | `{LANE_OFF.id}` | 1 | 0.0000 | -1.0000 | 0.0000 | -1.0000 |"
    ) in report
    assert (
        f"| {APOSTROPHE_VARIANT} | `{LANE_NORMALIZE.id}` | 1 | 1.0000 | +0.0000 | 1.0000 "
        "| +0.0000 | +1.0000 | +1.0000 |"
    ) in report
