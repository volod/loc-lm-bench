"""Language-fixture invariants and paired retrieval-translation measurement."""

from pathlib import Path

import pytest
from tests.llb.eval._query_robustness_helpers import (
    FakeEndpoint,
    FakeStore,
    _item,
    build_fake_graph,
)

from llb.core.config import RunConfig
from llb.eval.query_robustness.evaluate import (
    LANE_NORMALIZE,
    LANE_OFF,
    LANE_TRANSLATE,
    evaluate_query_robustness,
)
from llb.eval.query_robustness.languages import (
    LANGUAGE_MIXED,
    LANGUAGE_RU,
    compose_mixed_question,
    fixture_translation_queries,
    language_variant_id,
    language_fixture_status,
    load_language_variants,
    select_ukrainian_baseline,
)
from llb.eval.query_robustness.report import write_robustness_artifacts
from llb.eval.query_robustness.run import make_query_executor
from llb.goldset.schema import GoldItem, dump_goldset


def _variant(item: GoldItem, variant_class: str, question: str) -> GoldItem:
    lang = "ru" if variant_class == LANGUAGE_RU else "uk-ru"
    return GoldItem.model_validate(
        {
            **item.model_dump(),
            "id": language_variant_id(item.id, variant_class),
            "lang": lang,
            "question": question,
            "provenance": "frontier-drafted",
            "verified": False,
        }
    )


def _fixture(tmp_path: Path, item: GoldItem) -> Path:
    path = tmp_path / "goldset.jsonl"
    dump_goldset(
        [
            _variant(item, LANGUAGE_RU, "Какой закон?"),
            _variant(item, LANGUAGE_MIXED, "Який закон действует?"),
        ],
        path,
    )
    return path


def test_language_fixture_keeps_every_gold_field_and_span_unchanged(tmp_path: Path):
    item = _item()
    path = _fixture(tmp_path, item)
    variants = load_language_variants(path, [item], [LANGUAGE_RU, LANGUAGE_MIXED])

    assert language_fixture_status(path) == "drafted"
    assert variants[(item.id, LANGUAGE_RU)] == "Какой закон?"
    assert variants[(item.id, LANGUAGE_MIXED)] == "Який закон действует?"

    changed = _variant(item, LANGUAGE_RU, "Какой закон?")
    changed.reference_answer = "інша відповідь"
    dump_goldset([changed, _variant(item, LANGUAGE_MIXED, "Який закон действует?")], path)
    with pytest.raises(ValueError, match="changed gold content"):
        load_language_variants(path, [item], [LANGUAGE_RU, LANGUAGE_MIXED])


def test_language_fixture_review_state_must_be_uniform(tmp_path: Path):
    item = _item()
    russian = _variant(item, LANGUAGE_RU, "Какой закон?")
    mixed = _variant(item, LANGUAGE_MIXED, "Який закон действует?")
    russian.provenance = "human-verified"
    russian.verified = True
    path = tmp_path / "goldset.jsonl"
    dump_goldset([russian, mixed], path)
    with pytest.raises(ValueError, match="uniformly drafted or human-verified"):
        language_fixture_status(path)

    mixed.provenance = "human-verified"
    mixed.verified = True
    dump_goldset([russian, mixed], path)
    assert language_fixture_status(path) == "verified"


def test_mixed_question_composition_is_deterministic_and_uses_both_languages():
    ukrainian = "Коли було засновано Герцогство Нормандія?"
    russian = "Когда было основано Герцогство Нормандия?"
    mixed = compose_mixed_question(ukrainian, russian)
    assert mixed == "Коли было засновано Герцогство Нормандия?"
    assert mixed not in {ukrainian, russian}


def test_language_baseline_excludes_a_non_ukrainian_question_even_when_mislabeled():
    item = _item()
    english = GoldItem.model_validate(
        {**item.model_dump(), "id": "english", "question": "What law applies?"}
    )
    selected, excluded = select_ukrainian_baseline([item, english])
    assert selected == [item]
    assert excluded == [english.id]


def test_language_lanes_report_recall_mrr_answer_quality_and_translation_upper_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    item = _item()
    variants = load_language_variants(
        _fixture(tmp_path, item), [item], [LANGUAGE_RU, LANGUAGE_MIXED]
    )
    translations = fixture_translation_queries(variants, [item])
    monkeypatch.setattr("llb.eval.graph.build_rag_graph", build_fake_graph)
    executor = make_query_executor(
        RunConfig(top_k=1, max_tokens=16),
        FakeStore(),
        FakeEndpoint(),
        translations,
        lanes=(LANE_OFF, LANE_NORMALIZE, LANE_TRANSLATE),
    )
    clean_rows = [
        {
            "item_id": item.id,
            "objective_score": 1.0,
            "retrieval_hit": 1.0,
            "first_hit_rank": 1,
        }
    ]
    result = evaluate_query_robustness(
        [item],
        clean_rows,
        executor,
        seed=13,
        typo_rate=0.1,
        variant_classes=[LANGUAGE_RU, LANGUAGE_MIXED],
        language_variants=variants,
    )

    lanes = {(lane.variant_class, lane.mitigation): lane for lane in result.lanes}
    for variant_class in (LANGUAGE_RU, LANGUAGE_MIXED):
        assert lanes[(variant_class, LANE_OFF.id)].recall_at_k == 0.0
        assert lanes[(variant_class, LANE_NORMALIZE.id)].mrr == 0.0
        translated = lanes[(variant_class, LANE_TRANSLATE.id)]
        assert translated.recall_at_k == 1.0
        assert translated.mrr == 1.0
        assert translated.objective_score == 1.0
        assert translated.mrr_recovery == 1.0

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
            "language_fixture": "samples/goldsets/example_ru/goldset.jsonl",
            "language_fixture_status": "drafted",
            "language_baseline_excluded_ids": [],
        },
    )
    report = Path(paths["report"]).read_text(encoding="utf-8")
    assert "clean MRR" in report
    assert "benchmark-only exact paired retrieval upper bound" in report
    assert "run's `scores.jsonl` or correctness aggregates" in report
    assert "`normalize,typos` adds" not in report
    assert f"| {LANGUAGE_RU} | `{LANE_TRANSLATE.id}` |" in report
