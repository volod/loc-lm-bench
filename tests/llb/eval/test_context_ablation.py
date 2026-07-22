"""rag-vs-long-context-ablation -- three context lanes over one identical item set.

Pure and file-driven: the comparison consumes canonical per-case rows, the context sources are
plain state->state closures, and the orchestration takes an injected lane runner, so the whole
vertical runs in the lightweight CI install (no FAISS, no backend, no GPU). The CLI wiring layers
real stores and `run-eval` on top.
"""

import json
from pathlib import Path

import pytest

from llb.core.config import RunConfig
from llb.eval import common as eval_common
from llb.eval.context_ablation import (
    compare_context_strategies,
    format_report,
    lane_config,
    parse_lanes,
    run_context_ablation,
)
from llb.eval.context_ablation.derived import is_contaminated, skipped_item_ids
from llb.eval.context_ablation.models import (
    DERIVED_LONG_CONTEXT_DELTA,
    DERIVED_LONG_CONTEXT_DELTA_FITTING,
    DERIVED_RETRIEVAL_UPLIFT,
    LANE_CLOSED_BOOK,
    LANE_LONG_CONTEXT,
    LANE_RAG,
    VERDICT_LONG_CONTEXT_WINS,
    VERDICT_NO_RETRIEVAL_GAIN,
    VERDICT_RAG_PAYS_OFF,
    VERDICT_RETRIEVAL_INCONCLUSIVE,
)
from llb.eval.context_ablation.sources import (
    build_context_lane,
    closed_book_source,
    long_context_source,
    whole_document_chunk,
)
from llb.eval.graph import CLOSED_BOOK_TEMPLATE, build_messages
from llb.goldset.schema import GoldItem

FITS = "docs fit"
ALWAYS_FITS = lambda chars: True  # noqa: E731 -- a one-line test double reads better inline
NEVER_FITS = lambda chars: False  # noqa: E731


def _row(item_id: str, objective: float, hit: float = 1.0, **extra) -> dict:
    return {
        "item_id": item_id,
        "split": "final",
        "status": "ok",
        "objective_score": objective,
        "token_f1": objective,
        "exact": 0.0,
        "contains": 0.0,
        "retrieval_hit": hit,
        **extra,
    }


def _lanes(closed: list[dict], rag: list[dict], long_context: list[dict] | None = None) -> dict:
    lanes = {LANE_CLOSED_BOOK: closed, LANE_RAG: rag}
    if long_context is not None:
        lanes[LANE_LONG_CONTEXT] = long_context
    return lanes


def _types(*item_ids: str) -> dict[str, str]:
    return {item_id: "factoid" for item_id in item_ids}


def _derived(report, label):
    return next(entry for entry in report["derived"] if entry["label"] == label)


# --- lane selection -----------------------------------------------------------------------


def test_the_default_selection_is_all_three_lanes_with_closed_book_first():
    assert parse_lanes("closed_book,rag,long_context") == [
        LANE_CLOSED_BOOK,
        LANE_RAG,
        LANE_LONG_CONTEXT,
    ]


def test_closed_book_is_pulled_to_the_front_so_the_baseline_never_moves():
    assert parse_lanes("long_context,rag,closed_book")[0] == LANE_CLOSED_BOOK


def test_a_lane_selection_deduplicates_in_the_order_given():
    assert parse_lanes("rag, closed_book ,rag") == [LANE_CLOSED_BOOK, LANE_RAG]


@pytest.mark.parametrize("spec", ["", "faiss", "rag,oracle"])
def test_an_unknown_context_lane_is_rejected(spec: str):
    with pytest.raises(ValueError):
        parse_lanes(spec)


def test_lane_config_selects_the_strategy_run_eval_reproduces_the_bundle_from():
    base = RunConfig(model="m")
    lane = lane_config(base, LANE_LONG_CONTEXT, run_name_prefix="context-ablation")
    assert lane.context_strategy == LANE_LONG_CONTEXT
    assert lane.run_name == "context-ablation-long_context"
    assert lane_config(base, LANE_RAG, run_name_prefix="x").context_strategy == LANE_RAG


# --- context sources ----------------------------------------------------------------------


def test_closed_book_sends_no_context_but_must_not_raise_a_retrieval_miss():
    """`retrieval_miss` short-circuits generation, and a lane that never calls the model
    measures nothing -- an empty context is the POINT of this lane, not a failure."""
    update = closed_book_source()({"question": "q", "gold_spans": []})
    assert update["retrieved"] == []
    assert update["context"] == ""
    assert "status" not in update


def test_the_closed_book_prompt_carries_the_question_and_no_context_block():
    messages = build_messages("Столиця України?", "IGNORED", template_id=CLOSED_BOOK_TEMPLATE)
    assert "IGNORED" not in "".join(message["content"] for message in messages)
    assert any("Столиця України?" in message["content"] for message in messages)


def test_long_context_lays_the_whole_gold_document_in_offset_exact():
    documents = {"d1.txt": "abcdef"}
    state = {"gold_spans": [{"doc_id": "d1.txt", "char_start": 2, "char_end": 4, "text": "cd"}]}
    update = long_context_source(documents, ALWAYS_FITS)(state)
    chunk = update["retrieved"][0]
    assert (chunk["char_start"], chunk["char_end"], chunk["text"]) == (0, 6, "abcdef")
    assert "abcdef" in update["context"]
    assert "status" not in update


def test_a_multi_document_item_carries_every_gold_document_once():
    documents = {"a.txt": "AAA", "b.txt": "BBB"}
    spans = [
        {"doc_id": "b.txt", "char_start": 0, "char_end": 1, "text": "B"},
        {"doc_id": "a.txt", "char_start": 0, "char_end": 1, "text": "A"},
        {"doc_id": "b.txt", "char_start": 1, "char_end": 2, "text": "B"},
    ]
    update = long_context_source(documents, ALWAYS_FITS)({"gold_spans": spans})
    assert [chunk["doc_id"] for chunk in update["retrieved"]] == ["b.txt", "a.txt"]


def test_a_document_that_does_not_fit_is_skipped_never_truncated():
    documents = {"d1.txt": "x" * 100_000}
    state = {"gold_spans": [{"doc_id": "d1.txt", "char_start": 0, "char_end": 1, "text": "x"}]}
    update = long_context_source(documents, NEVER_FITS)(state)
    assert update["status"] == eval_common.CONTEXT_OVERFLOW
    assert update["retrieved"] == []
    assert update["context"] == ""


def test_a_gold_document_missing_from_the_corpus_fails_loudly():
    state = {"gold_spans": [{"doc_id": "gone.txt", "char_start": 0, "char_end": 1, "text": "x"}]}
    with pytest.raises(SystemExit, match="gone.txt"):
        long_context_source({}, ALWAYS_FITS)(state)


def test_a_skipped_case_never_reaches_the_model():
    from llb.eval import graph

    calls: list[object] = []

    class Launcher:
        def chat(self, messages, **kwargs):  # pragma: no cover - must never run
            calls.append(messages)
            raise AssertionError("a skipped case must not call the backend")

    node = graph.make_generate_node(Launcher(), max_tokens=8, temperature=0.0, timeout=1)
    update = node({"question": "q", "status": eval_common.CONTEXT_OVERFLOW})
    assert update == {"answer": "", "usage": {}}
    assert calls == []


def test_the_rag_strategy_installs_no_context_source_at_all():
    assert build_context_lane(RunConfig(context_strategy=LANE_RAG)) is None


def test_the_closed_book_strategy_selects_the_closed_book_prompt():
    lane = build_context_lane(RunConfig(context_strategy=LANE_CLOSED_BOOK))
    assert lane is not None and lane.template_id == CLOSED_BOOK_TEMPLATE


def test_the_long_context_strategy_reads_the_corpus_and_keeps_the_rag_prompt(tmp_path: Path):
    (tmp_path / "d1.txt").write_text("документ", encoding="utf-8")
    lane = build_context_lane(
        RunConfig(context_strategy=LANE_LONG_CONTEXT, corpus_root=tmp_path), ALWAYS_FITS
    )
    assert lane is not None and lane.template_id is None
    state = {
        "gold_spans": [{"doc_id": "d1.txt", "char_start": 0, "char_end": 8, "text": "документ"}]
    }
    assert lane.source(state)["retrieved"][0]["text"] == "документ"


def test_the_context_budget_is_what_decides_a_skip():
    """`fits_context_chars` is the one budget rule; a small explicit budget must skip a big doc."""
    config = RunConfig(context_budget=1024, max_tokens=256)
    lane = build_context_lane(RunConfig(context_strategy=LANE_CLOSED_BOOK))
    assert lane is not None  # sanity: the strategy switch itself is wired
    from llb.optimize.tuning_space import fits_context_chars

    assert fits_context_chars(config, None, 0, 0, 500)
    assert not fits_context_chars(config, None, 0, 0, 100_000)


def test_a_whole_document_chunk_is_a_verbatim_corpus_slice():
    chunk = whole_document_chunk("d.md", "text")
    assert chunk["text"] == "text"
    assert (chunk["char_start"], chunk["char_end"]) == (0, 4)


# --- derived numbers ----------------------------------------------------------------------


def test_retrieval_uplift_is_rag_minus_closed_book_paired_per_item():
    ids = [f"q{i}" for i in range(8)]
    report = compare_context_strategies(
        _lanes([_row(i, 0.0, hit=0.0) for i in ids], [_row(i, 1.0) for i in ids]),
        _types(*ids),
        resamples=200,
    )
    uplift = _derived(report, DERIVED_RETRIEVAL_UPLIFT)
    assert (uplift["candidate"], uplift["reference"]) == (LANE_RAG, LANE_CLOSED_BOOK)
    assert uplift["paired"]["delta"]["mean"] == pytest.approx(1.0)
    assert report["verdict"]["decision"] == VERDICT_RAG_PAYS_OFF


def test_the_long_context_delta_is_stated_against_rag_not_against_the_baseline():
    ids = [f"q{i}" for i in range(8)]
    report = compare_context_strategies(
        _lanes(
            [_row(i, 0.0, hit=0.0) for i in ids],
            [_row(i, 0.5) for i in ids],
            [_row(i, 1.0) for i in ids],
        ),
        _types(*ids),
        resamples=200,
    )
    delta = _derived(report, DERIVED_LONG_CONTEXT_DELTA)
    assert (delta["candidate"], delta["reference"]) == (LANE_LONG_CONTEXT, LANE_RAG)
    assert delta["paired"]["delta"]["mean"] == pytest.approx(0.5)
    assert report["verdict"]["decision"] == VERDICT_LONG_CONTEXT_WINS


def test_a_skipped_item_gets_a_second_delta_over_the_items_the_lane_could_answer():
    """A skipped item scores zero; counting it as a long-context loss would be a lie."""
    ids = [f"q{i}" for i in range(8)]
    long_rows = [_row(i, 1.0) for i in ids[:-1]]
    long_rows.append(_row(ids[-1], 0.0, hit=0.0, status=eval_common.CONTEXT_OVERFLOW))
    report = compare_context_strategies(
        _lanes([_row(i, 0.0, hit=0.0) for i in ids], [_row(i, 0.5) for i in ids], long_rows),
        _types(*ids),
        resamples=200,
    )
    assert report["lanes"][LANE_LONG_CONTEXT]["skipped_item_ids"] == [ids[-1]]
    assert _derived(report, DERIVED_LONG_CONTEXT_DELTA)["n"] == 8
    fitting = _derived(report, DERIVED_LONG_CONTEXT_DELTA_FITTING)
    assert fitting["n"] == 7
    assert fitting["paired"]["delta"]["mean"] == pytest.approx(0.5)
    assert report["verdict"]["skipped"][LANE_LONG_CONTEXT] == 1


def test_without_a_skip_there_is_no_second_population_to_report():
    ids = ["q0", "q1"]
    report = compare_context_strategies(
        _lanes(
            [_row(i, 0.0) for i in ids], [_row(i, 0.0) for i in ids], [_row(i, 0.0) for i in ids]
        ),
        _types(*ids),
        resamples=0,
    )
    assert [entry["label"] for entry in report["derived"]] == [
        DERIVED_RETRIEVAL_UPLIFT,
        DERIVED_LONG_CONTEXT_DELTA,
    ]


def test_a_two_lane_comparison_reports_only_the_uplift():
    ids = ["q0", "q1"]
    report = compare_context_strategies(
        _lanes([_row(i, 0.0) for i in ids], [_row(i, 1.0) for i in ids]), _types(*ids), resamples=0
    )
    assert [entry["label"] for entry in report["derived"]] == [DERIVED_RETRIEVAL_UPLIFT]


def test_skips_are_read_from_the_terminal_status_not_from_a_missing_row():
    rows = [_row("a", 0.0), _row("b", 0.0, status=eval_common.CONTEXT_OVERFLOW)]
    assert skipped_item_ids(rows) == ["b"]


# --- contamination ------------------------------------------------------------------------


def test_a_closed_book_answer_that_matches_the_reference_is_flagged():
    assert is_contaminated({"exact": 1.0, "contains": 0.0})
    assert is_contaminated({"exact": 0.0, "contains": 1.0})
    assert not is_contaminated({"exact": 0.0, "contains": 0.0, "objective_score": 0.9})


def test_the_contamination_rate_is_measured_on_the_closed_book_lane_only():
    ids = ["q0", "q1", "q2", "q3"]
    closed = [_row(i, 0.0) for i in ids]
    closed[0] = _row("q0", 1.0, exact=1.0, contains=1.0)
    report = compare_context_strategies(
        _lanes(closed, [_row(i, 1.0, exact=1.0, contains=1.0) for i in ids]),
        _types(*ids),
        resamples=0,
    )
    contamination = report["contamination"]
    assert contamination["lane"] == LANE_CLOSED_BOOK
    assert contamination["item_ids"] == ["q0"]
    assert contamination["rate"] == pytest.approx(0.25)
    assert report["verdict"]["contamination_rate"] == pytest.approx(0.25)
    assert [item["contaminated"] for item in report["items"]] == [True, False, False, False]


# --- verdicts -----------------------------------------------------------------------------


def test_a_noisy_uplift_stays_inconclusive_instead_of_claiming_rag_pays_off():
    ids = [f"q{i}" for i in range(12)]
    rag = [_row(i, 0.0) for i in ids]
    rag[0] = _row("q0", 1.0)
    report = compare_context_strategies(
        _lanes([_row(i, 0.0) for i in ids], rag), _types(*ids), resamples=200
    )
    assert report["verdict"]["decision"] == VERDICT_RETRIEVAL_INCONCLUSIVE
    assert "includes no difference" in report["verdict"]["reason"]


def test_a_model_that_answers_as_well_from_its_weights_records_no_retrieval_gain():
    ids = [f"q{i}" for i in range(6)]
    report = compare_context_strategies(
        _lanes([_row(i, 1.0) for i in ids], [_row(i, 1.0) for i in ids]), _types(*ids), resamples=50
    )
    assert report["verdict"]["decision"] == VERDICT_NO_RETRIEVAL_GAIN


def test_the_verdict_reads_the_fitting_delta_so_skips_cannot_sink_the_long_context_lane():
    ids = [f"q{i}" for i in range(10)]
    long_rows = [_row(i, 1.0) for i in ids[:4]]
    long_rows += [_row(i, 0.0, hit=0.0, status=eval_common.CONTEXT_OVERFLOW) for i in ids[4:]]
    report = compare_context_strategies(
        _lanes([_row(i, 0.0) for i in ids], [_row(i, 0.5) for i in ids], long_rows),
        _types(*ids),
        resamples=200,
    )
    assert _derived(report, DERIVED_LONG_CONTEXT_DELTA)["paired"]["delta"]["mean"] < 0
    assert report["verdict"]["decision"] == VERDICT_LONG_CONTEXT_WINS


def test_an_unknown_baseline_lane_is_rejected():
    with pytest.raises(ValueError, match="baseline lane"):
        compare_context_strategies({LANE_RAG: [_row("a", 1.0)]}, {}, baseline=LANE_CLOSED_BOOK)


# --- report -------------------------------------------------------------------------------


def test_the_report_leads_with_the_derived_numbers_and_stays_ascii():
    ids = [f"q{i}" for i in range(6)]
    closed = [_row(i, 0.0) for i in ids]
    closed[0] = _row("q0", 1.0, exact=1.0, contains=1.0)
    report = compare_context_strategies(
        _lanes(closed, [_row(i, 1.0) for i in ids], [_row(i, 1.0) for i in ids]),
        _types(*ids),
        resamples=50,
    )
    text = format_report(report, metadata={"model": "m", "backend": "b"})
    assert "# RAG versus long context" in text
    assert text.index("### Derived numbers") < text.index("### Per lane")
    assert "retrieval_uplift" in text
    assert "### Flagged items" in text
    assert "contaminated" in text
    assert text.isascii()


# --- orchestration ------------------------------------------------------------------------


def _gold_item(item_id: str, verified: bool = True) -> GoldItem:
    return GoldItem(
        id=item_id,
        lang="uk",
        question=f"питання {item_id}",
        reference_answer="відповідь",
        source_doc_id="doc.txt",
        source_spans=[{"doc_id": "doc.txt", "char_start": 0, "char_end": 9, "text": "відповідь"}],
        provenance="human-authored",
        verified=verified,
        split="final",
    )


def _write_bundle(goldset: Path, verified: bool = True) -> None:
    items = [_gold_item("q1", verified), _gold_item("q2", verified)]
    goldset.write_text(
        "".join(item.model_dump_json(exclude_none=True) + "\n" for item in items), encoding="utf-8"
    )
    (goldset.parent / "needle_items.jsonl").write_text(
        '{"id": "q1", "question_type": "multi-hop"}\n{"id": "q2", "question_type": "factoid"}\n',
        encoding="utf-8",
    )


def _recording_lane(tmp_path: Path, seen: list[tuple[str, str, tuple[str, ...]]]):
    """A fake lane runner whose objective rises with the amount of context each lane laid in."""
    objective = {LANE_CLOSED_BOOK: 0.0, LANE_RAG: 0.5, LANE_LONG_CONTEXT: 1.0}

    def fake_lane(config: RunConfig, items: list[GoldItem], split: str) -> Path:
        seen.append((config.run_name, config.context_strategy, tuple(i.id for i in items)))
        run_dir = tmp_path / "run-eval" / f"{config.run_name}-{split}"
        run_dir.mkdir(parents=True, exist_ok=True)
        scores = run_dir / "scores.jsonl"
        scores.write_text(
            "".join(
                json.dumps(_row(item.id, objective[config.context_strategy])) + "\n"
                for item in items
            ),
            encoding="utf-8",
        )
        return scores

    return fake_lane


def test_every_lane_scores_the_same_selected_items_and_the_comparison_persists(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    seen: list[tuple[str, str, tuple[str, ...]]] = []

    run = run_context_ablation(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        out_dir=tmp_path / "context-ablation",
        resamples=50,
        run_lane=_recording_lane(tmp_path, seen),
    )

    assert [entry[1] for entry in seen] == [LANE_CLOSED_BOOK, LANE_RAG, LANE_LONG_CONTEXT]
    assert {entry[2] for entry in seen} == {("q1", "q2")}
    assert run.report["item_ids"] == ["q1", "q2"]
    assert run.report["lanes"][LANE_RAG]["run_dirs"] == [
        str(tmp_path / "run-eval" / f"context-ablation-{LANE_RAG}-final")
    ]
    assert _derived(run.report, DERIVED_RETRIEVAL_UPLIFT)["paired"]["delta"][
        "mean"
    ] == pytest.approx(0.5)
    persisted = json.loads(Path(run.paths["comparison"]).read_text(encoding="utf-8"))
    assert persisted["metadata"]["split"] == "final"
    assert persisted["metadata"]["grounding"] == "verified"
    assert Path(run.paths["report"]).read_text(encoding="utf-8").startswith("# RAG versus")


def test_the_baseline_lane_cannot_be_dropped_from_the_selection(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    with pytest.raises(ValueError, match="closed_book"):
        run_context_ablation(
            RunConfig(data_dir=tmp_path, goldset_path=goldset), [LANE_RAG, LANE_LONG_CONTEXT]
        )


def test_a_single_lane_is_not_a_comparison(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    with pytest.raises(ValueError, match="at least one other lane"):
        run_context_ablation(RunConfig(data_dir=tmp_path, goldset_path=goldset), [LANE_CLOSED_BOOK])


def test_several_splits_pool_into_one_compared_item_set(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    items = [_gold_item("q1"), _gold_item("q2")]
    items[1].split = "tuning"
    goldset.write_text(
        "".join(item.model_dump_json(exclude_none=True) + "\n" for item in items), encoding="utf-8"
    )
    seen: list[tuple[str, str, tuple[str, ...]]] = []

    run = run_context_ablation(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        [LANE_CLOSED_BOOK, LANE_RAG],
        splits=["final", "tuning"],
        out_dir=tmp_path / "context-ablation",
        resamples=0,
        run_lane=_recording_lane(tmp_path, seen),
    )

    assert [entry[2] for entry in seen] == [("q1",), ("q2",), ("q1",), ("q2",)]
    assert run.report["item_ids"] == ["q1", "q2"]
    assert len(run.report["lanes"][LANE_RAG]["run_dirs"]) == 2


def test_a_split_that_selects_nothing_fails_instead_of_shrinking_the_item_set(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    with pytest.raises(SystemExit, match="tuning"):
        run_context_ablation(
            RunConfig(data_dir=tmp_path, goldset_path=goldset),
            splits=["final", "tuning"],
            run_lane=_recording_lane(tmp_path, []),
        )
