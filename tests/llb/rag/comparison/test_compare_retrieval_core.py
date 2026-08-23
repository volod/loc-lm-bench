"""GraphRAG backend residual 3 -- graph-vs-FAISS retrieval comparison core (`llb.rag.comparison.run`).

Pure: driven by fake stores exposing the `.retrieve` seam, so it runs in the lightweight CI install
(no FAISS, no DuckDB, no GPU). The CLI wiring (`compare-retrieval`) layers real stores on top.
"""

from llb.rag.comparison.run import compare_retrieval
from llb.rag.comparison.models import ROW_ORACLE_DOC
from llb.rag.comparison.rows import (
    add_rerank_rows,
    add_stitch_rows,
    duplicate_census,
    format_comparison,
    stitch_report,
)
from llb.rag.stitching import StitchingRetriever
from llb.rag.duplicates.collapse import KEPT_BY_REQUEST, KEPT_BY_STRATEGY


from tests.llb.rag._compare_retrieval_helpers import (
    _FakeStore,
    _chunk,
    _exact_chunk,
    _items,
    _MetaStore,
)


class _QuestionStore:
    """Per-question hits plus a call ledger proving paired evidence reuses the scored pass."""

    def __init__(self, hits: dict[str, list[dict]]) -> None:
        self.hits = hits
        self.calls: list[str] = []

    def retrieve(self, question: str, k: int) -> list[dict]:
        self.calls.append(question)
        return self.hits[question][:k]


def test_compare_scores_each_backend_and_picks_recall_winner():
    stores = {
        "faiss": _FakeStore([_chunk("d1", 0, 10)]),  # overlaps the gold span -> hit
        "graph/local_khop": _FakeStore([_chunk("d1", 50, 60)]),  # no overlap -> miss
    }
    report = compare_retrieval(stores, _items(), k=5)
    assert report["k"] == 5 and report["n"] == 1
    assert report["backends"]["faiss"]["recall_at_k"] == 1.0
    assert report["backends"]["graph/local_khop"]["recall_at_k"] == 0.0
    assert report["best_recall"] == "faiss"


def test_compare_breaks_recall_ties_by_mrr_then_label():
    # both recall 1.0, but local_khop hits at rank 1 (higher MRR) vs faiss at rank 2
    stores = {
        "faiss": _FakeStore([_chunk("d1", 50, 60), _chunk("d1", 0, 10)]),
        "graph/local_khop": _FakeStore([_chunk("d1", 0, 10)]),
    }
    report = compare_retrieval(stores, _items(), k=5)
    assert report["backends"]["faiss"]["recall_at_k"] == 1.0
    assert report["backends"]["graph/local_khop"]["recall_at_k"] == 1.0
    assert report["best_recall"] == "graph/local_khop"  # higher MRR wins the tie


def test_compare_empty_backends_has_no_winner():
    report = compare_retrieval({}, _items(), k=3)
    assert report["best_recall"] is None
    assert report["backends"] == {}


def test_format_comparison_is_ascii_and_lists_backends():
    report = compare_retrieval({"faiss": _FakeStore([_chunk("d1", 0, 10)])}, _items(), k=5)
    text = format_comparison(report)
    assert text.isascii()  # AGENTS.md: ASCII-only output
    assert "faiss" in text and "recall@k" in text and "best (recall@k): faiss" in text


def test_format_comparison_handles_no_backends():
    text = format_comparison(compare_retrieval({}, _items(), k=5))
    assert "no backends loaded" in text


def test_duplicate_census_reads_only_stores_with_build_meta():
    stats = {
        "n": 4,
        "unique": 3,
        "collapsed": 1,
        "duplicate_chunks": 2,
        "duplicate_share": 0.5,
        "groups": 1,
        "largest_group": 2,
        "intra_document_groups": 1,
        "cross_document_groups": 0,
    }
    stores = {
        "faiss": _MetaStore([_chunk("d1", 0, 10)], stats),
        "graph/local_khop": _FakeStore([_chunk("d1", 0, 10)]),  # no meta -> no census row
    }
    census, kept = duplicate_census(stores)
    assert set(census) == {"faiss"}
    assert kept == {}  # it collapsed, so there is no reason to report
    report = compare_retrieval(stores, _items(), k=5)
    report["duplicates"] = census
    rendered = format_comparison(report)
    assert "1 intra-document, 0 cross-document" in rendered
    assert "3 indexed (1 collapsed)" in rendered
    assert rendered.isascii()


def test_census_says_all_copies_are_indexed_when_the_store_kept_them():
    """A `late` store -- or `--keep-duplicate-chunks` -- MEASURES its repeats and indexes them all."""
    stats = {
        "n": 4,
        "unique": 3,
        "collapsed": 1,
        "duplicate_chunks": 2,
        "duplicate_share": 0.5,
        "groups": 1,
        "largest_group": 2,
        "intra_document_groups": 1,
        "cross_document_groups": 0,
    }
    stores = {
        "late": _MetaStore([_chunk("d1", 0, 10)], stats, collapse_duplicates=False, strategy="late")
    }
    census, kept = duplicate_census(stores)
    assert kept == {"late": KEPT_BY_STRATEGY}
    report = compare_retrieval(stores, _items(), k=5)
    report["duplicates"] = census
    report["duplicates_kept"] = kept
    rendered = format_comparison(report)
    assert f"all 4 indexed ({KEPT_BY_STRATEGY})" in rendered
    assert "3 indexed (1 collapsed)" not in rendered
    assert rendered.isascii()


def test_census_separates_an_operator_request_from_the_strategy_rule():
    """`--keep-duplicate-chunks` on a text-only strategy reads as the request it was."""
    stats = {
        "n": 4,
        "unique": 3,
        "collapsed": 1,
        "duplicate_chunks": 2,
        "duplicate_share": 0.5,
        "groups": 1,
        "largest_group": 2,
        "intra_document_groups": 1,
        "cross_document_groups": 0,
    }
    stores = {
        "faiss": _MetaStore(
            [_chunk("d1", 0, 10)], stats, collapse_duplicates=False, strategy="recursive"
        )
    }
    _, kept = duplicate_census(stores)
    assert kept == {"faiss": KEPT_BY_REQUEST}


def test_census_of_an_artifact_recorded_before_the_key_still_reads_as_collapsed():
    report = compare_retrieval({"faiss": _FakeStore([_chunk("d1", 0, 10)])}, _items(), k=5)
    report["duplicates"] = {
        "faiss": {
            "n": 4,
            "unique": 3,
            "collapsed": 1,
            "duplicate_chunks": 2,
            "duplicate_share": 0.5,
            "groups": 1,
            "largest_group": 2,
            "intra_document_groups": 1,
            "cross_document_groups": 0,
        }
    }
    assert "3 indexed (1 collapsed)" in format_comparison(report)


def test_compare_reports_question_type_slices_without_retrieving_twice():
    store = _FakeStore([_chunk("d1", 0, 10)])
    report = compare_retrieval(
        {"fused/local_khop": store},
        [*_items(), *_items()],
        k=5,
        slice_labels=["comparative", "multi-hop"],
    )
    assert report["slices"]["comparative"]["n"] == 1
    assert report["slices"]["multi-hop"]["backends"]["fused/local_khop"]["mrr"] == 1.0
    rendered = format_comparison(report)
    assert "slice comparative (n=1)" in rendered
    assert "slice multi-hop (n=1)" in rendered


def test_focus_slices_are_reported_and_empty_ones_are_named_not_scored():
    store = _FakeStore([_chunk("d1", 0, 10)])
    report = compare_retrieval(
        {"faiss": store},
        [*_items(), *_items()],
        k=5,
        slice_labels=["numeric", "factoid"],
    )
    # every focus slice is in the JSON, so a zero reads as "this corpus labels none"
    assert {"numeric", "comparative", "multi-hop"} <= set(report["slices"])
    assert report["slices"]["numeric"]["n"] == 1
    assert report["slices"]["comparative"]["n"] == 0
    rendered = format_comparison(report)
    assert "slice numeric (n=1)" in rendered
    assert "slice factoid (n=1)" in rendered
    assert "slice comparative (n=0)" not in rendered
    assert "slices with no labeled item: comparative, multi-hop" in rendered


def test_compare_rejects_misaligned_slice_labels():
    import pytest

    with pytest.raises(ValueError, match="align"):
        compare_retrieval({}, _items(), k=5, slice_labels=[])


def test_add_rerank_rows_pairs_each_backend_and_skips_the_oracle():
    # rerank-context-order: the reranked twin scores the SAME store's candidates after the
    # cross-encoder cut, so the report shows the pre/post-rerank delta per backend. A scorer
    # that ranks the gold-hitting chunk first lifts MRR from 1/2 to 1 on the reranked row.
    def gold_first_scorer(question: str, texts: list[str]) -> list[float]:
        return [1.0 if text == "gold" else 0.0 for text in texts]

    hits = [_chunk("d1", 50, 60), {**_chunk("d1", 0, 10), "text": "gold"}]
    stores = {"faiss": _FakeStore(hits), ROW_ORACLE_DOC: _FakeStore(hits)}
    rows = add_rerank_rows(stores, gold_first_scorer, candidates=5)
    assert set(rows) == {"faiss", "faiss+rerank", ROW_ORACLE_DOC}  # oracle gets no twin
    report = compare_retrieval(rows, _items(), k=2)
    assert report["backends"]["faiss"]["mrr"] == 0.5  # gold at rank 2 pre-rerank
    assert report["backends"]["faiss+rerank"]["mrr"] == 1.0  # reranked to rank 1
    assert report["best_recall"] == "faiss+rerank"


def test_compare_keeps_item_vectors_and_adopts_a_paired_recall_winner_from_one_pass():
    questions = [f"q{index}" for index in range(20)]
    items = [
        (question, [{"doc_id": "gold", "char_start": 0, "char_end": 10, "text": "g"}])
        for question in questions
    ]
    miss = [_chunk("miss", 0, 10)]
    hit = [_chunk("gold", 0, 10)]
    baseline = _QuestionStore(
        {question: hit if index % 2 else miss for index, question in enumerate(questions)}
    )
    candidate = _QuestionStore({question: hit for question in questions})

    report = compare_retrieval(
        {"recursive": baseline, "sentence": candidate},
        items,
        k=1,
        item_ids=[f"item-{index}" for index in range(20)],
        baseline="recursive",
    )

    assert report["backends"]["recursive"]["recall_at_k"] == 0.5
    assert report["backends"]["sentence"]["recall_at_k"] == 1.0
    paired = report["backends"]["sentence"]["paired_vs_baseline"]["metrics"]["recall_at_k"]
    assert paired["delta"]["mean"] == 0.5
    assert (paired["wins"], paired["losses"], paired["ties"]) == (10, 0, 10)
    assert report["paired_items"][0] == {
        "item_id": "item-0",
        "lanes": {
            "recursive": {
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "span_char_coverage_at_k": 0.0,
                "span_intact_at_k": 0.0,
            },
            "sentence": {
                "recall_at_k": 1.0,
                "mrr": 1.0,
                "span_char_coverage_at_k": 1.0,
                "span_intact_at_k": 1.0,
            },
        },
    }
    assert report["verdict"]["decision"] == "adopt"
    assert report["verdict"]["lane"] == "sentence"
    assert baseline.calls == questions and candidate.calls == questions
    rendered = format_comparison(report)
    assert "paired vs recursive" in rendered
    assert "10/0/10" in rendered
    assert "Verdict: ADOPT `sentence`" in rendered
    assert rendered.isascii()


def test_mrr_can_adopt_only_when_recall_is_itemwise_identical():
    questions = [f"q{index}" for index in range(20)]
    spans = [{"doc_id": "gold", "char_start": 0, "char_end": 10, "text": "g"}]
    items = [(question, spans) for question in questions]
    miss = _chunk("miss", 0, 10)
    hit = _chunk("gold", 0, 10)
    baseline = _QuestionStore({question: [miss, hit] for question in questions})
    candidate = _QuestionStore({question: [hit, miss] for question in questions})

    report = compare_retrieval(
        {"recursive": baseline, "sentence": candidate},
        items,
        k=2,
        baseline="recursive",
    )

    recall = report["backends"]["sentence"]["paired_vs_baseline"]["metrics"]["recall_at_k"]
    assert (recall["wins"], recall["losses"]) == (0, 0)
    assert report["verdict"]["decision"] == "adopt"
    assert "itemwise-identical recall_at_k" in report["verdict"]["reason"]


def test_compare_rejects_misaligned_item_ids_and_unknown_baseline():
    import pytest

    stores = {"faiss": _FakeStore([_chunk("d1", 0, 10)])}
    with pytest.raises(ValueError, match="item ids"):
        compare_retrieval(stores, _items(), k=1, item_ids=[])
    with pytest.raises(ValueError, match="baseline"):
        compare_retrieval(stores, _items(), k=1, baseline="missing")


def test_disabling_resampling_cannot_turn_a_point_lead_into_adopt():
    questions = [f"q{index}" for index in range(20)]
    spans = [{"doc_id": "gold", "char_start": 0, "char_end": 10, "text": "g"}]
    items = [(question, spans) for question in questions]
    baseline = _QuestionStore({question: [_chunk("miss", 0, 10)] for question in questions})
    candidate = _QuestionStore({question: [_chunk("gold", 0, 10)] for question in questions})

    report = compare_retrieval(
        {"recursive": baseline, "sentence": candidate},
        items,
        k=1,
        baseline="recursive",
        resamples=0,
    )

    assert report["verdict"]["decision"] == "retain"
    assert "unmeasured" in report["verdict"]["reason"]
    assert "unmeasured" in format_comparison(report)


def test_intactness_columns_separate_two_lanes_recall_calls_equal():
    """A lane that cuts every gold span in half ties on recall and loses on intactness."""
    questions = [f"q{index}" for index in range(12)]
    spans = [{"doc_id": "d1", "char_start": 40, "char_end": 60, "text": "g"}]
    items = [(question, spans) for question in questions]
    whole = _QuestionStore({question: [_chunk("d1", 0, 100)] for question in questions})
    cut = _QuestionStore({question: [_chunk("d1", 0, 50)] for question in questions})

    report = compare_retrieval(
        {"recursive": whole, "sentence": cut},
        items,
        k=5,
        slice_labels=["numeric"] * len(questions),
        baseline="recursive",
    )

    assert report["backends"]["recursive"]["recall_at_k"] == 1.0
    assert report["backends"]["sentence"]["recall_at_k"] == 1.0  # recall cannot see the cut
    assert report["backends"]["recursive"]["span_char_coverage_at_k"] == 1.0
    assert report["backends"]["recursive"]["span_intact_at_k"] == 1.0
    assert report["backends"]["sentence"]["span_char_coverage_at_k"] == 0.5
    assert report["backends"]["sentence"]["span_intact_at_k"] == 0.0
    # the slice carries the same pair, scored from the same retrieval pass
    numeric = report["slices"]["numeric"]["backends"]
    assert numeric["sentence"]["span_intact_at_k"] == 0.0
    assert numeric["recursive"]["span_intact_at_k"] == 1.0
    # and the paired block reads the difference the recall block calls flat
    paired = report["backends"]["sentence"]["paired_vs_baseline"]["metrics"]
    assert paired["recall_at_k"]["delta"]["mean"] == 0.0
    assert paired["span_intact_at_k"]["delta"]["mean"] == -1.0
    assert paired["span_char_coverage_at_k"]["delta"]["mean"] == -0.5


def test_two_lanes_that_score_every_item_alike_are_named_identical_not_merely_flat():
    """The vector-backend case: a zero-discordance ledger settles the comparison for good."""
    questions = [f"q{index}" for index in range(20)]
    spans = [{"doc_id": "d1", "char_start": 0, "char_end": 10, "text": "g"}]
    items = [(question, spans) for question in questions]
    hits = {question: [_chunk("d1", 0, 10)] for question in questions}

    report = compare_retrieval(
        {"chroma": _QuestionStore(dict(hits)), "faiss": _QuestionStore(dict(hits))},
        items,
        k=1,
        baseline="faiss",
    )

    # `chroma` leads only because the label sort has to return something.
    assert report["best_recall"] == "chroma"
    verdict = report["verdict"]
    assert verdict["decision"] == "retain" and verdict["lane"] == "faiss"
    assert "itemwise identical to `faiss` on every scored metric over 20 items" in verdict["reason"]
    assert "no larger item set could separate the two lanes" in verdict["reason"]


def test_intactness_never_decides_the_verdict():
    """Only recall (or itemwise-identical recall plus MRR) may adopt; intactness reports only."""
    questions = [f"q{index}" for index in range(12)]
    spans = [{"doc_id": "d1", "char_start": 40, "char_end": 60, "text": "g"}]
    items = [(question, spans) for question in questions]
    whole = _QuestionStore({question: [_chunk("d1", 0, 100)] for question in questions})
    cut = _QuestionStore({question: [_chunk("d1", 0, 50)] for question in questions})

    report = compare_retrieval(
        {"recursive": cut, "sentence": whole}, items, k=5, baseline="recursive"
    )

    assert report["verdict"]["decision"] == "retain"
    assert report["verdict"]["lane"] == "recursive"
    adjustment = report["verdict"]["selection_adjustment"]
    assert adjustment["family_size"] == 2  # 1 candidate lane x 2 verdict bars, no intactness


def test_the_report_renders_the_intactness_columns_and_their_paired_block():
    report = compare_retrieval(
        {
            "recursive": _FakeStore([_chunk("d1", 0, 100)]),
            "sentence": _FakeStore([_chunk("d1", 0, 5)]),
        },
        [("питання", [{"doc_id": "d1", "char_start": 0, "char_end": 10, "text": "g"}])],
        k=5,
        slice_labels=["numeric"],
        baseline="recursive",
    )
    rendered = format_comparison(report)
    assert "cover@k intact@k  chars@k" in rendered
    assert "coverage delta [lo, hi]" in rendered and "intact delta [lo, hi]" in rendered
    assert rendered.isascii()
    # the slice block reuses the same four quality columns and the served-cost column beside them
    slice_row = [line for line in rendered.splitlines() if line.startswith("    sentence")]
    assert slice_row and slice_row[0].split()[1:] == ["1.000", "1.000", "0.500", "0.000", "1"]


def test_a_separated_loss_is_named_regressed_not_flat():
    """`reading_of` is one-sided, so a lane the baseline beats must not print `flat`."""
    questions = [f"q{index}" for index in range(12)]
    spans = [{"doc_id": "d1", "char_start": 40, "char_end": 60, "text": "g"}]
    items = [(question, spans) for question in questions]
    whole = _QuestionStore({question: [_chunk("d1", 0, 100)] for question in questions})
    cut = _QuestionStore({question: [_chunk("d1", 0, 50)] for question in questions})

    report = compare_retrieval(
        {"recursive": whole, "sentence": cut}, items, k=5, baseline="recursive"
    )
    intact = report["backends"]["sentence"]["paired_vs_baseline"]["metrics"]["span_intact_at_k"]
    assert intact["delta"]["hi"] < 0.0
    rendered = format_comparison(report)
    sentence_rows = [line for line in rendered.splitlines() if line.startswith("  sentence   ")]
    assert any("regressed" in line for line in sentence_rows)


def test_stitched_twin_converts_fragments_and_reproduces_its_base_lanes_finding_metrics():
    """The fragmented-evidence lever, read the way the report reads it.

    The base lane retrieves a gold span cut across two adjacent chunks: found, fully covered, not
    intact. Its stitched twin retrieves the SAME chunks and merges them, so recall and coverage
    must reproduce the base lane exactly and only intactness may move.
    """
    questions = [f"q{index}" for index in range(20)]
    spans = [{"doc_id": "d1", "char_start": 40, "char_end": 60, "text": "g"}]
    items = [(question, spans) for question in questions]
    fragments = [_exact_chunk("d1", 0, 50), _exact_chunk("d1", 50, 100)]
    stores = add_stitch_rows({"recursive": _FakeStore(fragments)})

    report = compare_retrieval(
        stores,
        items,
        k=10,
        slice_labels=["procedural"] * len(questions),
        baseline="recursive",
    )
    report["stitching"] = stitch_report(report, stores)

    base = report["backends"]["recursive"]
    stitched = report["backends"]["recursive+stitch"]
    assert base["span_intact_at_k"] == 0.0 and stitched["span_intact_at_k"] == 1.0
    assert stitched["recall_at_k"] == base["recall_at_k"] == 1.0
    assert stitched["span_char_coverage_at_k"] == base["span_char_coverage_at_k"] == 1.0
    assert stitched["served_chars_at_k"] == base["served_chars_at_k"] == 100.0
    assert report["slices"]["procedural"]["backends"]["recursive+stitch"]["span_intact_at_k"] == 1.0

    entry = report["stitching"]["recursive+stitch"]
    assert entry["base"] == "recursive"
    assert entry["recall_invariant"] and entry["coverage_invariant"]
    assert entry["census"]["merged_per_query"] == 1.0

    rendered = format_comparison(report)
    assert "chars@k" in rendered
    assert "invariance held" in rendered
    assert "mrr compresses with the block count" in rendered
    assert rendered.isascii()


def test_stitch_report_names_a_lane_that_failed_the_invariance_it_rests_on():
    """A stitched lane that moved recall did not reflow evidence -- the report must say so."""
    items = [("q", [{"doc_id": "d1", "char_start": 0, "char_end": 10, "text": "g"}])]
    stores = {
        "recursive": _FakeStore([_chunk("miss", 0, 10)]),
        "recursive+stitch": _FakeStore([_chunk("d1", 0, 10)]),
    }
    report = compare_retrieval(stores, items, k=10, baseline="recursive")
    report["stitching"] = stitch_report(report, {"recursive+stitch": StitchingRetriever(object())})

    assert report["stitching"]["recursive+stitch"]["recall_invariant"] is False
    assert "INVARIANCE FAILED" in format_comparison(report)


def test_stitching_is_absent_from_a_report_with_no_stitched_lane():
    report = compare_retrieval({"faiss": _FakeStore([_chunk("d1", 0, 10)])}, _items(), k=5)
    assert stitch_report(report, {"faiss": _FakeStore([])}) == {}
    assert "+stitch" not in format_comparison(report)


def test_each_slice_carries_its_own_paired_reading_not_only_a_point_row():
    """A 14-item slice turns on one question, so a slice point delta needs its own interval."""
    questions = [f"q{index}" for index in range(12)]
    spans = [{"doc_id": "d1", "char_start": 0, "char_end": 100, "text": "g"}]
    items = [(question, spans) for question in questions]
    whole = _QuestionStore({question: [_chunk("d1", 0, 100)] for question in questions})
    cut = _QuestionStore({question: [_chunk("d1", 0, 50)] for question in questions})
    # the first six items are procedural, the rest factoid: the two slices must read differently
    labels = ["procedural"] * 6 + ["factoid"] * 6

    report = compare_retrieval(
        {"recursive": cut, "wider": whole},
        items,
        k=5,
        slice_labels=labels,
        baseline="recursive",
        resamples=200,
    )

    procedural = report["slices"]["procedural"]
    assert procedural["n"] == 6
    paired = procedural["backends"]["wider"]["paired_vs_baseline"]
    assert paired["baseline"] == "recursive"
    intact = paired["metrics"]["span_intact_at_k"]
    assert intact["delta"]["mean"] == 1.0
    assert (intact["wins"], intact["losses"], intact["ties"]) == (6, 0, 0)
    # the slice's own items only -- the baseline lane pairs against itself at exactly zero
    assert (
        procedural["backends"]["recursive"]["paired_vs_baseline"]["metrics"]["span_intact_at_k"][
            "delta"
        ]["mean"]
        == 0.0
    )
    # an empty focus slice scores nothing and pairs nothing rather than reporting invented zeros
    assert report["slices"]["numeric"]["n"] == 0
    assert "paired_vs_baseline" not in report["slices"]["numeric"]["backends"]["wider"]

    rendered = format_comparison(report)
    slice_block = rendered.split("slice procedural")[1].split("slice ")[0]
    assert "intact delta [lo, hi]" in slice_block
    assert "coverage delta [lo, hi]" in slice_block
    # the finding pair stays on the aggregate table, so a slice block does not repeat it
    assert "recall delta [lo, hi]" not in slice_block
