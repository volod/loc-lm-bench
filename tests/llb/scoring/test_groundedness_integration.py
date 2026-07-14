"""Tests for groundedness integration."""

from llb.eval import common
from llb.eval import graph
from llb.executor.cases import ScoreOptions, score_case
from test_groundedness import _item, _state


def test_build_messages_cited_uses_citation_prompt():
    plain = graph.build_messages("q", "ctx")
    cited = graph.build_messages("q", "ctx", cited=True)
    assert plain[0]["content"] != cited[0]["content"]
    assert "[1]" in cited[0]["content"]  # the cited-answer system prompt instructs [i] citations


def test_score_case_without_options_has_no_answer_side_fields():
    row = score_case(_item(), _state())
    assert "groundedness" not in row
    assert "citation_validity" not in row


def test_score_case_records_groundedness_and_citations_when_enabled():
    row = score_case(
        _item(),
        _state(),
        options=ScoreOptions(score_groundedness=True, cited_answers=True),
    )
    assert row["groundedness"] == 1.0
    assert row["citation_validity"] == 1.0
    assert row["citation_coverage"] == 1.0
    assert row["hallucinated_citation_rate"] == 0.0
    assert row["n_citations"] == 1


def test_manifest_metrics_carry_mean_citation_coverage():
    from llb.executor.runner_metrics import _attach_answer_side_metrics

    rows = [
        {"citation_validity": 0.0, "citation_coverage": 1.0},
        {"citation_validity": 0.0, "citation_coverage": 0.0},
    ]
    metrics: dict = {}
    _attach_answer_side_metrics(metrics, rows)  # type: ignore[arg-type]
    assert metrics["citation_coverage"] == 0.5
    assert metrics["citation_validity"] == 0.0


def test_score_case_citation_order_follows_context_order():
    # reverse_rank flips prompt positions, so [1] now points at the LAST retrieved chunk
    state = dict(_state())
    state["answer"] = "Патент захищає винахід двадцять років [1]."
    row = score_case(
        _item(),
        state,
        options=ScoreOptions(cited_answers=True, context_order=common.ORDER_REVERSE_RANK),
    )
    assert row["citation_validity"] == 1.0  # [1] == reversed()[0] == the patent chunk
