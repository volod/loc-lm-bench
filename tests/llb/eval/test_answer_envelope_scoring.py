"""typed-rag-answer-envelope -- declared-field answer-side scoring and the recorded bundle.

The envelope changes WHERE the answer-side signals come from, never how correctness is computed.
These tests pin both halves: the declared claims/citations produce the same-shaped columns the
prose heuristics do, and a run with the envelope OFF records exactly the columns it always did.
"""

import json

import pytest

from llb.backends.base import BackendLauncher, ChatResult
from llb.core.config import RunConfig
from llb.eval import common
from llb.eval.answer_envelope import lane, metrics
from llb.eval.answer_envelope.models import AnswerEnvelope
from llb.executor.cases import ScoreOptions, score_case
from llb.executor.runner import run_eval
from llb.goldset.schema import GoldItem

DOC = "Конвенцію підписано 1994 року у Женеві. Реєстр веде патентне відомство."
CHUNKS = [
    {"doc_id": "a.txt", "char_start": 0, "char_end": 38, "text": DOC[:38]},
    {"doc_id": "b.txt", "char_start": 39, "char_end": 70, "text": DOC[39:]},
]

# The per-case columns a scored free-text case has always carried. A new column leaking onto the
# free-text path breaks this list, which is the point of writing it out.
FREE_TEXT_COLUMNS = {
    "item_id",
    "split",
    "status",
    "objective_score",
    "token_f1",
    "token_precision",
    "token_recall",
    "ranking_score",
    "exact",
    "contains",
    "retrieval_hit",
    "first_hit_rank",
    "tokens_per_s",
    "latency_s",
    "completion_tokens",
    "answer_preview",
    # Answer-side gold-span coverage is scored on every path, envelope or prose
    # (answer-side-span-coverage-metric), so it belongs to the free-text shape too.
    "answer_span_coverage",
    "answer_all_spans",
    "answer_spans_measured",
    # The response-integrity guard is likewise always on, envelope or prose
    # (thinking-suppression-and-answer-language-guard): a declared envelope can carry a leaked
    # reasoning prefix or an off-language answer text just as free prose can.
    "reasoning_leak",
    "reasoning_leak_marker",
    "reasoning_leak_chars",
    "answer_language",
    "language_mismatch",
}


def _item(item_id="uk-1"):
    return GoldItem(
        id=item_id,
        lang="uk",
        question="Коли підписано конвенцію?",
        reference_answer="1994 року",
        source_doc_id="a.txt",
        source_spans=[{"doc_id": "a.txt", "char_start": 20, "char_end": 30, "text": DOC[20:30]}],
        provenance="human-authored",
        verified=True,
        split="final",
    )


def _envelope(citations=(1,), text="Конвенцію підписано 1994 року"):
    return AnswerEnvelope(
        answer="Конвенцію підписано 1994 року.",
        abstained=False,
        claims=[{"text": text, "citations": list(citations)}],
    )


def _state(envelope=None, answer=None, status=common.OK, chunks=None):
    state = {
        "answer": envelope.answer if envelope is not None else answer,
        "status": status,
        "retrieved": list(CHUNKS if chunks is None else chunks),
        "usage": {"completion_tokens": 12, "latency_s": 0.5, "tokens_per_s": 24.0},
    }
    if envelope is not None:
        state.update(
            {
                "envelope": envelope.model_dump(),
                "envelope_status": common.OK,
                "envelope_repaired": False,
            }
        )
    return state


def _options(**kwargs):
    return ScoreOptions(score_groundedness=True, cited_answers=True, **kwargs)


# --- declared-field metrics -------------------------------------------------------------------


def test_declared_citations_are_validated_against_the_chunk_they_name():
    report = metrics.envelope_citation_report(_envelope(citations=[1]), CHUNKS)
    assert report["citation_validity"] == 1.0
    assert report["citation_coverage"] == 1.0
    assert report["hallucinated_citation_rate"] == 0.0


def test_an_in_range_citation_of_the_wrong_chunk_is_invalid_but_not_hallucinated():
    report = metrics.envelope_citation_report(_envelope(citations=[2]), CHUNKS)
    assert report["citation_validity"] == 0.0
    assert report["hallucinated_citation_rate"] == 0.0
    assert report["citation_coverage"] == 1.0  # it DID cite -- the grounding is what failed


def test_an_out_of_range_citation_is_hallucinated():
    report = metrics.envelope_citation_report(_envelope(citations=[9]), CHUNKS)
    assert report["hallucinated_citation_rate"] == 1.0


def test_a_claim_with_no_citation_lowers_coverage_without_touching_validity():
    report = metrics.envelope_citation_report(_envelope(citations=[]), CHUNKS)
    assert report["citation_coverage"] == 0.0
    assert report["n_citations"] == 0 and report["citation_validity"] == 0.0


def test_declared_groundedness_counts_supported_claims():
    grounded = metrics.envelope_groundedness(_envelope(), CHUNKS)
    ungrounded = metrics.envelope_groundedness(
        _envelope(text="Реєстр веде міністерство фінансів України"), CHUNKS
    )
    assert grounded == 1.0 and ungrounded == 0.0


def test_a_declared_abstention_asserts_nothing_the_context_could_support():
    empty = AnswerEnvelope(answer="", abstained=True, claims=[])
    assert metrics.envelope_groundedness(empty, CHUNKS) == 0.0


def test_citations_are_read_in_prompt_layout_order():
    # Under reverse_rank the model saw the chunks flipped, so [1] names the LAST retrieved chunk.
    ranked = score_case(
        _item(),
        _state(envelope=_envelope(citations=[1], text="Реєстр веде патентне відомство")),
        options=_options(context_order=common.ORDER_RANK, answer_format=lane.ENVELOPE),
    )
    reversed_row = score_case(
        _item(),
        _state(envelope=_envelope(citations=[1], text="Реєстр веде патентне відомство")),
        options=_options(context_order=common.ORDER_REVERSE_RANK, answer_format=lane.ENVELOPE),
    )
    assert ranked["citation_validity"] == 0.0  # [1] is the first chunk, which lacks the claim
    assert reversed_row["citation_validity"] == 1.0  # flipped: [1] is the chunk that carries it


# --- the recorded per-case row ----------------------------------------------------------------


def test_an_envelope_answer_scores_exactly_as_the_free_text_of_the_same_string():
    envelope = _envelope()
    declared = score_case(
        _item(), _state(envelope=envelope), options=_options(answer_format=lane.ENVELOPE)
    )
    prose = score_case(_item(), _state(answer=envelope.answer), options=_options())
    for column in ("objective_score", "token_f1", "token_precision", "token_recall", "contains"):
        assert declared[column] == prose[column]


def test_envelope_columns_are_recorded_only_on_an_envelope_run():
    declared = score_case(
        _item(), _state(envelope=_envelope()), options=_options(answer_format=lane.ENVELOPE)
    )
    assert declared["envelope_status"] == common.OK
    assert declared["n_claims"] == 1
    assert declared["repaired"] is False
    assert declared["envelope_abstained"] is False
    prose = score_case(_item(), _state(answer="Конвенцію підписано 1994 року."))
    assert set(prose) == FREE_TEXT_COLUMNS


def test_a_non_conformant_case_records_its_typed_status_and_no_claims():
    state = _state(answer="")
    state.update({"envelope_status": common.SCHEMA_INVALID, "envelope_repaired": True})
    state["status"] = common.SCHEMA_INVALID
    row = score_case(_item(), state, options=_options(answer_format=lane.ENVELOPE))
    assert row["envelope_status"] == common.SCHEMA_INVALID
    assert row["repaired"] is True and row["n_claims"] == 0
    assert row["objective_score"] == 0.0  # typed as a format failure, still scored honestly


# --- the whole vertical -----------------------------------------------------------------------
#
# These go through the REAL runner rather than an injected `runner_fn`, so `run_eval` compiles the
# LangGraph app and the two tests below need the `[eval]` extra: `heavy_env` is what deselects them
# in the base [dev] install GitHub CI runs (they run in local `make ci` / `make test`).


class ScriptedLauncher(BackendLauncher):
    """Answers each case from a queue; the queue length is what the repair budget is spent from."""

    def __init__(self, responses):
        super().__init__(model="fake-uk", meta={"backend": "fake"})
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, max_tokens, temperature, timeout):
        self.calls += 1
        return ChatResult(text=self._responses.pop(0), completion_tokens=20, latency_s=0.5)


class FakeStore:
    def retrieve(self, question, k):
        return CHUNKS[:k]


def run_envelope_eval(tmp_path, responses, model="fake-uk", **overrides):
    """One envelope run over one item per response pair, through the real runner and fakes."""
    cfg = RunConfig(
        data_dir=tmp_path,
        run_name="envelope-test",
        model=model,
        top_k=2,
        answer_format="envelope",
        score_groundedness=True,
        cited_answers=True,
        **overrides,
    )
    items = [_item("uk-1"), _item("uk-2")]
    launcher = ScriptedLauncher(responses)
    return (
        cfg,
        run_eval(
            cfg,
            items=items,
            store=FakeStore(),
            launcher=launcher,
            mirror=lambda *a: None,
            emit=False,
        ),
        launcher,
    )


def _valid(answer="Конвенцію підписано 1994 року."):
    return json.dumps(
        {
            "answer": answer,
            "abstained": False,
            "claims": [{"text": "Конвенцію підписано 1994 року", "citations": [1]}],
        },
        ensure_ascii=False,
    )


@pytest.mark.heavy_env
def test_a_run_publishes_conformance_apart_from_correctness(tmp_path):
    cfg, result, launcher = run_envelope_eval(tmp_path, [_valid(), '{"answer": "Так"}', "проза"])
    metrics_out = result["metrics"]
    assert metrics_out["envelope_conformance"] == 0.5
    assert metrics_out["envelope_repair_rate"] == 0.5  # first-attempt conformance is 1 - 0.5
    assert metrics_out["envelope_malformed_rate"] == 0.5
    assert metrics_out["envelope_schema_invalid_rate"] == 0.0  # the repair moved it to malformed
    assert launcher.calls == 3  # two cases, one of them repaired once
    # correctness stays its own number, never blended with the format verdict
    assert "objective_score" in metrics_out and metrics_out["objective_score"] < 1.0
    rows = [
        json.loads(line)
        for line in (cfg.run_dir(result["run_timestamp"]) / "scores.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["envelope_status"] for row in rows] == [common.OK, common.MALFORMED]
    assert [row["repaired"] for row in rows] == [False, True]


@pytest.mark.heavy_env
def test_with_the_envelope_off_the_bundle_carries_no_envelope_column(tmp_path):
    cfg = RunConfig(data_dir=tmp_path, run_name="free-text", model="fake-uk", top_k=2)
    result = run_eval(
        cfg,
        items=[_item("uk-1")],
        store=FakeStore(),
        launcher=ScriptedLauncher(["Конвенцію підписано 1994 року."]),
        mirror=lambda *a: None,
        emit=False,
    )
    row = json.loads(
        (cfg.run_dir(result["run_timestamp"]) / "scores.jsonl").read_text(encoding="utf-8")
    )
    # The seam adds NOTHING to a free-text bundle: the recorded row keeps the retrieval-lane
    # columns it always had and gains no declared-answer column, and neither do the metrics.
    assert FREE_TEXT_COLUMNS < set(row)
    assert not any(key.startswith("envelope") or key in {"repaired", "n_claims"} for key in row)
    assert not any(key.startswith("envelope") for key in result["metrics"])
