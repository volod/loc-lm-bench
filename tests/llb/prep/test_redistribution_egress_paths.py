"""Every export and frontier egress path refuses the committed projection's local-only document.

Each test stages `samples/corpora/acquired_projection_v1/` through real ingestion and drives one
path over material that includes `fixture-doc-07.md` (`licence: local-only`). The path must REFUSE
as a whole -- naming that document, writing no output, and making no provider call -- rather than
filter it. The control stages the identical fixture with that one sidecar set to
`redistributable` and must complete with the document's text in the payload, so each refusal is
attributable to the recorded licence alone and the gate is not vacuous: delete any wiring and the
refusal test fails because the path then exports the document.
"""

import json
import shutil
from pathlib import Path

import pytest

from llb.auto_rag.verification import verify_bundle
from llb.core.config import RunConfig
from llb.executor.runner_judge import _judge_cases
from llb.finetune.dataset import export_finetune_set
from llb.finetune.embedder.pairs import PAIRS_FILENAME, export_contrastive_pairs
from llb.goldset.schema import dump_goldset
from llb.judge.calibration_worksheet import write_worksheet_rows
from llb.prep.corpus.redistribution import RedistributionRefused, corpus_licences
from llb.prep.frontier.client import prepare_goldset
from llb.prep.goldset.cross_check import cross_check_corpus_goldset
from llb.prep.ontology.endpoints.config import EndpointCompleters, EndpointConfig, EndpointPlan
from llb.prep.ontology.endpoints.egress import refuse_frontier_corpus_egress
from llb.prep.ontology.pipeline.run import draft_goldset
from llb.rag.chunking.corpus import chunk_corpus
from llb.rag.embedding_bakeoff.run import api_lane_enabled
from llb.scoring.frontier_agreement import run_frontier_agreement
from tests.llb.bundle_fixtures import write_manifest
from tests.llb.prep.redistribution_fixtures import (
    LOCAL_ONLY_DOC,
    LOCAL_ONLY_PHRASE,
    PERMITTED_DOC,
    PERMITTED_PHRASE,
    fixture_items,
    span_item,
    staged_projection,
)

REFUSED = (LOCAL_ONLY_DOC,)


def _assert_refused(caught: pytest.ExceptionInfo[RedistributionRefused]) -> None:
    assert caught.value.doc_ids == REFUSED
    assert LOCAL_ONLY_DOC in str(caught.value)


def _never_called(*_args, **_kwargs):
    raise AssertionError("a refused egress path reached its provider")


# --- the two exports ----------------------------------------------------------------------------


def _export_pairs(tmp_path: Path, corpus: Path) -> Path:
    goldset = tmp_path / "goldset.jsonl"
    dump_goldset(fixture_items(corpus), goldset)
    out = tmp_path / "pairs"
    export_contrastive_pairs(goldset_path=goldset, corpus_root=corpus, out_dir=out, size=200)
    return out


def test_contrastive_pair_export_refuses_the_local_only_document(tmp_path: Path):
    corpus = staged_projection(tmp_path)

    with pytest.raises(RedistributionRefused) as caught:
        _export_pairs(tmp_path, corpus)

    _assert_refused(caught)
    assert not (tmp_path / "pairs" / PAIRS_FILENAME).exists()


def test_contrastive_pair_export_control_carries_the_document(tmp_path: Path):
    corpus = staged_projection(tmp_path, permit_local_only=True)

    out = _export_pairs(tmp_path, corpus)

    rows = [json.loads(line) for line in (out / PAIRS_FILENAME).read_text().splitlines()]
    assert any(LOCAL_ONLY_PHRASE in row["positive"] for row in rows)


def _finetune_bundle(tmp_path: Path, corpus: Path | None) -> tuple[Path, Path]:
    items = fixture_items(corpus or staged_projection(tmp_path / "unused"))
    goldset = tmp_path / "goldset.jsonl"
    dump_goldset(items, goldset)
    run = tmp_path / "run-tuning"
    run.mkdir()
    config: dict[str, object] = {"model": "base-model", "goldset_path": str(goldset)}
    if corpus is not None:
        config["corpus_root"] = str(corpus)
    write_manifest(run, split="tuning", config=config, n_cases=len(items))
    scores = [
        {"item_id": item.id, "split": "tuning", "status": "ok", "objective_score": 0.0}
        for item in items
    ]
    retrieval = [
        {
            "item_id": item.id,
            "retrieved": [
                {
                    "doc_id": PERMITTED_DOC,
                    "char_start": 0,
                    "char_end": 10,
                    "rank": 1,
                    "text_preview": PERMITTED_PHRASE,
                }
            ],
            "gold_spans": [span.model_dump() for span in item.source_spans],
        }
        for item in items
    ]
    (run / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scores))
    (run / "retrieval.jsonl").write_text("".join(json.dumps(row) + "\n" for row in retrieval))
    return run, goldset


def test_finetune_export_refuses_the_local_only_document(tmp_path: Path):
    corpus = staged_projection(tmp_path)
    run, goldset = _finetune_bundle(tmp_path, corpus)

    with pytest.raises(RedistributionRefused) as caught:
        export_finetune_set(run_dir=run, goldset_path=goldset, out_dir=tmp_path / "dataset")

    _assert_refused(caught)
    assert not (tmp_path / "dataset").exists()


def test_finetune_export_control_carries_the_document(tmp_path: Path):
    corpus = staged_projection(tmp_path, permit_local_only=True)
    run, goldset = _finetune_bundle(tmp_path, corpus)

    manifest = export_finetune_set(run_dir=run, goldset_path=goldset, out_dir=tmp_path / "dataset")

    assert manifest["item_ids"] == ["local-only-1", "permitted-1"]
    sft = (tmp_path / "dataset" / "sft.jsonl").read_text(encoding="utf-8")
    assert json.dumps(LOCAL_ONLY_PHRASE)[1:-1] in sft


def test_finetune_export_checks_an_explicit_corpus_over_the_recorded_one(tmp_path: Path):
    recorded = staged_projection(tmp_path / "recorded", permit_local_only=True)
    run, goldset = _finetune_bundle(tmp_path, recorded)
    refusing = staged_projection(tmp_path / "relocated")

    with pytest.raises(RedistributionRefused):
        export_finetune_set(
            run_dir=run, goldset_path=goldset, out_dir=tmp_path / "out", corpus_root=refusing
        )


def test_finetune_export_refuses_when_the_named_corpus_is_gone(tmp_path: Path):
    corpus = staged_projection(tmp_path, permit_local_only=True)
    run, goldset = _finetune_bundle(tmp_path, corpus)
    shutil.rmtree(corpus)

    with pytest.raises(ValueError, match="is not on this host"):
        export_finetune_set(run_dir=run, goldset_path=goldset, out_dir=tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_finetune_export_of_a_run_naming_no_corpus_is_unchanged(tmp_path: Path):
    run, goldset = _finetune_bundle(tmp_path, None)

    manifest = export_finetune_set(run_dir=run, goldset_path=goldset, out_dir=tmp_path / "dataset")

    assert manifest["n_sft"] == 2


# --- frontier egress behind consent -------------------------------------------------------------


def _frontier_plan() -> EndpointPlan:
    frontier = EndpointConfig(
        kind="frontier", model="openai/fake", egress_consent=True, max_calls=10
    )
    local = EndpointConfig(kind="local", model="fake-local")
    return EndpointPlan(extraction=local, drafting=frontier)


def test_frontier_ontology_drafting_refuses_before_any_call_or_bundle(tmp_path: Path):
    corpus = staged_projection(tmp_path)
    out = tmp_path / "bundle"

    with pytest.raises(RedistributionRefused) as caught:
        draft_goldset(
            corpus,
            _frontier_plan(),
            completers=EndpointCompleters.single(_never_called),
            out_dir=out,
        )

    _assert_refused(caught)
    assert "drafting" in caught.value.egress
    assert not out.exists()


def test_frontier_ontology_drafting_gate_follows_the_documents_it_will_read(tmp_path: Path):
    corpus = staged_projection(tmp_path)
    local = EndpointPlan.single(EndpointConfig(kind="local", model="fake-local"))

    refuse_frontier_corpus_egress(local, corpus)  # nothing leaves the host
    refuse_frontier_corpus_egress(_frontier_plan(), corpus, doc_limit=6)  # docs 01-06 only
    with pytest.raises(RedistributionRefused):
        refuse_frontier_corpus_egress(_frontier_plan(), corpus, doc_limit=7)


def _judge_batch(corpus: Path):
    item = fixture_items(corpus)[0]
    chunk = next(
        chunk
        for chunk in chunk_corpus(corpus, "recursive", 400, 0)
        if chunk["doc_id"] == LOCAL_ONLY_DOC
    )

    class Batch:
        answers = [(item, "an answer")]
        retrieval_pairs = [([chunk], [])]
        rows: list[dict] = [{"item_id": item.id}]

    return Batch()


def _frontier_config(tmp_path: Path, corpus: Path) -> RunConfig:
    return RunConfig(
        data_dir=tmp_path,
        corpus_root=corpus,
        scorer_policy="frontier",
        judge_model="openai/fake-judge",
        scorer_egress_consent=True,
        frontier_max_calls=5,
    )


def test_frontier_scorer_refuses_retrieved_local_only_context(tmp_path: Path):
    corpus = staged_projection(tmp_path)

    with pytest.raises(RedistributionRefused) as caught:
        _judge_cases(
            _frontier_config(tmp_path, corpus),
            _judge_batch(corpus),
            judge_rho=0.9,
            scorer=_never_called,
            staging_dir=tmp_path,
        )

    _assert_refused(caught)


def test_frontier_scorer_control_judges_the_same_case(tmp_path: Path):
    corpus = staged_projection(tmp_path, permit_local_only=True)
    seen: list[list[str]] = []

    def scorer(records, _model):
        seen.extend(record["contexts"] for record in records)
        return [{"faithfulness": 1.0, "answer_relevancy": 1.0} for _ in records]

    _judge_cases(
        _frontier_config(tmp_path, corpus),
        _judge_batch(corpus),
        judge_rho=0.9,
        scorer=scorer,
        staging_dir=tmp_path,
    )
    assert any(LOCAL_ONLY_PHRASE in context for contexts in seen for context in contexts)


def _drafted_bundle(tmp_path: Path, corpus: Path) -> Path:
    """A drafted bundle as the ontology pipeline writes it: goldset, corpus copy, provenance."""
    bundle = tmp_path / "bundle"
    (bundle / "corpus").mkdir(parents=True)
    for doc_id in (LOCAL_ONLY_DOC, PERMITTED_DOC):
        shutil.copy(corpus / doc_id, bundle / "corpus" / doc_id)
    items = [
        span_item(corpus, LOCAL_ONLY_DOC, LOCAL_ONLY_PHRASE, item_id="i-0", split="calibration"),
        span_item(corpus, PERMITTED_DOC, PERMITTED_PHRASE, item_id="i-1", split="tuning"),
        span_item(corpus, PERMITTED_DOC, PERMITTED_PHRASE, item_id="i-2", split="final"),
    ]
    for item in items:
        item.verified = False
        item.provenance = "ontology-drafted"
    dump_goldset(items, bundle / "goldset.jsonl")
    licences = corpus_licences(corpus)
    documents = [
        {"doc_id": doc, "licence": licences[doc]} for doc in (LOCAL_ONLY_DOC, PERMITTED_DOC)
    ]
    (bundle / "provenance.json").write_text(json.dumps({"documents": documents}), encoding="utf-8")
    return bundle


def _verify(tmp_path: Path, bundle: Path, frontier_complete) -> dict:
    return verify_bundle(
        bundle,
        tmp_path / "verify",
        policy="frontier",
        judge_model="openai/fake-judge",
        judge_base_url=None,
        threshold=0.5,
        min_accept_rate=0.0,
        egress_consent=True,
        max_usd=None,
        max_calls=10,
        scorer_ledger=tmp_path / "scorer_ledger.jsonl",
        frontier_complete=frontier_complete,
    )


def test_auto_rag_frontier_verification_refuses_by_bundle_provenance(tmp_path: Path):
    bundle = _drafted_bundle(tmp_path, staged_projection(tmp_path))

    with pytest.raises(RedistributionRefused) as caught:
        _verify(tmp_path, bundle, _never_called)

    _assert_refused(caught)
    assert not (tmp_path / "scorer_ledger.jsonl").exists()


def test_auto_rag_frontier_verification_control_calls_the_provider(tmp_path: Path):
    bundle = _drafted_bundle(tmp_path, staged_projection(tmp_path, permit_local_only=True))
    prompts: list[str] = []

    def complete(prompt: str) -> tuple[str, float, int, int]:
        prompts.append(prompt)
        return json.dumps({"faithfulness": 0.9, "answer_relevancy": 0.9}), 0.0, 10, 5

    _verify(tmp_path, bundle, complete)
    assert len(prompts) == 3


def _agreement_inputs(tmp_path: Path, corpus: Path) -> tuple[Path, Path]:
    items = fixture_items(corpus, split="calibration")
    goldset = tmp_path / "goldset.jsonl"
    dump_goldset(items, goldset)
    rows = [
        {
            "item_id": item.id,
            "split": "calibration",
            "provenance": "human-authored",
            "question": item.question,
            "reference_answer": item.reference_answer,
            "model_answer": f"answer #{index}",
            "human_answer": "",
            "human_rating": str(index + 1),
            "human_note": "",
            "human_status": "rated",
            "judge_rating": f"{0.4 * (index + 1):.2f}",
        }
        for index, item in enumerate(items)
    ]
    worksheet = tmp_path / "worksheet.csv"
    write_worksheet_rows(worksheet, rows)
    return worksheet, goldset


def test_frontier_judge_agreement_refuses_before_any_provider(tmp_path: Path):
    corpus = staged_projection(tmp_path)
    worksheet, goldset = _agreement_inputs(tmp_path, corpus)

    with pytest.raises(RedistributionRefused) as caught:
        run_frontier_agreement(
            worksheet,
            ["openai/fake-judge"],
            goldset=goldset,
            corpus_root=corpus,
            out_dir=tmp_path / "agreement",
            max_calls=10,
            complete_factory=_never_called,
        )

    _assert_refused(caught)
    assert not (tmp_path / "agreement").exists()


def test_frontier_judge_agreement_control_sends_the_document(tmp_path: Path):
    corpus = staged_projection(tmp_path, permit_local_only=True)
    worksheet, goldset = _agreement_inputs(tmp_path, corpus)
    prompts: list[str] = []

    def factory(_model: str):
        def complete(prompt: str) -> tuple[str, float, int, int]:
            prompts.append(prompt)
            return json.dumps({"faithfulness": 0.5, "answer_relevancy": 0.5}), 0.0, 10, 5

        return complete

    run_frontier_agreement(
        worksheet,
        ["openai/fake-judge"],
        goldset=goldset,
        corpus_root=corpus,
        out_dir=tmp_path / "agreement",
        max_calls=10,
        complete_factory=factory,
    )
    assert any("may not be" in prompt for prompt in prompts)


def test_compare_embeddings_api_row_refuses_before_asking_consent(tmp_path: Path):
    corpus = staged_projection(tmp_path)

    with pytest.raises(RedistributionRefused) as caught:
        api_lane_enabled("cohere/fake-embed", "open", _never_called, corpus)

    _assert_refused(caught)


def test_compare_embeddings_api_row_control_asks_consent(tmp_path: Path):
    corpus = staged_projection(tmp_path, permit_local_only=True)

    assert api_lane_enabled("cohere/fake-embed", "open", lambda: True, corpus) is True


def test_frontier_goldset_drafting_refuses_before_any_call(tmp_path: Path):
    corpus = staged_projection(tmp_path)
    out = tmp_path / "goldset.jsonl"

    with pytest.raises(RedistributionRefused) as caught:
        prepare_goldset(corpus, model="openai/fake", complete=_never_called, out_path=out)

    _assert_refused(caught)
    assert not out.exists()


def test_frontier_goldset_drafting_control_sends_every_document(tmp_path: Path):
    corpus = staged_projection(tmp_path, permit_local_only=True)
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return "[]"

    prepare_goldset(corpus, model="openai/fake", complete=complete)
    assert any(LOCAL_ONLY_PHRASE in prompt for prompt in prompts)


def test_second_frontier_cross_check_refuses_by_stem_or_path(tmp_path: Path):
    corpus = staged_projection(tmp_path)
    items = fixture_items(corpus)
    items[0].source_doc_id = Path(LOCAL_ONLY_DOC).stem  # drafts may cite the bare stem

    with pytest.raises(RedistributionRefused) as caught:
        cross_check_corpus_goldset(items, corpus, _never_called)

    _assert_refused(caught)


def test_second_frontier_cross_check_control_verifies_both_items(tmp_path: Path):
    corpus = staged_projection(tmp_path, permit_local_only=True)
    seen: list[str] = []

    def verify(item, doc_text):
        seen.append(item.id)
        return True, True, "ok"

    report = cross_check_corpus_goldset(fixture_items(corpus), corpus, verify)
    assert len(report.verdicts) == 2
    assert "local-only-1" in seen


def test_draft_cli_refuses_a_frontier_endpoint_before_asking_consent(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    from llb.cli.prep import draft_execution

    corpus = staged_projection(tmp_path)
    monkeypatch.setattr(draft_execution, "_confirm_frontier_egress", _never_called)
    request = SimpleNamespace(
        endpoint="frontier", egress_consent=False, doc_limit=None, max_calls=None, max_usd=None
    )
    resolved = draft_execution._ResolvedDraft(
        request=request, resuming=False, corpus_root=corpus, model="openai/fake"
    )

    with pytest.raises(RedistributionRefused) as caught:
        draft_execution._apply_egress_policy(resolved)

    _assert_refused(caught)
    assert request.egress_consent is False
