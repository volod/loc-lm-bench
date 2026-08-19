"""`compare-embeddings` roster screening end to end (fake stores; no FAISS, GPU, or network).

The bake-off's own guard rail, in three parts: an unregistered candidate fails the run, a candidate
that needs `trust_remote_code` is declined VISIBLY, and a candidate whose repository code targets a
transformers major this interpreter is not is routed to the legacy scoring pass. All three land in
`report.json` as skipped roster entries rather than quietly making the table one row shorter.

The transformers major is injected here rather than read from the host: which candidates a run can
score is a property of the environment, and a test that only passes on one of the two environments
this repo deliberately maintains would be testing the host.
"""

import json

import pytest
from typer.testing import CliRunner

from llb.cli.app import app
from llb.core import env
from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset
from llb.rag.embedding import remote_code_opt_in
from llb.rag.candidate_screen import SKIP_LEGACY_STACK, SKIP_REMOTE_CODE
from llb.rag.card_parity import unpublished_result
from llb.rag.model_stack import (
    PINNED_TRANSFORMERS_MAJOR,
    REQUIRED_TRANSFORMERS_MAJOR_LEGACY,
)
from llb.rag.embedding_bakeoff_models import BuiltStore

from _embedding_bakeoff_uncertainty_helpers import BASELINE, _HitSetStore, _questions

REMOTE_CODE_MODEL = "jinaai/jina-embeddings-v3"


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Keep the run dir the command creates inside the test's tmp path, not the operator's."""
    monkeypatch.setenv(env.DATA_DIR, str(tmp_path / "data"))


@pytest.fixture
def goldset_paths(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    goldset = tmp_path / "goldset.jsonl"
    dump_goldset(
        [
            GoldItem(
                id=f"item-{i}",
                question=question,
                reference_answer="x",
                source_doc_id="d1",
                source_spans=[
                    SourceSpan(doc_id="d1", char_start=0, char_end=10, text="0123456789")
                ],
                provenance="ontology-drafted",
                split="final",
            )
            for i, question in enumerate(_questions(8))
        ],
        goldset,
    )
    return corpus, goldset


def _fake_builder(monkeypatch, hit_questions):
    built: list[str] = []

    def builder(cfg, stores_dir, **_kwargs):
        def build(model: str) -> BuiltStore:
            built.append(model)
            return BuiltStore(
                store=_HitSetStore(set(hit_questions)), embed_seconds=1.0, index_bytes=100
            )

        return build

    monkeypatch.setattr("llb.cli.rag.compare_embeddings.local_store_builder", builder)
    # The card gate loads a model; these fake-store runs assert screening, not parity.
    monkeypatch.setattr(
        "llb.cli.rag.compare_embeddings.probe_encoder_card",
        lambda model, **_kwargs: unpublished_result(model),
    )
    return built


def _on_stack(monkeypatch, major: int) -> None:
    """Pretend this interpreter holds that transformers major (pinned pass vs legacy pass)."""
    monkeypatch.setattr("llb.rag.model_stack.installed_transformers_major", lambda: major)


def _invoke(corpus, goldset, out, models, *extra):
    return CliRunner().invoke(
        app,
        [
            "compare-embeddings",
            "--goldset",
            str(goldset),
            "--corpus-root",
            str(corpus),
            "--models",
            models,
            "--k",
            "1",
            "--baseline",
            BASELINE,
            "--resamples",
            "50",
            "--out",
            str(out),
            *extra,
        ],
    )


def test_unregistered_candidate_fails_the_run_with_an_actionable_message(
    tmp_path, monkeypatch, goldset_paths
):
    corpus, goldset = goldset_paths
    built = _fake_builder(monkeypatch, _questions(8)[:4])
    result = _invoke(corpus, goldset, tmp_path / "report.md", f"{BASELINE},acme/mystery-encoder")
    assert result.exit_code == 2
    assert "acme/mystery-encoder" in result.output
    assert built == []  # refused BEFORE any store was built


def test_remote_code_candidate_is_skipped_and_recorded_without_the_opt_in(
    tmp_path, monkeypatch, goldset_paths
):
    corpus, goldset = goldset_paths
    built = _fake_builder(monkeypatch, _questions(8)[:4])
    out = tmp_path / "report.md"
    result = _invoke(corpus, goldset, out, f"{BASELINE},{REMOTE_CODE_MODEL}")
    assert result.exit_code == 0, result.output
    assert built == [BASELINE]
    report = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert [row["model"] for row in report["skipped"]] == [REMOTE_CODE_MODEL]
    assert report["skipped"][0]["reason"] == SKIP_REMOTE_CODE
    # The reader of report.md must see the row is declined, not beaten.
    assert "Roster entries not scored" in out.read_text(encoding="utf-8")
    assert REMOTE_CODE_MODEL not in {row["model"] for row in report["candidates"]}


def test_a_candidate_needing_the_legacy_transformers_is_routed_not_run(
    tmp_path, monkeypatch, goldset_paths
):
    """Opted in, on the PINNED stack: the row is still not scored, and says which pass scores it.

    This is the failure the routing exists to prevent -- on transformers 5.x this candidate raises
    at load, and its sibling `gte-multilingual-base` does something worse: it loads and returns
    numbers that do not reproduce its own card.
    """
    corpus, goldset = goldset_paths
    monkeypatch.setenv(env.LLB_TRUST_REMOTE_CODE, "")
    _on_stack(monkeypatch, PINNED_TRANSFORMERS_MAJOR)
    built = _fake_builder(monkeypatch, _questions(8)[:4])
    out = tmp_path / "report.md"
    result = _invoke(corpus, goldset, out, f"{BASELINE},{REMOTE_CODE_MODEL}", "--allow-remote-code")
    assert result.exit_code == 0, result.output
    assert built == [BASELINE]
    report = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert [row["reason"] for row in report["skipped"]] == [SKIP_LEGACY_STACK]
    assert "compare-embeddings-legacy" in report["skipped"][0]["detail"]


def test_allow_remote_code_builds_the_candidate_and_arms_the_process_knob(
    tmp_path, monkeypatch, goldset_paths
):
    corpus, goldset = goldset_paths
    # setenv (not delenv) so monkeypatch records the pre-state and REMOVES the knob the command
    # sets below -- a leaked opt-in would silently arm every later test in the session.
    monkeypatch.setenv(env.LLB_TRUST_REMOTE_CODE, "")
    # The legacy pass is where this candidate's repository code runs, so that is the stack the
    # scored-row assertions below belong on.
    _on_stack(monkeypatch, REQUIRED_TRANSFORMERS_MAJOR_LEGACY)
    built = _fake_builder(monkeypatch, _questions(8)[:4])
    out = tmp_path / "report.md"
    result = _invoke(corpus, goldset, out, f"{BASELINE},{REMOTE_CODE_MODEL}", "--allow-remote-code")
    assert result.exit_code == 0, result.output
    assert built == [BASELINE, REMOTE_CODE_MODEL]
    report = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert "skipped" not in report
    row = next(r for r in report["candidates"] if r["model"] == REMOTE_CODE_MODEL)
    # The row records BOTH the convention it was scored under and that repo code ran.
    assert row["family"] == "jina-v3" and row["trust_remote_code"] is True
    # The lazy reload behind retrieve() and the throughput profiler build their own Embedder,
    # so the opt-in has to reach them too.
    assert remote_code_opt_in()
