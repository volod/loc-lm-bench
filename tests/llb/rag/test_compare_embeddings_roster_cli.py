"""`compare-embeddings` roster screening end to end (fake stores; no FAISS, GPU, or network).

The bake-off's own guard rail: an unregistered candidate fails the run, and a candidate that needs
`trust_remote_code` is declined VISIBLY -- it lands in `report.json` as a skipped roster entry
rather than quietly making the table one row shorter.
"""

import json

import pytest
from typer.testing import CliRunner

from llb.cli.app import app
from llb.core import env
from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset
from llb.rag.embedding import remote_code_opt_in
from llb.rag.candidate_screen import SKIP_REMOTE_CODE
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
    return built


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


def test_allow_remote_code_builds_the_candidate_and_arms_the_process_knob(
    tmp_path, monkeypatch, goldset_paths
):
    corpus, goldset = goldset_paths
    # setenv (not delenv) so monkeypatch records the pre-state and REMOVES the knob the command
    # sets below -- a leaked opt-in would silently arm every later test in the session.
    monkeypatch.setenv(env.LLB_TRUST_REMOTE_CODE, "")
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
