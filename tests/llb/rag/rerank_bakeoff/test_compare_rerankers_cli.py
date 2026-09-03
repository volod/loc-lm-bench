"""`compare-rerankers` end to end over a fake store and fake cross-encoders (no FAISS/GPU/network).

What the command itself owns: one shared pool retrieved at the configured depth, the roster screen,
the VRAM budget the fit gate reads, and the two written artifacts.
"""

import pytest
from typer.testing import CliRunner

from tests.llb.rag._rerank_bakeoff_helpers import (
    BASELINE,
    CANDIDATE,
    REMOTE_CODE_CANDIDATE,
    SCORERS,
    pool,
)

from llb.cli.app import app
from llb.core import env
from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset
from llb.rag.encoders.candidate_screen import SKIP_REMOTE_CODE
from llb.rag.embedding_bakeoff.models import BuiltStore
from llb.rag.rerank_bakeoff.models import ROW_NO_RERANK, LoadedScorer
from llb.rag.comparison.sidecar import sidecar_report

GOLD_POSITIONS = [4, 4, 1, 4, 4, 1, 4, 4]
POOL_DEPTH = 6


class _PoolStore:
    """Returns each question's fixed candidate pool, cut at the requested depth."""

    def __init__(self, by_question: dict[str, list]):
        self._by_question = by_question
        self.meta = {"dim": 8, "n_indexed": 10, "embedding_model": "m"}
        self.depths: list[int] = []

    def retrieve(self, question: str, k: int) -> list:
        self.depths.append(k)
        return self._by_question[question][:k]


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
                question=f"питання-{i:02d}",
                reference_answer="x",
                source_doc_id="d1",
                source_spans=[
                    SourceSpan(doc_id="d1", char_start=0, char_end=10, text="0123456789")
                ],
                provenance="ontology-drafted",
                split="final",
            )
            for i in range(len(GOLD_POSITIONS))
        ],
        goldset,
    )
    return corpus, goldset


@pytest.fixture
def store(monkeypatch):
    """Bind the fake store builder and the fake loader the command would otherwise download."""
    by_question = {
        f"питання-{i:02d}": pool(position, POOL_DEPTH) for i, position in enumerate(GOLD_POSITIONS)
    }
    fake = _PoolStore(by_question)
    released: list[str] = []
    fake.embedder = type("E", (), {"release": lambda self: released.append("released")})()

    monkeypatch.setattr(
        "llb.cli.rag.embedding_stores.local_store_builder",
        lambda cfg, stores_dir, **_kw: (
            lambda model: BuiltStore(store=fake, embed_seconds=1.0, index_bytes=100)
        ),
    )

    def fake_loader(**_kwargs):
        def load(model: str) -> LoadedScorer:
            return LoadedScorer(
                scorer=SCORERS[model],
                device="cpu",
                load_seconds=0.4,
                vram_mb={BASELINE: 1100.0, CANDIDATE: 2300.0}.get(model),
            )

        return load

    # Both seams: the command isolates each candidate in its own process by default and falls
    # back to the in-process loader under --in-process.
    monkeypatch.setattr("llb.rag.rerank_bakeoff.worker.isolated_loader", fake_loader)
    monkeypatch.setattr("llb.rag.rerank_bakeoff.loader.cross_encoder_loader", fake_loader)
    fake.released = released
    return fake


def _invoke(corpus, goldset, out, models, *extra):
    return CliRunner().invoke(
        app,
        [
            "compare-rerankers",
            "--goldset",
            str(goldset),
            "--corpus-root",
            str(corpus),
            "--models",
            models,
            "--k",
            "3",
            "--rerank-candidates",
            str(POOL_DEPTH),
            "--resamples",
            "200",
            "--out",
            str(out),
            *extra,
        ],
    )


def test_the_run_ranks_every_candidate_on_one_shared_pool_and_writes_both_artifacts(
    tmp_path, goldset_paths, store
):
    corpus, goldset = goldset_paths
    out = tmp_path / "report.md"
    result = _invoke(corpus, goldset, out, f"{BASELINE},{CANDIDATE}")
    assert result.exit_code == 0, result.output
    report = sidecar_report(out.with_suffix(".json"))
    assert {row["model"] for row in report["candidates"]} == {ROW_NO_RERANK, BASELINE, CANDIDATE}
    # One retrieval pass per item, at the configured pool depth -- not one pass per candidate.
    assert store.depths == [POOL_DEPTH] * len(GOLD_POSITIONS)
    assert report["pool_depth"] == POOL_DEPTH and report["k"] == 3
    assert out.read_text(encoding="utf-8").startswith("# Reranker bake-off")


def test_the_encoder_is_released_before_the_first_candidate_loads(tmp_path, goldset_paths, store):
    """A reranker's measured footprint has to be its own, not the encoder's plus its own."""
    corpus, goldset = goldset_paths
    result = _invoke(corpus, goldset, tmp_path / "report.md", BASELINE)
    assert result.exit_code == 0, result.output
    assert store.released == ["released"]


def test_an_unregistered_candidate_fails_the_run_before_anything_loads(
    tmp_path, goldset_paths, store
):
    corpus, goldset = goldset_paths
    result = _invoke(corpus, goldset, tmp_path / "report.md", f"{BASELINE},acme/mystery-reranker")
    assert result.exit_code == 2
    assert "acme/mystery-reranker" in result.output
    assert store.depths == []  # refused BEFORE the corpus was even retrieved


def test_a_remote_code_candidate_is_skipped_and_recorded_without_the_opt_in(
    tmp_path, goldset_paths, store
):
    corpus, goldset = goldset_paths
    out = tmp_path / "report.md"
    result = _invoke(corpus, goldset, out, f"{BASELINE},{REMOTE_CODE_CANDIDATE}")
    assert result.exit_code == 0, result.output
    report = sidecar_report(out.with_suffix(".json"))
    assert [row["model"] for row in report["skipped"]] == [REMOTE_CODE_CANDIDATE]
    assert report["skipped"][0]["reason"] == SKIP_REMOTE_CODE
    assert REMOTE_CODE_CANDIDATE not in {row["model"] for row in report["candidates"]}
    assert "Candidates not scored" in out.read_text(encoding="utf-8")


def test_a_declared_generator_residency_turns_the_footprint_into_a_fit_gate(
    tmp_path, goldset_paths, store, monkeypatch
):
    monkeypatch.setattr(
        "llb.backends.hardware.detect_gpus",
        lambda: [type("G", (), {"total_mb": 16000})()],
    )
    corpus, goldset = goldset_paths
    out = tmp_path / "report.md"
    result = _invoke(
        corpus, goldset, out, f"{BASELINE},{CANDIDATE}", "--generator-vram-mb", "14000"
    )
    assert result.exit_code == 0, result.output
    report = sidecar_report(out.with_suffix(".json"))
    assert report["headroom"]["headroom_mb"] == pytest.approx(1488.0)
    # 2300 MB does not fit beside a 14 GB generator; 1100 MB does.
    assert [row["model"] for row in report["skipped"]] == [CANDIDATE]
    assert next(r for r in report["candidates"] if r["model"] == BASELINE)["fits_headroom"] is True


def test_the_in_process_escape_hatch_uses_the_other_loader(tmp_path, goldset_paths, store):
    corpus, goldset = goldset_paths
    out = tmp_path / "report.md"
    result = _invoke(corpus, goldset, out, BASELINE, "--in-process")
    assert result.exit_code == 0, result.output
    report = sidecar_report(out.with_suffix(".json"))
    assert {row["model"] for row in report["candidates"]} == {ROW_NO_RERANK, BASELINE}


def test_an_unknown_adoption_bar_fails_the_run(tmp_path, goldset_paths, store):
    corpus, goldset = goldset_paths
    result = _invoke(
        corpus, goldset, tmp_path / "report.md", BASELINE, "--adoption-bars", "recall_at_k,nonsense"
    )
    assert result.exit_code == 2 and "nonsense" in result.output
