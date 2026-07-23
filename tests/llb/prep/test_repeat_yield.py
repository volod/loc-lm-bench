"""Per-question yield audit for `--repeat-blocks drop` (`llb.prep.pdf.repeat_yield`).

Pure: fake stores exposing the `.retrieve` seam and hand-built gold items, plus one end-to-end
CLI test over the committed intra-document-repeats fixture with a straddling item. No FAISS, no
embedder, no GPU on the pure lane.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llb.core.contracts.rag import ChunkRecord
from llb.goldset.schema import GoldItem, SourceSpan
from llb.main import app
from llb.prep.pdf.repeat_yield import (
    VERDICT_DROPPED,
    VERDICT_HELD,
    VERDICT_LOST,
    VERDICT_RECOVERED,
    audit_repeat_yield,
    format_yield_report,
)

RUNNER = CliRunner()
FIXTURE = Path("samples/corpora/intra_document_repeats_uk_v1/corpus")
REPEATED_DOC = "nastanova-oblik.md"
PROCEDURE = "Порядок збереження документа"


class _StubStore:
    """Returns a fixed chunk list per question (the id it should hit, or nothing)."""

    def __init__(self, hits: dict[str, ChunkRecord]) -> None:
        self._hits = hits

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        hit = self._hits.get(question)
        return [hit] if hit is not None else []


def _item(item_id: str, start: int, end: int) -> GoldItem:
    return GoldItem(
        id=item_id,
        question=item_id,  # the question doubles as the stub-store key
        reference_answer="a",
        source_doc_id="d.md",
        source_spans=[
            SourceSpan(doc_id="d.md", char_start=start, char_end=end, text="x" * (end - start))
        ],
        provenance="human-authored",
        verified=True,
        split="final",
    )


def _chunk(start: int, end: int) -> ChunkRecord:
    return {"doc_id": "d.md", "char_start": start, "char_end": end, "text": "x" * (end - start)}


def test_held_recovered_lost_and_dropped_verdicts():
    baseline = [_item("held", 0, 5), _item("lost", 10, 15), _item("gone", 20, 25)]
    # the stripped set drops "gone" (straddled) and re-homes "held"+"lost" onto a survivor at 100
    stripped = [_item("held", 100, 105), _item("lost", 100, 105)]
    # baseline could answer all three (incl. the to-be-dropped "gone")
    baseline_store = _StubStore(
        {"held": _chunk(0, 5), "lost": _chunk(10, 15), "gone": _chunk(20, 25)}
    )
    stripped_store = _StubStore({"held": _chunk(100, 105)})  # "lost" no longer retrieved

    report = audit_repeat_yield(
        baseline,
        stripped,
        baseline_store,
        stripped_store,
        dropped_ids={"gone"},
        rehomed_ids={"held", "lost"},
        k=10,
    )

    verdicts = {entry["id"]: entry["verdict"] for entry in report["moved"]}
    assert verdicts == {"held": VERDICT_HELD, "lost": VERDICT_LOST, "gone": VERDICT_DROPPED}
    assert report["lost"] == ["lost", "gone"]  # lost re-home + dropped item the baseline could hit
    assert report["adopt"] is False
    assert report["baseline_recall"] == 1.0  # all three baseline items hit
    assert report["stripped_recall"] == 0.5  # 1 of the 2 scored (non-dropped) items hit


def test_adopt_when_every_touched_question_is_held_or_recovered():
    baseline = [_item("held", 0, 5), _item("recovered", 10, 15)]
    stripped = [_item("held", 100, 105), _item("recovered", 100, 105)]
    baseline_store = _StubStore({"held": _chunk(0, 5)})  # baseline misses "recovered"
    stripped_store = _StubStore({"held": _chunk(100, 105), "recovered": _chunk(100, 105)})

    report = audit_repeat_yield(
        baseline,
        stripped,
        baseline_store,
        stripped_store,
        dropped_ids=set(),
        rehomed_ids={"held", "recovered"},
        k=10,
    )

    verdicts = {entry["id"]: entry["verdict"] for entry in report["moved"]}
    assert verdicts == {"held": VERDICT_HELD, "recovered": VERDICT_RECOVERED}
    assert report["adopt"] is True
    assert report["lost"] == []


def test_dropped_item_the_baseline_missed_is_not_counted_as_lost():
    baseline = [_item("gone", 0, 5)]
    baseline_store = _StubStore({})  # baseline could not answer it anyway
    stripped_store = _StubStore({})

    report = audit_repeat_yield(
        baseline, [], baseline_store, stripped_store, dropped_ids={"gone"}, rehomed_ids=set(), k=10
    )

    assert report["lost"] == []
    assert report["adopt"] is True
    assert report["moved"][0]["verdict"] == VERDICT_DROPPED


def test_format_is_ascii_and_states_the_verdict():
    report = audit_repeat_yield(
        [_item("lost", 0, 5)],
        [_item("lost", 100, 105)],
        _StubStore({"lost": _chunk(0, 5)}),
        _StubStore({}),
        dropped_ids=set(),
        rehomed_ids={"lost"},
        k=10,
    )
    text = format_yield_report(report)
    assert text.isascii()
    assert "HOLD" in text and "lost" in text
    assert "pooled recall@10" in text


def test_cli_wires_strip_stores_and_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """End-to-end CLI over the fixture: real strip + remap classification, stubbed store build.

    The heavy embedder/FAISS build is the CUDA-run part; here it is replaced by a stub that hits
    every item, so the test exercises the strip -> remap -> audit wiring on a re-homed item.
    """
    import llb.rag.store as store_module

    text = (FIXTURE / REPEATED_DOC).read_text(encoding="utf-8")
    third = text.index(PROCEDURE, text.index(PROCEDURE, text.index(PROCEDURE) + 1) + 1)
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text(
        GoldItem(
            id="proc",
            question="Як зберегти документ?",
            reference_answer="Натисніть кнопку Зберегти.",
            source_doc_id=REPEATED_DOC,
            source_spans=[
                SourceSpan(
                    doc_id=REPEATED_DOC,
                    char_start=third,
                    char_end=third + len(PROCEDURE),
                    text=PROCEDURE,
                )
            ],
            provenance="human-authored",
            verified=True,
            split="final",
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "work"

    def fake_build(corpus_root, *args, **kwargs):
        # a store that always returns a chunk spanning the whole repeated doc, so it hits any span
        hit: ChunkRecord = {
            "doc_id": REPEATED_DOC,
            "char_start": 0,
            "char_end": 10_000,
            "text": "",
        }
        return _StubStore({"Як зберегти документ?": hit})

    monkeypatch.setattr(store_module.RagStore, "build", staticmethod(fake_build))

    result = RUNNER.invoke(
        app,
        [
            "audit-repeat-yield",
            "--corpus",
            str(FIXTURE),
            "--goldset",
            str(goldset),
            "--out",
            str(out),
            "--chunk-size",
            "200",
            "--chunk-overlap",
            "30",
            "--strategy",
            "sentence",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads((out / "repeat_yield.json").read_text(encoding="utf-8"))
    assert report["n"] == 1
    assert report["moved"][0]["change"] == "rehomed"
    assert report["moved"][0]["verdict"] == VERDICT_HELD
    assert (out / "drop-corpus" / REPEATED_DOC).is_file()
