"""Per-question yield audit for `--repeat-blocks drop` (`llb.prep.pdf.repeat_yield`).

Pure: fake stores exposing the `.retrieve` seam and hand-built gold items, plus one end-to-end
CLI test over the committed intra-document-repeats fixture with a straddling item. No FAISS, no
embedder, no GPU on the pure lane.
"""

import json


from pathlib import Path


import pytest


from llb.core.contracts.rag import ChunkRecord


from llb.goldset.schema import GoldItem, SourceSpan


from llb.main import app


from llb.prep.pdf.repeat_yield import (
    VERDICT_HELD,
)


from tests.llb.prep._repeat_yield_helpers import (
    RUNNER,
    FIXTURE,
    REPEATED_DOC,
    PROCEDURE,
    _StubStore,
)


def test_cli_wires_strip_stores_and_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """End-to-end CLI over the fixture: real strip + remap classification, stubbed store build.

    The heavy embedder/FAISS build is the CUDA-run part; here it is replaced by a stub that hits
    every item, so the test exercises the strip -> remap -> audit wiring on a re-homed item.
    """
    import llb.rag.vector_store.store as store_module

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


def test_cli_recover_straddle_keeps_a_boundary_crossing_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`--recover-straddle` turns a would-be dropped_from_set item into a scored one."""
    import llb.rag.vector_store.store as store_module

    text = (FIXTURE / REPEATED_DOC).read_text(encoding="utf-8")
    second = text.index(PROCEDURE, text.index(PROCEDURE) + 1)
    start, end = second - 40, second + len(PROCEDURE)  # kept tail + dropped procedure head
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text(
        GoldItem(
            id="straddle",
            question="Як зберегти документ?",
            reference_answer="Натисніть кнопку Зберегти.",
            source_doc_id=REPEATED_DOC,
            source_spans=[
                SourceSpan(
                    doc_id=REPEATED_DOC, char_start=start, char_end=end, text=text[start:end]
                )
            ],
            provenance="human-authored",
            verified=True,
            split="final",
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    def fake_build(corpus_root, *args, **kwargs):
        hit: ChunkRecord = {"doc_id": REPEATED_DOC, "char_start": 0, "char_end": 10_000, "text": ""}
        return _StubStore({"Як зберегти документ?": hit})

    monkeypatch.setattr(store_module.RagStore, "build", staticmethod(fake_build))

    def run(extra: list[str]) -> dict:
        out = tmp_path / ("recover" if extra else "drop")
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
                *extra,
            ],
        )
        assert result.exit_code == 0, result.output
        return json.loads((out / "repeat_yield.json").read_text(encoding="utf-8"))

    without = run([])
    recovered = run(["--recover-straddle"])

    assert without["dropped_from_set"] == 1 and without["kept"] == 0
    assert recovered["dropped_from_set"] == 0 and recovered["kept"] == 1
    assert recovered["moved"][0]["verdict"] == VERDICT_HELD  # the recovered span is retrieved
