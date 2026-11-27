"""The correct stage inside a book-ocr run, and the per-block proposal cache."""

from pathlib import Path

import numpy as np

from llb.core.contracts.data_prep.book import BookStageSpec
from llb.prep.book.cache import page_dir
from llb.prep.book.config import RunRequest
from llb.prep.book.constants import BLOCKS_FILENAME
from llb.prep.book.correct.page import correct_page
from llb.prep.book.correct.proposal import BlockPrompt, Proposal, ProposalResult
from llb.prep.book.engines.fake import FakeEngine
from llb.prep.book.engines.protocol import EngineRead, Proposed
from llb.prep.book.engines.registry import AdapterRegistry
from llb.prep.book.ledger.assemble import assemble
from llb.prep.book.ledger.io import read_ledger, write_ledger
from llb.prep.book.preprocess.images import write_png
from llb.prep.book.project.text import corpus_passage
from llb.prep.book.render.prose import passage
from llb.prep.book.render.types import RenderBlock
from llb.prep.book.resolve.constants import CORRECTIONS_FILENAME
from llb.prep.book.resolve.io import read_corrections, read_overlay, read_resolved
from llb.prep.book.review.classify import page_marks
from llb.prep.book.runner import check_request, resume_run, start_run
from llb.prep.book.stages import StageContext, StageRegistry
from llb.prep.book.stages.correct import CorrectStage
from llb.prep.book.stages.recognize import recognition_stages
from llb.prep.book.stages.render import PAGE_IMAGE_FILENAME, RenderStage
from llb.prep.book.stages.resolve import ResolveStage

_PRIMARY = {"primary": {"stage": "ocr", "engine": "fake", "variant": "original"}}
_BODY = "The value is 0.05."
_MODEL = "fake:scripted"


def _proposal(
    page_id: str, start: int, end: int, replacement: str, confusion_class: str, visual: bool
):
    return {
        "page_id": page_id,
        "text": _BODY,
        "start": start,
        "end": end,
        "replacement": replacement,
        "class": confusion_class,
        "evidence": "crop shows the glyph" if visual else "crop does not show it",
        "visual": visual,
    }


def _stages() -> tuple[BookStageSpec, ...]:
    proposals = [
        _proposal("p0001", 13, 14, "O", "digit", False),
        _proposal("p0001", 16, 17, "S", "digit", True),
        _proposal("p0002", 0, len(_BODY), "A rewritten sentence.", "prose", True),
    ]
    return (
        BookStageSpec(name="render"),
        BookStageSpec(name="ocr", params={"engine": "fake"}),
        BookStageSpec(name="blocks"),
        BookStageSpec(name="correct", model=_MODEL, params={**_PRIMARY, "proposals": proposals}),
        BookStageSpec(name="resolve", params=_PRIMARY),
    )


def _registry() -> StageRegistry:
    engines = recognition_stages(AdapterRegistry((FakeEngine(),)))
    return StageRegistry((RenderStage(), *engines, CorrectStage(), ResolveStage()))


def test_admitted_edits_reach_both_renderings_and_refusals_become_markers(
    scan_pdf: Path, out_root: Path
):
    run_dir = _run_pages(scan_pdf, out_root)
    _assert_ledger_unchanged(run_dir)
    _assert_admitted_page(run_dir)
    _assert_refused_page(run_dir)


def _run_pages(scan_pdf: Path, out_root: Path) -> Path:
    outcome = start_run(
        RunRequest(source=scan_pdf, pages="1-2", batch_pages=2, stages=_stages()),
        out_root,
        _registry(),
    )
    before = _ledger_bytes(outcome.run_dir)
    resumed = resume_run(outcome.run_dir, registry=_registry())
    assert resumed.outcomes.get("ran", 0) == 0
    assert _ledger_bytes(outcome.run_dir) == before
    return outcome.run_dir


def _ledger_bytes(run_dir: Path) -> dict[str, bytes]:
    paths = (run_dir / "pages" / pid / BLOCKS_FILENAME for pid in ("p0001", "p0002"))
    return {path.parent.name: path.read_bytes() for path in paths}


def _assert_ledger_unchanged(run_dir: Path) -> None:
    for pid in ("p0001", "p0002"):
        ledger = read_ledger(run_dir / "pages" / pid / BLOCKS_FILENAME)
        assert any(reading.text == _BODY for block in ledger.blocks for reading in block.readings)


def _assert_admitted_page(run_dir: Path) -> None:
    admitted = _page(run_dir, "p0001")
    body = next(block for block in admitted.resolved.blocks if block.primary_text == _BODY)
    assert "S" in body.text
    assert "S" in admitted.vault
    assert "S" in admitted.corpus
    assert "O" not in body.text
    assert "[!low-confidence]" in admitted.vault
    assert "llb-marker:unresolved:" in admitted.corpus
    edit = next(item for item in admitted.overlay.applied if item.author == "model")
    assert edit.raw_text == "5"
    assert edit.replacement == "S"
    assert edit.confusion_class == "digit"
    assert edit.evidence == "crop shows the glyph"
    assert edit.source == _MODEL
    stored = read_corrections(run_dir / "pages" / "p0001" / CORRECTIONS_FILENAME)
    assert stored.model == _MODEL
    assert any(item.reason == "no-visual" for item in stored.refused)


def _assert_refused_page(run_dir: Path) -> None:
    refused = _page(run_dir, "p0002")
    assert "rewritten" not in refused.vault
    assert "rewritten" not in refused.corpus
    assert "[!low-confidence]" in refused.vault
    assert "llb-marker:unresolved:" in refused.corpus
    assert all(item.author != "model" for item in refused.overlay.applied)
    page_file = read_corrections(run_dir / "pages" / "p0002" / CORRECTIONS_FILENAME)
    assert page_file.refused
    assert page_file.refused[0].reason == "rewrite"


def test_a_frontier_model_is_refused_before_a_page_runs(scan_pdf: Path):
    stages = (
        BookStageSpec(name="render"),
        BookStageSpec(name="ocr", params={"engine": "fake"}),
        BookStageSpec(name="blocks"),
        BookStageSpec(name="correct", model="openai:gpt-4o", params=_PRIMARY),
    )
    plan = check_request(RunRequest(source=scan_pdf, pages="1", stages=stages), _registry())
    assert not plan.ok
    assert any("unsupported backend" in error for error in plan.errors)


class _Counting:
    def __init__(self) -> None:
        self.calls = 0
        self.crops: list[int] = []

    def propose(self, prompt: BlockPrompt) -> ProposalResult:
        self.calls += 1
        self.crops.append(len(prompt.crop_png))
        if prompt.text != "alpha":
            return ProposalResult()
        return ProposalResult(
            (
                Proposal(
                    0,
                    1,
                    "A",
                    "latin-greek",
                    "crop shows A",
                    True,
                ),
            )
        )


def test_a_block_is_not_sent_to_the_model_twice(tmp_path: Path):
    ledger = assemble(
        "p0001",
        1,
        200,
        80,
        (
            _read("alpha", 0.0, 20.0, 0),
            _read("beta", 30.0, 50.0, 1),
        ),
    )
    run_dir = tmp_path / "run"
    work_1 = tmp_path / "work-1"
    work_2 = tmp_path / "work-2"
    inputs = _inputs(tmp_path, ledger)
    counter = _Counting()
    ctx = _context(run_dir, work_1, inputs)
    first = correct_page(ctx, counter)
    assert counter.calls == 2
    assert first.meta["admitted"] == 0
    assert all(size > 0 for size in counter.crops)
    cache = work_1 / "cache"
    stored = page_dir(run_dir, "p0001") / "correct" / "cache"
    stored.parent.mkdir(parents=True)
    stored.symlink_to(cache, target_is_directory=True)
    second = correct_page(_context(run_dir, work_2, inputs), counter)
    assert counter.calls == 2
    assert second.meta["cached"] == 2


def _page(run_dir: Path, pid: str):
    base = run_dir / "pages" / pid
    ledger = read_ledger(base / BLOCKS_FILENAME)
    resolved = read_resolved(base / "resolved.json")
    overlay = read_overlay(base / "overlay.json")
    blocks = {block.block_id: block for block in ledger.blocks}
    body = next(block for block in resolved.blocks if block.primary_text == _BODY)
    source = blocks[body.block_id]
    marks = tuple(
        mark
        for mark in page_marks(pid, blocks, resolved, overlay, [])
        if mark.block_id == body.block_id
    )
    rendered = RenderBlock(
        pdf_page=source.pdf_page,
        page_id=pid,
        ordinal=source.ordinal,
        reading_order=source.reading_order,
        kind=source.kind,
        role="paragraph",
        block_id=source.block_id,
        digest=body.anchor,
        text=body.text,
        region=source.region,
        resolved=body,
        marks=marks,
    )
    return _View(
        vault=passage(rendered, lambda page: f"source/book.pdf#page={page}"),
        corpus=corpus_passage(rendered),
        resolved=resolved,
        overlay=overlay,
    )


class _View:
    def __init__(self, vault: str, corpus: str, resolved, overlay) -> None:
        self.vault = vault
        self.corpus = corpus
        self.resolved = resolved
        self.overlay = overlay


def _read(text: str, y0: float, y1: float, order: int) -> EngineRead:
    block = Proposed("body", order, 0.0, y0, 100.0, y1, text, None)
    return EngineRead("fake", "llb-fake-1", "original", "ocr", (block,))


def _inputs(tmp_path: Path, ledger) -> dict[str, Path]:
    blocks = tmp_path / "blocks"
    render = tmp_path / "render"
    blocks.mkdir()
    render.mkdir()
    write_ledger(blocks / BLOCKS_FILENAME, ledger)
    write_png(render / PAGE_IMAGE_FILENAME, np.full((80, 200), 240, dtype=np.uint8))
    return {"blocks": blocks, "render": render}


def _context(run_dir: Path, work: Path, inputs: dict[str, Path]) -> StageContext:
    work.mkdir()
    return StageContext(
        page_id="p0001",
        pdf_page=1,
        inventory=None,  # type: ignore[arg-type]
        params={
            "stage": "ocr",
            "engine": "fake",
            "variant": "original",
            "model": None,
            "proposals": [],
        },
        model="fake:scripted",
        inputs=inputs,
        work_dir=work,
        run_dir=run_dir,
        document=_Doc(),
    )


class _Doc:
    page_count = 1
