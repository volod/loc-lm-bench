"""Resolve inside a book-ocr run: cache keys, one page, and an untouched ledger."""

import shutil
from pathlib import Path

from llb.core.contracts.data_prep.book import BookStageSpec
from llb.core.contracts.data_prep.book_resolve import (
    CorrectionEdit,
    ModelCorrections,
    ReviewerDecisions,
)
from llb.prep.book.cache import stage_dir
from llb.prep.book.config import RunRequest
from llb.prep.book.engines.fake import FakeEngine
from llb.prep.book.engines.registry import AdapterRegistry
from llb.prep.book.ledger.io import read_ledger
from llb.prep.book.resolve.anchor import block_anchor
from llb.prep.book.resolve.constants import CORRECTIONS_FILENAME
from llb.prep.book.resolve.io import read_resolved, write_corrections, write_decisions
from llb.prep.book.runner import check_request, resume_run, start_run
from llb.prep.book.stages import StageContext, StageRegistry, StageResult
from llb.prep.book.stages.recognize import recognition_stages
from llb.prep.book.stages.render import RenderStage
from llb.prep.book.stages.resolve import ResolveStage

_PRIMARY = {"primary": {"stage": "ocr", "engine": "fake", "variant": "original"}}


def _registry() -> StageRegistry:
    engines = recognition_stages(AdapterRegistry((FakeEngine(),)))
    return StageRegistry((RenderStage(), *engines, ResolveStage()))


def _run(scan: Path, out: Path, stages: tuple[BookStageSpec, ...], registry: StageRegistry):
    request = RunRequest(source=scan, pages="1-2", batch_pages=2, stages=stages)
    return start_run(request, out, registry)


def _decision(run_dir: Path) -> CorrectionEdit:
    block = read_ledger(run_dir / "pages" / "p0001" / "blocks.json").blocks[0]
    text = block.readings[0].text
    return CorrectionEdit(
        edit_id="rev-1",
        page_id="p0001",
        block_id=block.block_id,
        anchor=block_anchor(block),
        author="reviewer",
        start=0,
        end=len(text),
        raw_text=text,
        replacement="REVIEWED",
        evidence="reviewer read the crop",
        source="reviewer",
    )


def test_a_new_reviewer_decision_changes_its_page_alone(scan_pdf: Path, out_root: Path):
    stages = (
        BookStageSpec(name="render"),
        BookStageSpec(name="ocr", params={"engine": "fake"}),
        BookStageSpec(name="blocks"),
        BookStageSpec(name="resolve", params=_PRIMARY),
    )
    outcome = _run(scan_pdf, out_root, stages, _registry())
    run_dir = outcome.run_dir
    ledgers = {
        pid: (run_dir / "pages" / pid / "blocks.json").read_bytes() for pid in ("p0001", "p0002")
    }
    untouched = (run_dir / "pages" / "p0002" / "resolved.json").read_bytes()
    write_decisions(
        run_dir / "review" / "decisions.json",
        ReviewerDecisions(
            schema_id="llb.book-ocr-reviewer-decisions",
            schema_version="1.0.0",
            kind="book-ocr-reviewer-decisions",
            edits=[_decision(run_dir)],
        ),
    )
    resumed = resume_run(run_dir, registry=_registry())
    assert resumed.outcomes == {"ran": 1, "cached": 7}
    resolved = read_resolved(run_dir / "pages" / "p0001" / "resolved.json")
    assert resolved.blocks[0].text == "REVIEWED"
    assert (run_dir / "pages" / "p0002" / "resolved.json").read_bytes() == untouched
    for pid, raw in ledgers.items():
        assert (run_dir / "pages" / pid / "blocks.json").read_bytes() == raw


class _Correct:
    """Writes one model edit. The replacement is mutable so a deleted stage can write a new one."""

    name = "correct"
    version = "1"
    inputs = ("blocks",)
    reads_source = False
    model_required = False

    def __init__(self) -> None:
        self.replacement = "NINE"

    def tool_identity(self) -> str:
        return "fake-correct 1"

    def validate_params(self, params: object) -> list[str]:
        del params
        return []

    def effective_params(
        self, params: object, inventory: object, document: object, pdf_page: int
    ) -> dict[str, str]:
        del params, inventory, document, pdf_page
        return {}

    def run(self, ctx: StageContext) -> StageResult:
        ledger = read_ledger(ctx.inputs["blocks"] / "blocks.json")
        block = ledger.blocks[0]
        text = block.readings[0].text
        edit = CorrectionEdit(
            edit_id=f"model-{ctx.page_id}",
            page_id=ctx.page_id,
            block_id=block.block_id,
            anchor=block_anchor(block),
            author="model",
            start=0,
            end=len(text),
            raw_text=text,
            replacement=self.replacement,
            evidence="the crop shows the replacement",
            source="fake-correct",
        )
        write_corrections(
            ctx.work_dir / CORRECTIONS_FILENAME,
            ModelCorrections(
                schema_id="llb.book-ocr-model-corrections",
                schema_version="1.0.0",
                kind="book-ocr-model-corrections",
                page_id=ctx.page_id,
                pdf_page=ctx.pdf_page,
                edits=[edit],
            ),
        )
        return StageResult(meta={"edits": 1})


def test_a_model_edit_re_runs_only_the_page_whose_stage_output_changed(
    scan_pdf: Path, out_root: Path
):
    correct = _Correct()
    engines = recognition_stages(AdapterRegistry((FakeEngine(),)))
    registry = StageRegistry((RenderStage(), *engines, correct, ResolveStage()))
    stages = (
        BookStageSpec(name="render"),
        BookStageSpec(name="ocr", params={"engine": "fake"}),
        BookStageSpec(name="blocks"),
        BookStageSpec(name="correct"),
        BookStageSpec(name="resolve", params=_PRIMARY),
    )
    outcome = _run(scan_pdf, out_root, stages, registry)
    run_dir = outcome.run_dir
    assert read_resolved(run_dir / "pages" / "p0001" / "resolved.json").blocks[0].text == "NINE"
    ledgers = {
        pid: (run_dir / "pages" / pid / "blocks.json").read_bytes() for pid in ("p0001", "p0002")
    }
    correct.replacement = "EIGHT"
    shutil.rmtree(stage_dir(run_dir, "p0001", "correct"))
    resumed = resume_run(run_dir, registry=registry)
    assert resumed.outcomes == {"ran": 2, "cached": 8}
    assert read_resolved(run_dir / "pages" / "p0001" / "resolved.json").blocks[0].text == "EIGHT"
    assert read_resolved(run_dir / "pages" / "p0002" / "resolved.json").blocks[0].text == "NINE"
    for pid, raw in ledgers.items():
        assert (run_dir / "pages" / pid / "blocks.json").read_bytes() == raw


def test_resolve_is_refused_without_a_primary_or_the_ledger(scan_pdf: Path):
    registry = _registry()
    bare = check_request(
        RunRequest(source=scan_pdf, pages="1", stages=(BookStageSpec(name="resolve"),)),
        registry,
    )
    assert any("primary" in error for error in bare.errors)
    assert any("blocks" in error for error in bare.errors)
