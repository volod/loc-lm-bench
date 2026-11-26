"""The stage that merges one page's recognition readings into the block ledger."""

from collections.abc import Mapping, Sequence
from typing import Any

from llb.prep.book.constants import BLOCKS_FILENAME, READING_FILENAME
from llb.prep.book.engines.image import png_size
from llb.prep.book.engines.protocol import EngineRead, read_reading
from llb.prep.book.ledger.assemble import assemble
from llb.prep.book.ledger.io import write_ledger
from llb.prep.book.stages import StageContext, StageResult
from llb.prep.book.stages.recognize import (
    BLOCKS_STAGE,
    BLOCKS_TOOL_IDENTITY,
    RECOGNITION_STAGES,
)
from llb.prep.book.stages.render import PAGE_IMAGE_FILENAME, RENDER_STAGE

_STAGE_VERSION = "4"


class BlocksStage:
    """Merge the configured recognition stages of one page into `blocks.json`.

    The ledger records the page id and the block ids, so the cache key includes the PDF page.
    Identical page images do not share a ledger.
    """

    name = BLOCKS_STAGE
    version = _STAGE_VERSION
    inputs: tuple[str, ...] = ()
    reads_source = True
    model_required = False

    def tool_identity(self) -> str:
        return BLOCKS_TOOL_IDENTITY

    def inputs_for(
        self, configured_before: Sequence[str], params: Mapping[str, Any] | None = None
    ) -> tuple[str, ...]:
        del params
        selected = tuple(name for name in RECOGNITION_STAGES if name in configured_before)
        return (RENDER_STAGE, *selected)

    def validate_params(self, params: Mapping[str, Any]) -> list[str]:
        return [f"blocks: unknown parameter {key!r}" for key in params]

    def validate_configured(self, configured_before: Sequence[str]) -> list[str]:
        if any(name in configured_before for name in RECOGNITION_STAGES):
            return []
        return ["stage 'blocks' needs a layout, ocr, formula, or table stage before it"]

    def effective_params(
        self, params: Mapping[str, Any], inventory: Any, document: Any, pdf_page: int
    ) -> dict[str, Any]:
        del inventory, document, pdf_page
        return dict(params)

    def run(self, ctx: StageContext) -> StageResult:
        reads = _readings(ctx)
        width, height = png_size(ctx.inputs[RENDER_STAGE] / PAGE_IMAGE_FILENAME)
        ledger = assemble(ctx.page_id, ctx.pdf_page, width, height, reads)
        write_ledger(ctx.work_dir / BLOCKS_FILENAME, ledger)
        spans = sum(len(block.uncertainty) for block in ledger.blocks)
        return StageResult(meta={"blocks": len(ledger.blocks), "uncertainty_spans": spans})


def _readings(ctx: StageContext) -> list[EngineRead]:
    reads = []
    for name, directory in ctx.inputs.items():
        if name == RENDER_STAGE:
            continue
        path = directory / READING_FILENAME
        if not path.is_file():
            raise RuntimeError(f"stage {name} has no {READING_FILENAME}")
        reads.append(read_reading(path))
    if not reads:
        raise RuntimeError("blocks stage has no recognition input")
    return reads
