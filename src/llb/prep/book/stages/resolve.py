"""The `resolve` stage: one resolved reading per block, written beside the raw ledger.

The stage reads the block ledger, model edits when a `correct` stage ran, and reviewer decisions
from `review/decisions.json`. It never writes the ledger. Agreement edits are produced here;
model and reviewer edits are only applied.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llb.prep.book.constants import BLOCKS_FILENAME
from llb.prep.book.ledger.io import read_ledger
from llb.prep.book.resolve.apply import resolve_page
from llb.prep.book.resolve.config import (
    order_errors,
    page_choice,
    resolve_inputs,
    validate_resolve_params,
)
from llb.prep.book.resolve.constants import (
    CORRECT_STAGE,
    OVERLAY_FILENAME,
    RESOLVED_FILENAME,
    RESOLVE_STAGE,
    RESOLVE_TOOL,
    RESOLVE_VERSION,
    REVIEWER_INPUT,
)
from llb.prep.book.resolve.io import (
    load_model_corrections,
    load_reviewer_edits,
    reviewer_digest,
    write_overlay,
    write_resolved,
)
from llb.prep.book.resolve.primary import choice_from_params
from llb.prep.book.stages import StageContext, StageResult
from llb.prep.book.stages.recognize import BLOCKS_STAGE


class ResolveStage:
    """Choose each block's reading and apply the correction overlay."""

    name = RESOLVE_STAGE
    version = RESOLVE_VERSION
    inputs: tuple[str, ...] = ()
    reads_source = False
    model_required = False

    def tool_identity(self) -> str:
        return RESOLVE_TOOL

    def inputs_for(
        self, configured_before: Sequence[str], params: Mapping[str, Any] | None = None
    ) -> tuple[str, ...]:
        return resolve_inputs(configured_before, params)

    def validate_params(self, params: Mapping[str, Any]) -> list[str]:
        return validate_resolve_params(params)

    def validate_order(
        self, params: Mapping[str, Any], configured_before: Sequence[str]
    ) -> list[str]:
        return order_errors(params, configured_before)

    def effective_params(
        self, params: Mapping[str, Any], inventory: Any, document: Any, pdf_page: int
    ) -> dict[str, Any]:
        del inventory, document, pdf_page
        return page_choice(params, {})

    def page_params(
        self,
        params: Mapping[str, Any],
        inputs: Mapping[str, Path],
        inventory: Any,
        document: Any,
        pdf_page: int,
    ) -> dict[str, Any]:
        del inventory, document, pdf_page
        return page_choice(params, inputs)

    def extra_input_digests(self, run_dir: Path, page_id: str) -> dict[str, str]:
        """The reviewer file, sliced to this page, so a decision elsewhere is a cache hit."""
        return {REVIEWER_INPUT: reviewer_digest(run_dir, page_id)}

    def run(self, ctx: StageContext) -> StageResult:
        ledger_path = ctx.inputs[BLOCKS_STAGE] / BLOCKS_FILENAME
        before = ledger_path.read_bytes()
        ledger = read_ledger(ledger_path)
        edits = load_reviewer_edits(ctx.run_dir, ctx.page_id)
        corrections = load_model_corrections(ctx.inputs.get(CORRECT_STAGE), ctx.page_id)
        if corrections is not None:
            edits = [*corrections.edits, *edits]
        overlay, resolved = resolve_page(
            ledger,
            choice_from_params(ctx.params),
            edits,
            () if corrections is None else corrections.refused,
        )
        write_overlay(ctx.work_dir / OVERLAY_FILENAME, overlay)
        write_resolved(ctx.work_dir / RESOLVED_FILENAME, resolved)
        if ledger_path.read_bytes() != before:
            raise RuntimeError("resolve changed the block ledger")
        unresolved = sum(
            1
            for block in resolved.blocks
            for span in block.spans
            if span.provenance == "unresolved"
        )
        return StageResult(
            meta={
                "blocks": len(resolved.blocks),
                "applied": len(overlay.applied),
                "orphaned": len(overlay.orphaned),
                "unresolved_spans": unresolved,
                "fallbacks": sum(1 for block in resolved.blocks if block.fallback),
            }
        )
