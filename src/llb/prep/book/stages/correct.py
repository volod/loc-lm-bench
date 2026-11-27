"""The `correct` stage: per-block proposals, admitted or refused by the guard.

The stage is optional and is not in the default stage list. A `fake:` model reads scripted
proposals from its parameters. An `ollama:` model reads the block crop. Any other backend is
refused before a page runs. The raw ledger is not written.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llb.prep.book.correct.constants import CORRECT_VERSION, guard_identity
from llb.prep.book.correct.edges import neighbour_digest
from llb.prep.book.correct.ollama import OllamaProposer
from llb.prep.book.correct.page import correct_page
from llb.prep.book.correct.params import (
    flat_params,
    order_errors,
    validate_correct_model,
    validate_correct_params,
)
from llb.prep.book.correct.proposal import Proposer
from llb.prep.book.resolve.constants import CORRECT_STAGE
from llb.prep.book.stages import StageContext, StageResult
from llb.prep.book.stages.render import RENDER_STAGE
from llb.prep.book.stages.recognize import BLOCKS_STAGE


class CorrectStage:
    """Propose and guard per-block edits. Refusals are written beside the admitted ones."""

    name = CORRECT_STAGE
    version = CORRECT_VERSION
    inputs: tuple[str, ...] = ()
    reads_source = False
    model_required = True

    def __init__(self, proposer: Proposer | None = None) -> None:
        self._proposer = proposer
        self._owned: OllamaProposer | None = None
        self._owned_model = ""

    def tool_identity(self) -> str:
        return guard_identity()

    def inputs_for(
        self, configured_before: Sequence[str], params: Mapping[str, Any] | None = None
    ) -> tuple[str, ...]:
        del configured_before, params
        return (RENDER_STAGE, BLOCKS_STAGE)

    def validate_params(self, params: Mapping[str, Any]) -> list[str]:
        return validate_correct_params(params)

    def validate_model(self, model: str | None, params: Mapping[str, Any]) -> list[str]:
        return validate_correct_model(model, params)

    def validate_order(
        self, params: Mapping[str, Any], configured_before: Sequence[str]
    ) -> list[str]:
        return order_errors(params, configured_before)

    def effective_params(
        self, params: Mapping[str, Any], inventory: Any, document: Any, pdf_page: int
    ) -> dict[str, Any]:
        del inventory, document, pdf_page
        return flat_params(params)

    def extra_input_digests(self, run_dir: Path, page_id: str) -> dict[str, str]:
        """Facing blocks on the neighbour pages, so an edge change re-runs this page only."""
        return neighbour_digest(run_dir, page_id)

    def close_batch(self) -> None:
        """Unload an Ollama model this batch loaded. A fake proposer holds nothing."""
        if self._owned is not None:
            self._owned.unload()
            self._owned = None
            self._owned_model = ""

    def run(self, ctx: StageContext) -> StageResult:
        return correct_page(ctx, self._proposer_for(ctx))

    def _proposer_for(self, ctx: StageContext) -> Proposer | None:
        if self._proposer is not None:
            return self._proposer
        model = ctx.model or ""
        if not model.startswith("ollama:"):
            return None
        _backend, _sep, name = model.partition(":")
        if self._owned is None or self._owned_model != name:
            self.close_batch()
            self._owned = OllamaProposer(name)
            self._owned_model = name
        return self._owned
