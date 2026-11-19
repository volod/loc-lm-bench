"""Deterministic stand-in stages that exercise the ledger without any recognition engine.

They are registered so an operator can drive a whole book through the run, kill it, resume it, and
reuse it before any real preprocessing or OCR exists. Their output is a pure function of the page
image digest and their parameters, so a rerun is byte-identical. `fail_pages` forces a failure on
the named PDF pages, which is how the failure-isolation path is shown on a real book.
"""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from llb.prep.book.stages import StageContext, StageResult
from llb.prep.book.stages.render import PAGE_IMAGE_FILENAME, RENDER_STAGE

FAKE_TEXT_FILENAME = "text.txt"
FAKE_VARIANT_FILENAME = "variant.json"


class ForcedStageFailure(RuntimeError):
    """A failure the operator asked for with `fail_pages`."""


def _image_digest(ctx: StageContext) -> str:
    return hashlib.sha256((ctx.inputs[RENDER_STAGE] / PAGE_IMAGE_FILENAME).read_bytes()).hexdigest()


def _validate(name: str, params: Mapping[str, Any]) -> list[str]:
    errors = [f"{name}: unknown parameter {key!r}" for key in params if key not in _FAKE_PARAMS]
    fail_pages = params.get("fail_pages", [])
    if not isinstance(fail_pages, list) or not all(
        isinstance(page, int) and not isinstance(page, bool) for page in fail_pages
    ):
        errors.append(f"{name}: fail_pages must be a list of PDF page numbers")
    if not isinstance(params.get("note", ""), str):
        errors.append(f"{name}: note must be a string")
    return errors


def _maybe_fail(name: str, ctx: StageContext) -> None:
    if ctx.pdf_page in ctx.params.get("fail_pages", []):
        raise ForcedStageFailure(f"{name}: forced failure on PDF page {ctx.pdf_page}")


class FakePreprocessStage:
    """Records the variant it WOULD produce: the original image, untouched."""

    name = "fake-preprocess"
    version = "1"
    inputs: tuple[str, ...] = (RENDER_STAGE,)
    reads_source = False
    model_required = False

    def tool_identity(self) -> str:
        return "llb-fake 1"

    def validate_params(self, params: Mapping[str, Any]) -> list[str]:
        return _validate(self.name, params)

    def effective_params(
        self, params: Mapping[str, Any], inventory: Any, document: Any, pdf_page: int
    ) -> dict[str, Any]:
        return dict(params)  # nothing to resolve: every parameter is explicit

    def run(self, ctx: StageContext) -> StageResult:
        _maybe_fail(self.name, ctx)
        record = {"variant": "original", "image_sha256": _image_digest(ctx)}
        (ctx.work_dir / FAKE_VARIANT_FILENAME).write_text(
            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
        )
        return StageResult(meta={"variants": 1})


class FakeOcrStage:
    """Writes a placeholder transcription naming the page and the image it would have read."""

    name = "fake-ocr"
    version = "1"
    inputs: tuple[str, ...] = (RENDER_STAGE,)
    reads_source = False
    model_required = False

    def tool_identity(self) -> str:
        return "llb-fake 1"

    def validate_params(self, params: Mapping[str, Any]) -> list[str]:
        return _validate(self.name, params)

    def effective_params(
        self, params: Mapping[str, Any], inventory: Any, document: Any, pdf_page: int
    ) -> dict[str, Any]:
        return dict(params)  # nothing to resolve: every parameter is explicit

    def run(self, ctx: StageContext) -> StageResult:
        _maybe_fail(self.name, ctx)
        note = ctx.params.get("note", "")
        text = (
            f"[fake-ocr] {ctx.page_id} pdf-page={ctx.pdf_page} "
            f"image={_image_digest(ctx)[:16]} model={ctx.model or '-'} {note}".rstrip()
        )
        (ctx.work_dir / FAKE_TEXT_FILENAME).write_text(text + "\n", encoding="utf-8")
        return StageResult(meta={"chars": len(text)})


_FAKE_PARAMS = ("fail_pages", "note")
FAKE_STAGES = (FakePreprocessStage(), FakeOcrStage())
