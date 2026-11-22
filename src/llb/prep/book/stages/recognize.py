"""Layout, OCR, formula, and table stages, and the stage that merges them into the ledger.

An engine stage's cache key names the engine that actually runs on the page, after `by_page_class`
is resolved. The blocks stage's inputs are whichever of those stages the run configured.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from llb.prep.book.constants import BLOCKS_FILENAME, READING_FILENAME
from llb.prep.book.engines.image import png_size
from llb.prep.book.engines.lease import EngineLease
from llb.prep.book.engines.pdf_page import write_page_pdf
from llb.prep.book.engines.protocol import (
    EngineRead,
    EngineUnavailable,
    read_reading,
    write_reading,
)
from llb.prep.book.engines.registry import AdapterRegistry
from llb.prep.book.ledger.assemble import assemble
from llb.prep.book.ledger.io import write_ledger
from llb.prep.book.preprocess.constants import ORIGINAL, VARIANTS_STAGE, VARIANT_NAMES
from llb.prep.book.stages import StageContext, StageResult
from llb.prep.book.stages.render import PAGE_IMAGE_FILENAME, RENDER_STAGE

LAYOUT_STAGE = "layout"
OCR_STAGE = "ocr"
FORMULA_STAGE = "formula"
TABLE_STAGE = "table"
BLOCKS_STAGE = "blocks"
RECOGNITION_STAGES = (LAYOUT_STAGE, OCR_STAGE, FORMULA_STAGE, TABLE_STAGE)
TOOL_IDENTITY = "llb-book-engines 1"
BLOCKS_TOOL_IDENTITY = "llb-book-blocks 1"
_STAGE_VERSION = "1"
_PAGE_PDF = "page.pdf"
_ENGINE_PARAMS = ("engine", "variant", "page_class", "by_page_class")
_DEFAULT_VARIANT = ORIGINAL


class EngineStage:
    """One recognition stage. The engine is a parameter, so the same stage swaps engines."""

    version = _STAGE_VERSION
    inputs: tuple[str, ...] = (RENDER_STAGE,)
    reads_source = False
    model_required = False

    def __init__(self, name: str, adapters: AdapterRegistry) -> None:
        self.name = name
        self._adapters = adapters
        self._lease = EngineLease()

    def tool_identity(self) -> str:
        return TOOL_IDENTITY

    def open_batch(self) -> None:
        """Drop the previous batch's server so this batch may start it again."""
        self._lease.reset()

    def close_batch(self) -> None:
        """Stop the server this batch opened. The next stage does not inherit it."""
        self._lease.close()

    def validate_params(self, params: Mapping[str, Any]) -> list[str]:
        errors = [
            f"{self.name}: unknown parameter {key!r}" for key in params if key not in _ENGINE_PARAMS
        ]
        errors.extend(self._check_engine(params.get("engine"), "engine"))
        errors.extend(_check_variant(self.name, params.get("variant", _DEFAULT_VARIANT)))
        errors.extend(_check_page_class(self.name, params.get("page_class", "")))
        errors.extend(self._check_map(params.get("by_page_class", {})))
        return errors

    def effective_params(
        self, params: Mapping[str, Any], inventory: Any, document: Any, pdf_page: int
    ) -> dict[str, Any]:
        return {"engine": selected_engine(params), "variant": _variant(params)}

    def inputs_for(self, configured_before: Sequence[str]) -> tuple[str, ...]:
        if VARIANTS_STAGE in configured_before:
            return (RENDER_STAGE, VARIANTS_STAGE)
        return (RENDER_STAGE,)

    def validate_order(
        self, params: Mapping[str, Any], configured_before: Sequence[str]
    ) -> list[str]:
        name = _variant(params)
        if name == _DEFAULT_VARIANT:
            return []
        if name not in VARIANT_NAMES:
            return [f"{self.name}: variant {name!r} is not a known variant"]
        if VARIANTS_STAGE not in configured_before:
            return [f"{self.name}: variant {name!r} needs the variants stage before it"]
        return []

    def run(self, ctx: StageContext) -> StageResult:
        adapter = self._adapter(ctx)
        self._lease.use(adapter)
        pdf = ctx.work_dir / _PAGE_PDF
        write_page_pdf(ctx.document, ctx.pdf_page, pdf)
        try:
            reading = _read(self.name, ctx, adapter, pdf)
        finally:
            pdf.unlink(missing_ok=True)
        write_reading(ctx.work_dir, reading)
        return StageResult(meta=_meta(reading))

    def _adapter(self, ctx: StageContext) -> Any:
        engine_id = str(ctx.params["engine"])
        adapter = self._adapters.require(engine_id)
        if self.name not in adapter.stages:
            raise EngineUnavailable(engine_id, f"does not serve stage {self.name}")
        return adapter

    def _check_engine(self, engine: object, where: str) -> list[str]:
        if not isinstance(engine, str) or not engine:
            return [f"{self.name}: {where} must name an engine"]
        try:
            adapter = self._adapters.require(engine)
        except EngineUnavailable as exc:
            return [_named(exc)]
        if self.name not in adapter.stages:
            return [f"{self.name}: engine {engine} does not serve this stage"]
        try:
            adapter.available()
        except EngineUnavailable as exc:
            return [_named(exc)]
        return []

    def _check_map(self, raw: object) -> list[str]:
        if not isinstance(raw, dict):
            return [f"{self.name}: by_page_class must be a mapping of page class to engine"]
        errors: list[str] = []
        for key, engine in raw.items():
            if not isinstance(key, str):
                errors.append(f"{self.name}: by_page_class keys must be strings")
                continue
            errors.extend(self._check_engine(engine, f"by_page_class[{key}]"))
        return errors


class BlocksStage:
    """Merge the configured recognition stages of one page into `blocks.json`."""

    name = BLOCKS_STAGE
    version = _STAGE_VERSION
    inputs: tuple[str, ...] = ()
    reads_source = False
    model_required = False

    def tool_identity(self) -> str:
        return BLOCKS_TOOL_IDENTITY

    def inputs_for(self, configured_before: Sequence[str]) -> tuple[str, ...]:
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
        return dict(params)

    def run(self, ctx: StageContext) -> StageResult:
        reads = _readings(ctx)
        width, height = png_size(ctx.inputs[RENDER_STAGE] / PAGE_IMAGE_FILENAME)
        ledger = assemble(ctx.page_id, ctx.pdf_page, width, height, reads)
        write_ledger(ctx.work_dir / BLOCKS_FILENAME, ledger)
        spans = sum(len(block.uncertainty) for block in ledger.blocks)
        return StageResult(meta={"blocks": len(ledger.blocks), "uncertainty_spans": spans})


def recognition_stages(adapters: AdapterRegistry) -> tuple[EngineStage | BlocksStage, ...]:
    """The five stages a recovery run adds to render: four engines and the ledger merge."""
    engines = tuple(EngineStage(name, adapters) for name in RECOGNITION_STAGES)
    return (*engines, BlocksStage())


def selected_engine(params: Mapping[str, Any]) -> str:
    """The engine this page runs: the page-class override when one is configured, else `engine`."""
    page_class = str(params.get("page_class") or "")
    by_class = params.get("by_page_class") or {}
    if page_class and isinstance(by_class, dict) and page_class in by_class:
        return str(by_class[page_class])
    return str(params.get("engine") or "")


def _variant(params: Mapping[str, Any]) -> str:
    return str(params.get("variant") or _DEFAULT_VARIANT)


def _read(stage: str, ctx: StageContext, adapter: Any, pdf: Any) -> EngineRead:
    from llb.prep.book.preprocess.frames import engine_request, reading_on_render

    name = _variant(ctx.params)
    variants = ctx.inputs.get(VARIANTS_STAGE)
    render = ctx.inputs[RENDER_STAGE] / PAGE_IMAGE_FILENAME
    request = engine_request(render, variants, name, pdf, ctx.work_dir, ctx.model)
    width, height = png_size(render)
    return reading_on_render(adapter.read(request, stage), variants, width, height)


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


def _meta(reading: EngineRead) -> dict[str, Any]:
    return {
        "engine": reading.engine,
        "model": reading.model,
        "variant": reading.variant,
        "blocks": len(reading.blocks),
    }


def _check_variant(stage: str, variant: object) -> list[str]:
    if isinstance(variant, str) and variant:
        return []
    return [f"{stage}: variant must be a non-empty string"]


def _check_page_class(stage: str, page_class: object) -> list[str]:
    if isinstance(page_class, str):
        return []
    return [f"{stage}: page_class must be a string"]


def _named(exc: EngineUnavailable) -> str:
    return f"{type(exc).__name__}: {exc}"
