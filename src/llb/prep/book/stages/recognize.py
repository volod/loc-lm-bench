"""Layout, OCR, formula, and table stages.

An engine stage's cache key names the engine that actually runs on the page, after `by_page_class`
is resolved. The merge of those readings lives in `blocks.py`.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from llb.prep.book.engines.image import png_size
from llb.prep.book.engines.lease import EngineLease
from llb.prep.book.engines.pdf_page import write_page_pdf
from llb.prep.book.engines.protocol import EngineRead, EngineUnavailable, write_reading
from llb.prep.book.engines.registry import AdapterRegistry
from llb.prep.book.preprocess.constants import ORIGINAL, VARIANTS_STAGE, VARIANT_NAMES
from llb.prep.book.stages import StageContext, StageResult
from llb.prep.book.stages.render import PAGE_IMAGE_FILENAME, RENDER_STAGE
from llb.prep.book.stages.routing import (
    batch_page_order,
    engine_inputs,
    routed_params,
    selected_engine,
)

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
_BOX_MAP_RECIPE = "2"


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
        variant = _variant(params)
        resolved = {"engine": selected_engine(params), "variant": variant}
        if variant != ORIGINAL:
            resolved["box_map_recipe"] = _BOX_MAP_RECIPE
        return resolved

    def page_params(
        self,
        params: Mapping[str, Any],
        inputs: Mapping[str, Any],
        inventory: Any,
        document: Any,
        pdf_page: int,
    ) -> dict[str, Any]:
        """Resolve `page_class` from the classes stage, then the engine that class selects."""
        return self.effective_params(routed_params(params, inputs), inventory, document, pdf_page)

    def inputs_for(
        self, configured_before: Sequence[str], params: Mapping[str, Any] | None = None
    ) -> tuple[str, ...]:
        return engine_inputs(configured_before, params)

    def batch_order(self, run_dir: Any, config: Any, batch: Sequence[int], spec: Any) -> list[int]:
        """Pages of one batch grouped so each routed engine opens once."""
        return batch_page_order(run_dir, config.source.page_count, batch, spec.params)

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


def recognition_stages(adapters: AdapterRegistry) -> tuple[Any, ...]:
    """The five stages a recovery run adds to render: four engines and the ledger merge."""
    from llb.prep.book.stages.blocks import BlocksStage

    engines = tuple(EngineStage(name, adapters) for name in RECOGNITION_STAGES)
    return (*engines, BlocksStage())


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
