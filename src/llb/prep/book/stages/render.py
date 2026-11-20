"""The `render` stage: one page image per PDF page, at the scan's native resolution.

Its key holds the RESOLVED resolution and colorspace, so a change in how `native` or `auto`
resolves re-renders the pages it affects. Its output digest IS the page-image digest every later stage keys on, so a re-render that changes
no pixel re-runs nothing downstream, and a render change (resolution, colorspace, PyMuPDF build)
re-runs exactly what read the old image.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from llb.core.contracts.data_prep.book import PageInventoryRecord

from llb.prep.book.inventory import page_is_grey, render_dpi, render_page_png
from llb.prep.book.stages import StageContext, StageResult

if TYPE_CHECKING:
    import fitz

PAGE_IMAGE_FILENAME = "page.png"
RENDER_STAGE = "render"
DPI_NATIVE = "native"
COLORSPACE_AUTO = "auto"
COLORSPACE_GREY = "grey"
COLORSPACE_RGB = "rgb"
COLORSPACES = (COLORSPACE_AUTO, COLORSPACE_GREY, COLORSPACE_RGB)


class RenderStage:
    name = RENDER_STAGE
    version = "1"
    inputs: tuple[str, ...] = ()
    reads_source = True
    model_required = False

    def tool_identity(self) -> str:
        import fitz

        return f"pymupdf {fitz.VersionBind}"

    def validate_params(self, params: Mapping[str, Any]) -> list[str]:
        errors = [f"render: unknown parameter {key!r}" for key in params if key not in _PARAMS]
        dpi = params.get("dpi", DPI_NATIVE)
        if dpi != DPI_NATIVE and not (
            isinstance(dpi, int) and not isinstance(dpi, bool) and dpi > 0
        ):
            errors.append(f"render: dpi must be 'native' or a positive integer, got {dpi!r}")
        colorspace = params.get("colorspace", COLORSPACE_AUTO)
        if colorspace not in COLORSPACES:
            errors.append(f"render: colorspace must be one of {COLORSPACES}, got {colorspace!r}")
        return errors

    def effective_params(
        self,
        params: Mapping[str, Any],
        inventory: PageInventoryRecord,
        document: "fitz.Document",
        pdf_page: int,
    ) -> dict[str, Any]:
        colorspace = params.get("colorspace", COLORSPACE_AUTO)
        if colorspace == COLORSPACE_AUTO:
            colorspace = COLORSPACE_GREY if page_is_grey(document, pdf_page) else COLORSPACE_RGB
        return {
            "dpi": render_dpi(inventory, params.get("dpi", DPI_NATIVE)),
            "colorspace": colorspace,
        }

    def run(self, ctx: StageContext) -> StageResult:
        dpi, colorspace = int(ctx.params["dpi"]), str(ctx.params["colorspace"])
        png = render_page_png(ctx.document, ctx.pdf_page, dpi, colorspace == COLORSPACE_GREY)
        (ctx.work_dir / PAGE_IMAGE_FILENAME).write_bytes(png)
        return StageResult(meta={"dpi": dpi, "colorspace": colorspace})


_PARAMS = ("dpi", "colorspace")
