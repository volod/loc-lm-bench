"""Read the image a variant names, and send engine boxes back to the original render."""

from pathlib import Path

from llb.core.contracts.data_prep.book_preprocess import Transform, VariantRecord
from llb.prep.book.engines.image import png_size
from llb.prep.book.engines.protocol import EngineRead, EngineRequest
from llb.prep.book.preprocess.constants import MANIFEST_FILENAME, ORIGINAL
from llb.prep.book.preprocess.maps import map_box
from llb.prep.book.preprocess.records import read_manifest


class VariantUnavailable(RuntimeError):
    """The named variant was not kept for this page. There is no silent fallback."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason
        super().__init__(f"variant {name} unavailable: {reason}")


def engine_request(
    render_png: Path,
    variants_dir: Path | None,
    name: str,
    pdf: Path,
    work_dir: Path,
    model: str | None,
    language: str | None = None,
) -> EngineRequest:
    """The engine call for `name`, reading that variant's image when it is not the original."""
    image, width, height = page_image(render_png, variants_dir, name)
    return EngineRequest(
        image=image,
        pdf=pdf,
        work_dir=work_dir,
        variant=name,
        model=model,
        width=width,
        height=height,
        language=language,
    )


def page_image(render_png: Path, variants_dir: Path | None, name: str) -> tuple[Path, int, int]:
    """The image an engine should read, and its pixel size."""
    if name == ORIGINAL:
        width, height = png_size(render_png)
        return render_png, width, height
    if variants_dir is None:
        raise VariantUnavailable(name, "variants stage was not configured")
    path = kept_image(variants_dir, name)
    width, height = png_size(path)
    return path, width, height


def reading_on_render(
    reading: EngineRead, variants_dir: Path | None, src_width: int, src_height: int
) -> EngineRead:
    """`reading` with every box mapped into the original render. `original` is already there."""
    if reading.variant == ORIGINAL:
        return reading
    if variants_dir is None:
        raise VariantUnavailable(reading.variant, "variants stage was not configured")
    transform = kept_transform(variants_dir, reading.variant)
    blocks = [
        mapped
        for box in reading.blocks
        if (mapped := map_box(transform, box, src_width, src_height)) is not None
    ]
    return EngineRead(reading.engine, reading.model, reading.variant, reading.stage, tuple(blocks))


def kept_image(variants_dir: Path, name: str) -> Path:
    """The PNG of a kept variant, or `VariantUnavailable`."""
    record = _kept(variants_dir, name)
    if not record.image:
        raise VariantUnavailable(name, "kept variant has no image")
    path = variants_dir / record.image
    if not path.is_file():
        raise VariantUnavailable(name, "image file is missing")
    return path


def kept_transform(variants_dir: Path, name: str) -> Transform:
    """The point map of a kept variant."""
    record = _kept(variants_dir, name)
    if record.transform is None:
        raise VariantUnavailable(name, "kept variant has no point map")
    return record.transform


def _kept(variants_dir: Path, name: str) -> VariantRecord:
    manifest = read_manifest(variants_dir / MANIFEST_FILENAME)
    record = next((item for item in manifest.variants if item.name == name), None)
    if record is None:
        raise VariantUnavailable(name, "not generated for this page")
    if record.status != "kept":
        raise VariantUnavailable(name, record.reason or "refused")
    return record
