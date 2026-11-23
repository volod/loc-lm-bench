"""Turn an in-memory warp into the manifest models, and write the manifest."""

import json
from pathlib import Path
from typing import Any

from llb.core.contracts.data_prep.book_preprocess import (
    AffineTransform,
    GridTransform,
    PageSignals,
    VariantManifest,
    VariantRecord,
)
from llb.core.fsutil import atomic_write_text
from llb.prep.book.preprocess.constants import (
    GRID_STEP,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_ID,
    SCHEMA_VERSION,
    SIGNALS_FILENAME,
)
from llb.prep.book.preprocess.geometry import Affine


def write_signals(path: Path, signals: PageSignals) -> None:
    """Write `signals.json`."""
    _write(path, signals.model_dump(mode="json"))


def read_signals(path: Path) -> PageSignals:
    """The signal vector, refusing a file that is not the current contract."""
    return PageSignals.model_validate_json(path.read_text(encoding="utf-8"))


def signals_path(directory: Path) -> Path:
    return directory / SIGNALS_FILENAME


def write_manifest(path: Path, manifest: VariantManifest) -> None:
    """Validate by construction and write the manifest."""
    _write(path, manifest.model_dump(mode="json"))


def publish_page_signals(stage_directory: Path) -> None:
    """Copy a committed `signals` stage's file up to `pages/<pNNNN>/signals.json`."""
    source = stage_directory / SIGNALS_FILENAME
    if not source.is_file():
        return
    target = stage_directory.parent / SIGNALS_FILENAME
    text = source.read_text(encoding="utf-8")
    if target.is_file() and target.read_text(encoding="utf-8") == text:
        return
    PageSignals.model_validate_json(text)
    atomic_write_text(target, text)


def _write(path: Path, payload: object) -> None:
    text = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, text)


def read_manifest(path: Path) -> VariantManifest:
    """The variant manifest, refusing a file that is not the current contract."""
    return VariantManifest.model_validate_json(path.read_text(encoding="utf-8"))


def manifest_path(directory: Path) -> Path:
    return directory / MANIFEST_FILENAME


def page_manifest(page_id: str, pdf_page: int, variants: list[VariantRecord]) -> VariantManifest:
    return VariantManifest(
        schema_id=MANIFEST_SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        kind="book-ocr-variant-manifest",
        page_id=page_id,
        pdf_page=pdf_page,
        variants=variants,
    )


def kept(
    name: str,
    image: str | None,
    transform: AffineTransform | GridTransform | None,
    parameters: dict[str, Any],
    *,
    geometric: bool,
) -> VariantRecord:
    return VariantRecord(
        name=name,
        status="kept",
        reason=None,
        parameters=parameters,
        image=image,
        geometric=geometric,
        transform=transform,
    )


def refused(
    name: str, reason: str, parameters: dict[str, Any], *, geometric: bool
) -> VariantRecord:
    return VariantRecord(
        name=name,
        status="refused",
        reason=reason,
        parameters=parameters,
        image=None,
        geometric=geometric,
        transform=None,
    )


def affine_record(transform: Affine) -> AffineTransform:
    matrix = [[round(value, 6) for value in row] for row in transform.matrix]
    return AffineTransform(
        kind="affine",
        matrix=matrix,
        src_width=transform.src_width,
        src_height=transform.src_height,
        width=transform.width,
        height=transform.height,
    )


def grid_record(map_x: object, map_y: object, src_width: int, src_height: int) -> GridTransform:
    """Downsample a dense variant-to-original map. The edges are always included."""
    import numpy as np

    xs = np.asarray(map_x, dtype=np.float64)
    ys = np.asarray(map_y, dtype=np.float64)
    height, width = xs.shape
    rows = _samples(height)
    cols = _samples(width)
    return GridTransform(
        kind="grid",
        rows=rows,
        cols=cols,
        src_width=src_width,
        src_height=src_height,
        width=width,
        height=height,
        map_x=[[round(float(xs[y, x]), 2) for x in cols] for y in rows],
        map_y=[[round(float(ys[y, x]), 2) for x in cols] for y in rows],
    )


def _samples(length: int) -> list[int]:
    chosen = list(range(0, length, GRID_STEP))
    if not chosen or chosen[-1] != length - 1:
        chosen.append(length - 1)
    return chosen
