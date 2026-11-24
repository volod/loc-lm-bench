"""Which engine a page runs, and the order that opens each engine once per batch.

`by_page_class` names an engine per class. When the `classes` stage ran earlier, the class is that
stage's primary class, and its file is an input of the engine stage, so a class change is a cache
miss for the stages that route on it and a hit for the ones that do not. Pages that share an
engine are run together, in the order the batch first asked for that engine.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llb.prep.book.cache import stage_dir
from llb.prep.book.classes.constants import CLASSES_FILENAME, CLASSES_STAGE
from llb.prep.book.classes.records import read_classes
from llb.prep.book.preprocess.constants import VARIANTS_STAGE
from llb.prep.book.ranges import page_id
from llb.prep.book.stages.render import RENDER_STAGE


def engine_inputs(
    configured_before: Sequence[str], params: Mapping[str, Any] | None
) -> tuple[str, ...]:
    """`render`, `variants` when a correction may be selected, and `classes` when routing uses it."""
    chosen = [RENDER_STAGE]
    if VARIANTS_STAGE in configured_before:
        chosen.append(VARIANTS_STAGE)
    if _routes(params) and CLASSES_STAGE in configured_before:
        chosen.append(CLASSES_STAGE)
    return tuple(chosen)


def routed_params(params: Mapping[str, Any], inputs: Mapping[str, Path]) -> dict[str, Any]:
    """The configured parameters, with `page_class` taken from the classes stage when it ran."""
    routed = dict(params)
    directory = inputs.get(CLASSES_STAGE)
    if directory is None:
        return routed
    path = directory / CLASSES_FILENAME
    if path.is_file():
        routed["page_class"] = read_classes(path).primary
    return routed


def batch_page_order(
    run_dir: Path, page_count: int, batch: Sequence[int], params: Mapping[str, Any]
) -> list[int]:
    """`batch` grouped by the engine each page will open. Unrouted batches stay in page order."""
    if not _routes(params):
        return list(batch)
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for pdf_page in batch:
        engine = selected_engine(_page_params(run_dir, page_count, pdf_page, params))
        if engine not in groups:
            groups[engine] = []
            order.append(engine)
        groups[engine].append(pdf_page)
    return [pdf_page for engine in order for pdf_page in groups[engine]]


def selected_engine(params: Mapping[str, Any]) -> str:
    """The engine this page runs: the page-class override when one is configured, else `engine`."""
    return _selected(params, "by_page_class", "engine")


def selected_variant(params: Mapping[str, Any]) -> str:
    """The variant this page runs: the page-class override, else `variant`, else the original."""
    chosen = _selected(params, "variant_by_page_class", "variant")
    return chosen or "original"


def _selected(params: Mapping[str, Any], map_key: str, default_key: str) -> str:
    page_class = str(params.get("page_class") or "")
    by_class = params.get(map_key) or {}
    if page_class and isinstance(by_class, dict) and page_class in by_class:
        return str(by_class[page_class])
    return str(params.get(default_key) or "")


def _routes(params: Mapping[str, Any] | None) -> bool:
    if not isinstance(params, Mapping):
        return False
    for key in ("by_page_class", "variant_by_page_class"):
        mapping = params.get(key) or {}
        if isinstance(mapping, dict) and mapping:
            return True
    return False


def _page_params(
    run_dir: Path, page_count: int, pdf_page: int, params: Mapping[str, Any]
) -> dict[str, Any]:
    routed = dict(params)
    path = stage_dir(run_dir, page_id(pdf_page, page_count), CLASSES_STAGE) / CLASSES_FILENAME
    if path.is_file():
        routed["page_class"] = read_classes(path).primary
    return routed
