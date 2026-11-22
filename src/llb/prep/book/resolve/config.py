"""Resolve-stage parameters: which reading is primary for a page class.

`by_page_class` selects a reading the way an engine stage selects an engine. When the classes
stage ran, its primary class is the one the map sees, and that file is a resolve input.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llb.prep.book.classes.constants import CLASSES_FILENAME, CLASSES_STAGE
from llb.prep.book.classes.records import read_classes
from llb.prep.book.resolve.constants import CORRECT_STAGE
from llb.prep.book.stages.recognize import BLOCKS_STAGE, RECOGNITION_STAGES

_PARAMS = ("primary", "by_page_class", "page_class")
_CHOICE_KEYS = ("stage", "engine", "variant", "model")
_STAGES = ", ".join(RECOGNITION_STAGES)


def validate_resolve_params(params: Mapping[str, Any]) -> list[str]:
    """Problems with the configured primary reading. Empty when the mapping is usable."""
    errors = [f"resolve: unknown parameter {key!r}" for key in params if key not in _PARAMS]
    primary = params.get("primary")
    if primary is None:
        errors.append("resolve: primary must name the reading a page shows")
    else:
        errors.extend(_choice_errors(primary, "resolve primary"))
    by_class = params.get("by_page_class", {})
    if not isinstance(by_class, dict):
        errors.append("resolve: by_page_class must map a page class to a reading")
    else:
        for key, choice in by_class.items():
            if not isinstance(key, str):
                errors.append("resolve: by_page_class keys must be strings")
                continue
            errors.extend(_choice_errors(choice, f"resolve by_page_class[{key}]"))
    if not isinstance(params.get("page_class", ""), str):
        errors.append("resolve: page_class must be a string")
    return errors


def order_errors(params: Mapping[str, Any], configured_before: Sequence[str]) -> list[str]:
    """The primary reading's stage has to have run, or the ledger will not carry it."""
    if not isinstance(params.get("primary"), dict):
        return []
    missing = [
        name
        for name in _named_stages(params)
        if name in RECOGNITION_STAGES and name not in configured_before
    ]
    return [
        f"stage 'resolve' reads the {name!r} reading, which must be configured before it"
        for name in missing
    ]


def resolve_inputs(
    configured_before: Sequence[str], params: Mapping[str, Any] | None
) -> tuple[str, ...]:
    """`blocks`, `correct` when that stage ran, and `classes` when the primary depends on it."""
    chosen = [BLOCKS_STAGE]
    if CORRECT_STAGE in configured_before:
        chosen.append(CORRECT_STAGE)
    if _routes(params) and CLASSES_STAGE in configured_before:
        chosen.append(CLASSES_STAGE)
    return tuple(chosen)


def page_choice(params: Mapping[str, Any], inputs: Mapping[str, Path]) -> dict[str, Any]:
    """The primary reading this page will use, with `page_class` filled in from the class file."""
    page_class = _page_class(params, inputs)
    choice = params["primary"]
    by_class = params.get("by_page_class") or {}
    if page_class and isinstance(by_class, dict) and page_class in by_class:
        choice = by_class[page_class]
    model = choice.get("model")
    return {
        "page_class": page_class,
        "stage": str(choice["stage"]),
        "engine": str(choice["engine"]),
        "variant": str(choice.get("variant") or "original"),
        "model": None if not model else str(model),
    }


def _choice_errors(choice: object, where: str) -> list[str]:
    if not isinstance(choice, dict):
        return [f"{where} must be a mapping"]
    errors = [f"{where}: unknown parameter {key!r}" for key in choice if key not in _CHOICE_KEYS]
    stage = choice.get("stage")
    if stage not in RECOGNITION_STAGES:
        errors.append(f"{where}: stage must be one of {_STAGES}")
    engine = choice.get("engine")
    if not isinstance(engine, str) or not engine:
        errors.append(f"{where}: engine must name the primary reading")
    variant = choice.get("variant", "original")
    if not isinstance(variant, str) or not variant:
        errors.append(f"{where}: variant must be a non-empty string")
    model = choice.get("model")
    if model is not None and (not isinstance(model, str) or not model):
        errors.append(f"{where}: model must be a non-empty string when set")
    return errors


def _named_stages(params: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    primary = params.get("primary")
    if isinstance(primary, dict) and isinstance(primary.get("stage"), str):
        names.append(primary["stage"])
    by_class = params.get("by_page_class")
    if isinstance(by_class, dict):
        for choice in by_class.values():
            if isinstance(choice, dict) and isinstance(choice.get("stage"), str):
                names.append(choice["stage"])
    return list(dict.fromkeys(names))


def _routes(params: Mapping[str, Any] | None) -> bool:
    if not isinstance(params, Mapping):
        return False
    mapping = params.get("by_page_class") or {}
    return isinstance(mapping, dict) and bool(mapping)


def _page_class(params: Mapping[str, Any], inputs: Mapping[str, Path]) -> str:
    directory = inputs.get(CLASSES_STAGE)
    if directory is not None:
        path = directory / CLASSES_FILENAME
        if path.is_file():
            return read_classes(path).primary
    return str(params.get("page_class") or "")
