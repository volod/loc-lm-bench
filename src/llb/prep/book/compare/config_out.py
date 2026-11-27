"""The book-ocr configuration a comparison selects.

The file is the run config the dry run already understands. ``vram_mb`` is the
budget that dry run checks. ``ocr_language`` is the baseline's recorded language,
and it is part of the recognition cache key.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from llb.prep.book.preprocess.constants import DEWARP_OPENCV


@dataclass(frozen=True)
class StageRouting:
    """The engine a recognition stage runs, and the classes that override it."""

    engine: str
    by_page_class: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineChoice:
    """Everything the YAML needs, and nothing the report already explains."""

    language: str
    default_variant: str
    variant_by_page_class: dict[str, str]
    dewarp: str
    include_variants: bool
    stages: dict[str, StageRouting]
    vram_mb: dict[str, int]


def pipeline_document(choice: PipelineChoice) -> dict[str, object]:
    """A ``book-ocr`` config mapping. Pages are the whole book."""
    stages: list[object] = ["render", "signals", "classes"]
    if choice.include_variants:
        stages.append({"name": "variants", "params": {"dewarp": choice.dewarp or DEWARP_OPENCV}})
    for name in ("layout", "ocr", "formula", "table"):
        routing = choice.stages[name]
        params: dict[str, object] = {
            "engine": routing.engine,
            "variant": choice.default_variant,
            "ocr_language": choice.language,
            "vram_mb": dict(choice.vram_mb),
        }
        if routing.by_page_class:
            params["by_page_class"] = dict(routing.by_page_class)
        if choice.variant_by_page_class:
            params["variant_by_page_class"] = dict(choice.variant_by_page_class)
        stages.append({"name": name, "params": params})
    stages.append("blocks")
    return {"pages": "1-", "stages": stages}


def stage_routings(
    class_engines: dict[str, str],
    baseline_engine: str,
    serves: Callable[[str, str], bool],
) -> dict[str, StageRouting]:
    """Per recognition stage, the default engine and the classes that differ.

    A class whose winner does not serve the stage keeps the stage default.
    """
    winners = list(dict.fromkeys(class_engines.values()))
    routed: dict[str, StageRouting] = {}
    for name in ("layout", "ocr", "formula", "table"):
        default = _default(name, baseline_engine, winners, serves)
        overrides = {
            class_name: engine
            for class_name, engine in class_engines.items()
            if engine != default and serves(engine, name)
        }
        routed[name] = StageRouting(default, overrides)
    return routed


def _default(
    stage: str, baseline: str, winners: list[str], serves: Callable[[str, str], bool]
) -> str:
    if serves(baseline, stage):
        return baseline
    for engine in winners:
        if serves(engine, stage):
            return engine
    raise ValueError(f"no configured engine serves stage {stage}")
