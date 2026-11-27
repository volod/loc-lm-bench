"""The stage seam every recovery step plugs into.

A stage reads one page -- the source PDF and/or the committed outputs of the stages it names as
`inputs` -- and writes files into a private work directory. It never decides whether it should run:
the runner does that from the cache key, which is why a stage has to declare everything its output
depends on (`version`, `tool_identity`, its parameters, its model). Later tasks add real stages by
registering them; nothing in the runner changes.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from llb.core.contracts.data_prep.book import PageInventoryRecord

if TYPE_CHECKING:
    import fitz


@dataclass(frozen=True)
class StageContext:
    """Everything one stage invocation may read, and the one directory it may write."""

    page_id: str
    pdf_page: int
    inventory: PageInventoryRecord
    params: Mapping[str, Any]
    model: str | None
    inputs: Mapping[str, Path]  # upstream stage name -> its committed directory
    work_dir: Path
    run_dir: Path
    document: "fitz.Document"


@dataclass(frozen=True)
class StageResult:
    """Scalar facts a stage reports about its own output (recorded, never keyed on)."""

    meta: Mapping[str, Any] = field(default_factory=dict)


class Stage(Protocol):
    """One registered recovery step."""

    name: str
    version: str
    inputs: tuple[str, ...]
    reads_source: bool  # its key then includes the source digest and PDF page
    model_required: bool

    def tool_identity(self) -> str: ...

    def validate_params(self, params: Mapping[str, Any]) -> list[str]: ...

    def effective_params(
        self,
        params: Mapping[str, Any],
        inventory: PageInventoryRecord,
        document: "fitz.Document",
        pdf_page: int,
    ) -> dict[str, Any]:
        """The parameters this page actually runs with -- every default and `auto` resolved.

        The cache key digests THESE, not the configured ones, so a value the stage derives from
        the page (a native resolution, an automatic colorspace) is part of the key, and the stage
        receives exactly what its key names in `StageContext.params`.
        """
        ...

    def run(self, ctx: StageContext) -> StageResult: ...


class StageRegistry:
    """Name -> stage. The runner and the dry run only ever look stages up here."""

    def __init__(self, stages: Iterable[Stage]) -> None:
        self._stages: dict[str, Stage] = {}
        for stage in stages:
            if stage.name in self._stages:
                raise ValueError(f"stage {stage.name!r} registered twice")
            self._stages[stage.name] = stage

    def get(self, name: str) -> Stage:
        if name not in self._stages:
            raise KeyError(f"unknown stage {name!r}; registered: {', '.join(self.names())}")
        return self._stages[name]

    def names(self) -> list[str]:
        return sorted(self._stages)

    def __contains__(self, name: object) -> bool:
        return name in self._stages


def resolved_inputs(
    stage: Stage, configured_before: Iterable[str], params: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """The stages `stage` reads, given what was configured ahead of it.

    A stage with `inputs_for` chooses -- the block ledger reads whichever recognition stages the
    run configured, and an engine stage reads `classes` when `by_page_class` routes on it. Every
    other stage reads the inputs it declared.
    """
    choose = getattr(stage, "inputs_for", None)
    before = tuple(configured_before)
    if choose is not None:
        return tuple(choose(before, params))
    return tuple(stage.inputs)


def default_registry() -> StageRegistry:
    """The stages a `book-ocr` run can name today."""
    from llb.prep.book.engines.registry import default_adapter_registry
    from llb.prep.book.stages.classes import ClassesStage
    from llb.prep.book.stages.fake import FAKE_STAGES
    from llb.prep.book.stages.preprocess import SignalsStage, VariantsStage
    from llb.prep.book.stages.recognize import recognition_stages
    from llb.prep.book.stages.render import RenderStage
    from llb.prep.book.stages.resolve import ResolveStage
    from llb.prep.book.stages.correct import CorrectStage

    return StageRegistry(
        (
            RenderStage(),
            SignalsStage(),
            VariantsStage(),
            ClassesStage(),
            *FAKE_STAGES,
            *recognition_stages(default_adapter_registry()),
            CorrectStage(),
            ResolveStage(),
        )
    )
