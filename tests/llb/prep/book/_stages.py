"""Injected test stages: one that counts its invocations, one that dies mid-write."""

from collections import Counter
from collections.abc import Mapping
from typing import Any

from llb.prep.book.stages import StageContext, StageRegistry, StageResult
from llb.prep.book.stages.fake import FAKE_STAGES
from llb.prep.book.stages.render import PAGE_IMAGE_FILENAME, RENDER_STAGE, RenderStage


class SimulatedKill(BaseException):
    """What a SIGKILL looks like to the runner: nothing below `BaseException` catches it."""


class CountingStage:
    """Writes a digest of the page image; counts every invocation per page."""

    version = "1"
    reads_source = False
    model_required = False

    def __init__(self, name: str, inputs: tuple[str, ...] = (RENDER_STAGE,)) -> None:
        self.name = name
        self.inputs = inputs
        self.calls: Counter[int] = Counter()
        self.kill_on: int | None = None
        self.fail_on: set[int] = set()

    def tool_identity(self) -> str:
        return "test-counting 1"

    def validate_params(self, params: Mapping[str, Any]) -> list[str]:
        return []

    def effective_params(
        self, params: Mapping[str, Any], inventory: Any, document: Any, pdf_page: int
    ) -> dict[str, Any]:
        return dict(params)

    def run(self, ctx: StageContext) -> StageResult:
        self.calls[ctx.pdf_page] += 1
        image = (ctx.inputs[RENDER_STAGE] / PAGE_IMAGE_FILENAME).read_bytes()
        (ctx.work_dir / "out.txt").write_text(f"{len(image)} {ctx.params}\n", encoding="utf-8")
        if ctx.pdf_page == self.kill_on:
            self.kill_on = None  # die once, like a process that is then restarted
            raise SimulatedKill()
        if ctx.pdf_page in self.fail_on:
            raise RuntimeError(f"engine crashed on page {ctx.pdf_page}")
        (ctx.work_dir / "second.txt").write_text("complete\n", encoding="utf-8")
        return StageResult(meta={"bytes": len(image)})


class CountingRender(RenderStage):
    def __init__(self) -> None:
        self.calls: Counter[int] = Counter()

    def run(self, ctx: StageContext) -> StageResult:
        self.calls[ctx.pdf_page] += 1
        return super().run(ctx)


def registry_with(*stages: Any) -> StageRegistry:
    return StageRegistry(stages or (CountingRender(), *FAKE_STAGES))
