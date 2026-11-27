"""Reuse a whole-page reading that an earlier recognition stage already stored.

A candidate engine ignores the stage name: one command returns layout, lines, formulas, and
tables together. Copying that reading keeps the model unloaded for the later stages.
"""

from llb.prep.book.cache import stage_dir
from llb.prep.book.constants import READING_FILENAME
from llb.prep.book.engines.protocol import EngineRead, read_reading
from llb.prep.book.stages import StageContext
from llb.prep.book.stages.recognize import RECOGNITION_STAGES


def shared_reading(
    ctx: StageContext, engine: str, variant: str, stage: str
) -> tuple[EngineRead, str] | None:
    """A sibling stage's blocks, relabeled for `stage`, and that stage's name."""
    source = _source(ctx, engine, variant, stage)
    if source is None:
        return None
    found, reading = source
    copied = EngineRead(reading.engine, reading.model, reading.variant, stage, reading.blocks)
    return copied, found


def _source(
    ctx: StageContext, engine: str, variant: str, stage: str
) -> tuple[str, EngineRead] | None:
    for name in RECOGNITION_STAGES:
        if name == stage:
            continue
        path = stage_dir(ctx.run_dir, ctx.page_id, name) / READING_FILENAME
        if not path.is_file():
            continue
        try:
            reading = read_reading(path)
        except (OSError, ValueError):
            continue
        if reading.engine != engine or reading.variant != variant or not reading.blocks:
            continue
        return name, reading
    return None
