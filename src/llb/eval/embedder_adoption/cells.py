"""Build the sweep grid and turn one (cell, encoder) pair into an ordinary `run-eval` config.

The grid is deliberately a product of the two knobs that decide whether first-hit RANK binds:
`top_k` (how many chunks the answer ever sees) and the cross-encoder reranker (which can only
re-sort what the encoder handed it). Everything else -- corpus, chunking, model, seed -- is held
fixed by the single base config, so a cell's delta is attributable to the encoder alone.
"""

from collections.abc import Sequence
from pathlib import Path

from llb.core.config import RunConfig
from llb.eval.embedder_adoption.models import CellSpec, EmbedderLane

RUN_NAME_PREFIX = "embedder-adoption"

# `--rerankers` tokens: the two spellings that mean "no cross-encoder" and "the pinned one".
RERANKER_OFF = "off"
RERANKER_ON = "on"


def parse_rerankers(spec: str) -> list[str | None]:
    """Parse `off,on` (or an explicit cross-encoder id) into reranker settings, `None` == off."""
    from llb.rag.rerank import DEFAULT_RERANKER

    values: list[str | None] = []
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        setting = (
            None if token == RERANKER_OFF else (DEFAULT_RERANKER if token == RERANKER_ON else token)
        )
        if setting not in values:
            values.append(setting)
    if not values:
        raise ValueError("name at least one reranker setting (off | on | <cross-encoder id>)")
    return values


def build_cells(top_ks: Sequence[int], rerankers: Sequence[str | None]) -> list[CellSpec]:
    """The full `top_k` x reranker grid, in the order the two selections were given."""
    return [CellSpec(top_k, reranker) for top_k in top_ks for reranker in rerankers]


def cell_config(config: RunConfig, cell: CellSpec, lane: EmbedderLane) -> RunConfig:
    """`config` with this cell's retrieval budget and this encoder's model + store root applied.

    Built by revalidating an explicit field mapping rather than `with_overrides`, because a cell
    must be able to set `reranker` back to `None` (the off half of the grid) and `with_overrides`
    drops `None` by design.

    `data_dir` moves with the encoder: `RunConfig.index_dir()` resolves from it, and a store is
    embedded and queried by ONE encoder, so the two cannot be varied independently.
    """
    values = config.model_dump()
    values.update(
        run_name=f"{RUN_NAME_PREFIX}-{cell.label}-{lane.model}",
        data_dir=Path(lane.data_dir),
        embedding_model=lane.model,
        top_k=cell.top_k,
        reranker=cell.reranker,
    )
    return RunConfig.model_validate(values)
