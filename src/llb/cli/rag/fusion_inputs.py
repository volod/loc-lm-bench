"""Gold-item and store loading for graph-vector fusion commands."""

from typing import Any, Optional

import typer

FUSION_EVIDENCE_METHOD = "graph-vector-fusion-multihop"


def evidence_items(cfg: Any, split: Optional[str]) -> list[Any]:
    """Load the gold selection with question-type labels."""
    from llb.executor.cases import spans_as_dicts
    from llb.goldset.schema import load_goldset
    from llb.rag.fusion_evidence import EvidenceItem
    from llb.rag.question_types import load_question_types

    items = load_goldset(cfg.goldset_path)
    if split:
        items = [item for item in items if item.split == split]
    types = load_question_types(cfg.goldset_path)
    return [
        EvidenceItem(item.id, item.question, spans_as_dicts(item), types.get(item.id))
        for item in items
    ]


def load_lanes(cfg: Any, graph_strategies: Optional[str]) -> tuple[Any, dict[str, Any]]:
    """Load the vector store and each requested graph strategy."""
    from llb.executor.runner_retrieval import _load_store
    from llb.graph.constants import (
        BACKEND_GRAPH,
        STRATEGY_GLOBAL_COMMUNITY,
        STRATEGY_LOCAL_KHOP,
    )

    selected = (
        [name.strip() for name in graph_strategies.split(",") if name.strip()]
        if graph_strategies
        else [STRATEGY_LOCAL_KHOP, STRATEGY_GLOBAL_COMMUNITY]
    )
    try:
        vector = _load_store(cfg.with_overrides(retrieval_backend="faiss"))
        graphs = {
            strategy: _load_store(
                cfg.with_overrides(retrieval_backend=BACKEND_GRAPH, retrieval_strategy=strategy)
            )
            for strategy in selected
        }
    except (FileNotFoundError, SystemExit) as exc:
        typer.echo(f"[error] a compared store is not built: {exc}", err=True)
        raise typer.Exit(code=2) from None
    return vector, graphs
