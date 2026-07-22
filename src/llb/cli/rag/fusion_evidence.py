"""Fixed and question-routed graph fusion with multi-hop evidence."""

from pathlib import Path
from typing import Any, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED

FUSION_EVIDENCE_METHOD = "graph-vector-fusion-multihop"
DEFAULT_WEIGHT_GRID = "0,0.1,0.2,0.3,0.5,0.7,1.0"
# Default to the historical single depth (each lane asked for exactly the scored `k`), so the
# command's out-of-the-box row set is unchanged until an operator asks for a deeper pool.
DEFAULT_CANDIDATE_GRID = "k"
# Default to the historical identity rule (exact offsets), so the row set is unchanged until an
# operator asks for the containment/overlap policy.
DEFAULT_SPAN_IDENTITY_GRID = "exact"
DEFAULT_ROUTED_GRAPH_WEIGHT = 0.3


@app.command("compare-graph-fusion")
def compare_graph_fusion_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    k: int = typer.Option(10, min=1, help="recall@k / all-spans@k / MRR cutoff"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
    graph_weights: str = typer.Option(
        DEFAULT_WEIGHT_GRID, help="comma-separated graph shares to sweep (each within [0, 1])"
    ),
    routed_graph_weight: float = typer.Option(
        DEFAULT_ROUTED_GRAPH_WEIGHT,
        min=0.0,
        max=1.0,
        help="graph share used only for questions the question-type router sends to fusion; "
        "other questions are exact vector passthroughs",
    ),
    routing_sidecar: bool = typer.Option(
        True,
        "--routing-sidecar/--no-routing-sidecar",
        help="use question-type sidecar labels for routed rows; disable to exercise only the "
        "deterministic fallback",
    ),
    heuristic_long_question_words: int = typer.Option(
        16, min=1, help="sidecar-free router: minimum words for the long-question signal"
    ),
    heuristic_min_linked_entities: int = typer.Option(
        2,
        min=0,
        help="sidecar-free router: minimum capitalized entities required with a long question; "
        "zero makes length sufficient",
    ),
    graph_fusion_candidates: str = typer.Option(
        DEFAULT_CANDIDATE_GRID,
        help="comma-separated per-lane candidate depths to sweep; 'k' == the scored cutoff "
        "(the shallow pool), a larger number fuses a deeper pool and then cuts to k",
    ),
    graph_fusion_span_identity: str = typer.Option(
        DEFAULT_SPAN_IDENTITY_GRID,
        help="comma-separated span-identity policies to sweep: 'exact' (identical offsets) and/or "
        "'overlap' (fold a graph span into the vector chunk that contains it)",
    ),
    graph_strategies: Optional[str] = typer.Option(
        None, help="comma-separated graph strategies (default: local_khop,global_community)"
    ),
    focus_slice: Optional[str] = typer.Option(
        None, help="question type the verdict is decided on (default: multi-hop)"
    ),
    resamples: int = typer.Option(DEFAULT_RESAMPLES, min=0, help="bootstrap resamples"),
    confidence: float = typer.Option(DEFAULT_CONFIDENCE, min=0.5, max=0.999, help="CI level"),
    seed: int = typer.Option(DEFAULT_SEED, help="bootstrap resampling seed"),
    out_dir: Optional[Path] = typer.Option(
        None, help=f"artifact dir (default: $DATA_DIR/{FUSION_EVIDENCE_METHOD}/<timestamp>/)"
    ),
) -> None:
    """Compare fixed graph shares plus question routing on the multi-hop slice.

    `compare-retrieval` ranks backends over a whole gold set. This lane answers the narrower
    question a graph-weight recommendation needs: on items whose answer requires MORE THAN ONE
    source span, does fusing graph evidence into the vector lane retrieve more of that evidence,
    at which weight, and at what cost overall. Every number carries a paired bootstrap interval
    and the item-level win/loss ledger, because a multi-hop slice is a dozen items.

    Each physical lane is retrieved once per question and re-fused at every fixed or routed row,
    so the comparison costs one vector pass plus one pass per graph strategy.
    """
    import json

    from llb.core.store_generations import generation_timestamp
    from llb.rag.fusion_evidence import (
        build_sweep_rows,
        evaluate_fusion_evidence,
        format_report,
        parse_candidates,
        parse_span_identities,
        parse_weights,
    )
    from llb.rag.fusion_evidence.models import FOCUS_SLICE
    from llb.rag.fusion_evidence.rows import VECTOR_ROW
    from llb.rag.question_types import load_question_types_by_question
    from llb.rag.fusion_routing import HeuristicPolicy

    cfg = load_config(config, goldset_path=goldset)
    try:
        weights = parse_weights(graph_weights)
        candidates = parse_candidates(graph_fusion_candidates)
        identities = parse_span_identities(graph_fusion_span_identity)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    items = _evidence_items(cfg, split)
    if not items:
        typer.echo("[error] the gold set selection is empty", err=True)
        raise typer.Exit(code=2)
    vector, graphs = _load_lanes(cfg, graph_strategies)
    question_types = load_question_types_by_question(cfg.goldset_path) if routing_sidecar else {}
    rows = build_sweep_rows(
        vector,
        graphs,
        [item.question for item in items],
        k,
        weights,
        candidates,
        identities,
        routed_graph_weight,
        question_types,
        HeuristicPolicy(heuristic_long_question_words, heuristic_min_linked_entities),
    )
    report = evaluate_fusion_evidence(
        rows,
        items,
        k,
        baseline=VECTOR_ROW,
        focus_slice=focus_slice or FOCUS_SLICE,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    default_dir = cfg.data_dir / FUSION_EVIDENCE_METHOD / generation_timestamp()
    target = Path(out_dir) if out_dir else default_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "report.md").write_text(format_report(report), encoding="utf-8")
    (target / "run_config.json").write_text(
        json.dumps(cfg.fingerprint(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    verdict = report["verdict"]
    typer.echo(
        f"[compare-graph-fusion] {verdict['decision']}: {verdict['reason']}"
        if verdict["reason"]
        else f"[compare-graph-fusion] {verdict['decision']}"
    )
    typer.echo(f"[compare-graph-fusion] report -> {target / 'report.md'}")


def _evidence_items(cfg: Any, split: Optional[str]) -> list[Any]:
    """Load the gold selection as `EvidenceItem`s carrying their question-type slice."""
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


def _load_lanes(cfg: Any, graph_strategies: Optional[str]) -> tuple[Any, dict[str, Any]]:
    """The built vector store plus one loaded graph store per compared strategy."""
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
