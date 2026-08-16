"""Per-hop multi-hop retrievability probe: is a missing hop a budget or a query problem?"""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.cli.rag import fusion_inputs
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED
from llb.rag.multihop_probe.models import DEFAULT_BUDGETS, DEFAULT_PROBE_DEPTH

DEFAULT_BUDGET_GRID = ",".join(str(budget) for budget in DEFAULT_BUDGETS)


def _lane_label(cfg: object) -> str:
    """Name the probed lane the way a fusion row is named, so two artifacts are comparable."""
    backend = getattr(cfg, "retrieval_backend", "faiss")
    if backend in {"graph", "fused"}:
        return f"{backend}/{getattr(cfg, 'retrieval_strategy', '')}"
    return str(backend)


@app.command("probe-multihop-hops")
def probe_multihop_hops_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
    budgets: str = typer.Option(
        DEFAULT_BUDGET_GRID,
        help="comma-separated retrieval budgets the all-spans@k curve is read at; the SMALLEST "
        "is the operating budget every diagnosis is stated against",
    ),
    probe_depth: int = typer.Option(
        DEFAULT_PROBE_DEPTH,
        min=1,
        help="how deep a labeled span is searched for before the query counts as unable to reach "
        "it (raised to the largest budget when smaller)",
    ),
    focus_slice: Optional[str] = typer.Option(
        None, help="question type the diagnosis is reported for first (default: multi-hop)"
    ),
    retrieval_backend: Optional[str] = typer.Option(
        None, help="probe a different lane than the config names (faiss | graph | fused)"
    ),
    retrieval_strategy: Optional[str] = typer.Option(
        None, help="graph strategy when the probed lane is graph or fused"
    ),
    resamples: int = typer.Option(DEFAULT_RESAMPLES, min=0, help="bootstrap resamples"),
    confidence: float = typer.Option(DEFAULT_CONFIDENCE, min=0.5, max=0.999, help="CI level"),
    seed: int = typer.Option(DEFAULT_SEED, help="bootstrap resampling seed"),
    out_dir: Optional[Path] = typer.Option(
        None,
        help=f"artifact dir (default: $DATA_DIR/"
        f"{fusion_inputs.FUSION_EVIDENCE_METHOD}/<timestamp>-hop-probe/)",
    ),
) -> None:
    """Diagnose WHY a multi-hop item misses `all-spans@k`: the budget, the query, or the index.

    `compare-graph-fusion` sweeps ranking knobs and reports whether both hops arrive. When that
    number will not move, the remaining explanations are not about ranking. This lane ranks every
    labeled span twice -- by the item's own question at a deep pool, and by the span's own text --
    and reads the `all-spans@k` curve over a budget grid against those ranks. A hop the question
    reaches below the cut is a BUDGET problem (a larger k or a second pass fixes it); a hop only
    its own text reaches is a QUERY problem (decomposition is the lead); a hop neither reaches is
    neither, and is reported as such.
    """
    import json

    from llb.core.store_generations import generation_timestamp
    from llb.executor.runner_retrieval import _load_store
    from llb.rag.fusion_evidence.models import FOCUS_SLICE
    from llb.rag.multihop_probe import format_probe_report, parse_budgets, probe_multihop_hops

    cfg = load_config(
        config,
        goldset_path=goldset,
        retrieval_backend=retrieval_backend,
        retrieval_strategy=retrieval_strategy,
    )
    try:
        grid = parse_budgets(budgets)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    items = fusion_inputs.evidence_items(cfg, split)
    if not items:
        typer.echo("[error] the gold set selection is empty", err=True)
        raise typer.Exit(code=2)
    try:
        store = _load_store(cfg)
    except (FileNotFoundError, SystemExit) as exc:
        typer.echo(f"[error] the probed store is not built: {exc}", err=True)
        raise typer.Exit(code=2) from None
    report = probe_multihop_hops(
        store,
        items,
        budgets=grid,
        probe_depth=probe_depth,
        focus_slice=focus_slice or FOCUS_SLICE,
        lane=_lane_label(cfg),
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    default_dir = (
        cfg.data_dir / fusion_inputs.FUSION_EVIDENCE_METHOD / f"{generation_timestamp()}-hop-probe"
    )
    target = Path(out_dir) if out_dir else default_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / "probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "report.md").write_text(format_probe_report(report), encoding="utf-8")
    (target / "run_config.json").write_text(
        json.dumps(cfg.fingerprint(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    diagnosis = report["slices"][report["focus_slice"]]["diagnosis"]
    typer.echo(f"[probe-multihop-hops] {diagnosis['explanation']}: {diagnosis['reason']}")
    typer.echo(f"[probe-multihop-hops] report -> {target / 'report.md'}")
