"""Per-hop multi-hop retrievability probe: is a missing hop a budget or a query problem?"""

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.cli.rag import fusion_inputs
from llb.cli.rag.query_prep_endpoint import (
    parse_query_prep_steps,
    resolve_query_prep_endpoint,
)
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
    query_prep: Optional[str] = typer.Option(
        None,
        "--query-prep",
        help="paired raw/prepared probe through comma-separated query-prep steps",
    ),
    query_prep_model: Optional[str] = typer.Option(
        None, help="local model for rewrite/hyde/decompose query-prep steps"
    ),
    query_prep_backend: Optional[str] = typer.Option(
        None, help="local backend for model query prep: ollama | vllm | llamacpp"
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
    from llb.executor.runner_retrieval import build_query_prep
    from llb.rag.multihop_probe import (
        compare_multihop_query_prep,
        format_probe_report,
        format_query_prep_probe_report,
        parse_budgets,
        probe_multihop_hops,
    )

    cli_steps = parse_query_prep_steps(query_prep)
    cfg = load_config(
        config,
        goldset_path=goldset,
        retrieval_backend=retrieval_backend,
        retrieval_strategy=retrieval_strategy,
        query_prep=cli_steps if query_prep is not None else None,
    )
    steps = list(cfg.query_prep)
    try:
        grid = parse_budgets(budgets)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    items = fusion_inputs.evidence_items(cfg, split)
    if not items:
        typer.echo("[error] the gold set selection is empty", err=True)
        raise typer.Exit(code=2)
    selected_focus = focus_slice or FOCUS_SLICE
    if not any(item.question_type == selected_focus for item in items):
        typer.echo(f"[error] probe focus slice is empty: {selected_focus}", err=True)
        raise typer.Exit(code=2)
    try:
        store = _load_store(cfg)
    except (FileNotFoundError, SystemExit) as exc:
        typer.echo(f"[error] the probed store is not built: {exc}", err=True)
        raise typer.Exit(code=2) from None
    try:
        endpoint_cfg, launcher, endpoint = resolve_query_prep_endpoint(
            cfg,
            steps,
            model=query_prep_model,
            backend=query_prep_backend,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    lane = _lane_label(cfg)
    report: Any
    launcher_context = launcher if launcher is not None else nullcontext(None)
    with launcher_context as active:
        if steps:
            pipeline = build_query_prep(endpoint_cfg, store, active)
            assert pipeline is not None
            report = compare_multihop_query_prep(
                store,
                items,
                pipeline,
                budgets=grid,
                probe_depth=probe_depth,
                focus_slice=selected_focus,
                lane=lane,
                resamples=resamples,
                confidence=confidence,
                seed=seed,
            )
            if endpoint is not None:
                report["endpoint"] = endpoint
            rendered = format_query_prep_probe_report(report)
        else:
            report = probe_multihop_hops(
                store,
                items,
                budgets=grid,
                probe_depth=probe_depth,
                focus_slice=selected_focus,
                lane=lane,
                resamples=resamples,
                confidence=confidence,
                seed=seed,
            )
            rendered = format_probe_report(report)
    default_dir = (
        cfg.data_dir / fusion_inputs.FUSION_EVIDENCE_METHOD / f"{generation_timestamp()}-hop-probe"
    )
    target = Path(out_dir) if out_dir else default_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / "probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "report.md").write_text(rendered, encoding="utf-8")
    (target / "run_config.json").write_text(
        json.dumps(endpoint_cfg.fingerprint(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if steps:
        query_cohort = report["conversion"]["cohorts"]["query"]
        budget_cohort = report["conversion"]["cohorts"]["budget"]
        typer.echo(
            f"[probe-multihop-hops] query conversion "
            f"{query_cohort['all_spans_gained']}/{query_cohort['n']}; budget cost "
            f"{budget_cohort['span_coverage_regressed']}/{budget_cohort['n']}"
        )
    else:
        diagnosis = report["slices"][report["focus_slice"]]["diagnosis"]
        typer.echo(f"[probe-multihop-hops] {diagnosis['explanation']}: {diagnosis['reason']}")
    typer.echo(f"[probe-multihop-hops] report -> {target / 'report.md'}")
