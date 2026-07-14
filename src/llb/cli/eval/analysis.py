"""Context-position probe and miss-analysis / external-RAG scoring commands."""

import json
from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config


@app.command("probe-context-position")
def probe_context_position_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    model: Optional[str] = typer.Option(None, help="model name (Ollama tag or HF repo id)"),
    backend: Optional[str] = typer.Option(None, help="ollama | vllm | llamacpp"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    k: int = typer.Option(5, min=3, help="fixed context size (gold at head/middle/tail)"),
    split: str = typer.Option("final", help="gold split to probe"),
    limit: Optional[int] = typer.Option(None, help="cap the number of probed items"),
    candidate_depth: Optional[int] = typer.Option(
        None, help="retrieval depth scanned for the gold chunk + distractors (default 50)"
    ),
    max_model_len: Optional[int] = typer.Option(
        None, help="vLLM/llama.cpp served context window (overrides the config)"
    ),
    out_dir: Optional[Path] = typer.Option(
        None, help="probe output dir (default: $DATA_DIR/context-position/<timestamp>)"
    ),
) -> None:
    """Lost-in-the-middle probe (rerank-context-order): place each item's gold chunk at the
    head, middle, and tail of a fixed-k context of real retrieved distractors, score every
    position, and recommend a per-model `context_order` with bootstrap CIs."""
    from llb.bench.common import new_run_timestamp
    from llb.core.contracts import ChatMessage
    from llb.eval.position_probe import DEFAULT_CANDIDATE_DEPTH, run_probe
    from llb.eval.position_probe_report import render_report, write_probe
    from llb.executor.runner_backend import _make_launcher
    from llb.executor.runner_retrieval import _load_store
    from llb.goldset.schema import load_goldset

    cfg = load_config(
        config, model=model, backend=backend, goldset_path=goldset, max_model_len=max_model_len
    )
    items = [it for it in load_goldset(cfg.goldset_path) if it.verified and it.split == split]
    items.sort(key=lambda it: it.id)
    if limit is not None:
        items = items[:limit]
    if not items:
        typer.echo(f"[probe-context-position] no verified '{split}' items to probe", err=True)
        raise typer.Exit(code=2)
    store = _load_store(cfg)
    launcher = _make_launcher(cfg)
    with launcher as served:

        def chat(messages: list[ChatMessage]) -> tuple[str, str | None]:
            result = served.chat(
                messages,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                timeout=cfg.request_timeout_s,
            )
            return result.text or "", result.error

        report = run_probe(
            items,
            store,
            chat,
            model=cfg.model,
            backend=cfg.backend,
            k=k,
            candidate_depth=candidate_depth or DEFAULT_CANDIDATE_DEPTH,
        )
    _, run_ts = new_run_timestamp()
    out = out_dir or (cfg.data_dir / "context-position" / run_ts)
    paths = write_probe(report, out)
    typer.echo(render_report(report))
    typer.echo(f"[probe-context-position] report -> {paths['report']}")
    typer.echo(f"[probe-context-position] cases -> {paths['cases']}")


@app.command("analyze-misses")
def analyze_misses_cmd(
    run_dir: Path = typer.Option(
        ..., "--run-dir", help="finalized run-eval bundle whose misses to explain"
    ),
    goldset: Optional[Path] = typer.Option(
        None, help="goldset JSONL the run scored (default: the bundle manifest's goldset_path)"
    ),
    miss_threshold: Optional[float] = typer.Option(
        None,
        "--miss-threshold",
        help="objective score below which a scoreable (status=ok) case counts as a miss "
        "(default 0.5)",
    ),
    probe_top_k: Optional[str] = typer.Option(
        None,
        "--probe-top-k",
        help="comma-separated retrieval depths (e.g. 3,8): re-run ONLY the miss subset at each "
        "depth to confirm or reject the retrieval hypothesis (launches the model backend)",
    ),
    out_dir: Optional[Path] = typer.Option(
        None, help="analysis output dir (default: $DATA_DIR/miss-analysis/<timestamp>)"
    ),
) -> None:
    """Explain one run's wrong answers: classify every miss (retrieval / generation / refusal /
    format artifact / judge disagreement), cluster by document, topic, and question type, and
    write ranked, evidence-backed recommendations that `llb recommend` folds into its summary."""
    from llb.board.miss_analysis.classify import analyze_run
    from llb.board.miss_analysis.load import load_item_provenance
    from llb.board.miss_analysis.model import DEFAULT_MISS_THRESHOLD
    from llb.board.miss_analysis.recommendations import refresh_recommendations
    from llb.board.miss_analysis.report import analysis_out_dir, write_analysis
    from llb.board.miss_probe import parse_probe_depths, run_probes
    from llb.board.runs import load_run_records
    from llb.core.paths import resolve_data_dir
    from llb.goldset.schema import load_goldset

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        typer.echo(f"[analyze-misses] no manifest.json in {run_dir}", err=True)
        raise typer.Exit(code=2)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config") or {}
    goldset_path = goldset or Path(str(config.get("goldset_path", "")))
    if not str(goldset_path) or not goldset_path.is_file():
        typer.echo(
            f"[analyze-misses] goldset not found: '{goldset_path}' "
            "(the bundle's recorded goldset moved? pass --goldset)",
            err=True,
        )
        raise typer.Exit(code=2)
    items = load_goldset(goldset_path)
    provenance = load_item_provenance(goldset_path)

    # Comparable sibling runs (same split + case count, board dedup rules) back the
    # "try the named alternative model" recommendation with measured numbers.
    alternatives = [
        (record.result.model, record.result.objective_score)
        for record in load_run_records(run_dir.parent)
        if record.split == manifest.get("split")
        and record.result.n_cases == int(manifest.get("n_cases", -1))
    ]
    threshold = miss_threshold if miss_threshold is not None else DEFAULT_MISS_THRESHOLD
    analysis = analyze_run(
        run_dir, items, threshold=threshold, provenance=provenance, alternatives=alternatives
    )

    if probe_top_k and analysis.misses:
        depths = parse_probe_depths(probe_top_k)
        analysis.probes = run_probes(manifest, analysis.misses, items, depths)
        refresh_recommendations(analysis, alternatives=alternatives)
    elif probe_top_k:
        typer.echo("[analyze-misses] no misses to probe; skipping --probe-top-k")

    out = out_dir or analysis_out_dir(resolve_data_dir())
    paths = write_analysis(analysis, out)
    counted = ", ".join(f"{cls}={n}" for cls, n in analysis.class_counts.items() if n) or "none"
    typer.echo(
        f"[analyze-misses] {len(analysis.misses)} of {analysis.n_cases} cases missed ({counted})"
    )
    for rank, rec in enumerate(analysis.recommendations, 1):
        typer.echo(f"[analyze-misses] {rank}. {rec['line']}")
    typer.echo(f"[analyze-misses] report -> {paths['report']}")
    typer.echo(f"[analyze-misses] misses -> {paths['misses']}")


@app.command("score-external-rag")
def score_external_rag_cmd(
    answers: Path = typer.Option(
        ..., "--answers", help="answered goldset JSONL from an external or closed RAG system"
    ),
    csv_out: Optional[Path] = typer.Option(
        None, help="detailed per-row CSV path (default: <answers>.csv)"
    ),
    report_out: Optional[Path] = typer.Option(
        None, help="Markdown report path (default: <answers>.report.md)"
    ),
    answer_field: Optional[str] = typer.Option(
        None, help="answer field to score (default: auto: llm_answer, predicted_answer, ...)"
    ),
    sources_field: Optional[str] = typer.Option(
        None, help="source-list field to flatten (default: auto: llm_sources, sources, ...)"
    ),
    error_field: Optional[str] = typer.Option(
        None, help="error field (default: auto: llm_error, error)"
    ),
    source_limit: int = typer.Option(3, min=0, help="number of top sources to flatten into CSV"),
    label: Optional[str] = typer.Option(None, help="system label in the report"),
    strip_source_footer: bool = typer.Option(
        True,
        "--strip-source-footer/--keep-source-footer",
        help="strip a trailing Source:/Dzherelo: footer before objective scoring",
    ),
    start: Optional[int] = typer.Option(
        None, "--start", min=1, help="start review at 1-based row number"
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="confirmation-gated restart: clear JSONL human fields before reviewing",
    ),
    source_map: Optional[Path] = typer.Option(
        None,
        "--source-map",
        help="mapping sidecar (json/jsonl/csv) from provider article_id/url/article_title to "
        "corpus doc_id [+ char range]; enables the source-span audit columns",
    ),
) -> None:
    """Interactively score an external RAG JSONL; finalize CSV + report when complete."""
    from llb.scoring.external_rag_session.session import run_external_rag_session

    try:
        run_external_rag_session(
            answers,
            csv_out=csv_out,
            report_out=report_out,
            answer_field=answer_field,
            sources_field=sources_field,
            error_field=error_field,
            source_limit=source_limit,
            strip_source_footer=strip_source_footer,
            label=label,
            start=start,
            clear=clear,
            source_map=source_map,
        )
    except ValueError as exc:
        typer.echo(f"[score-external-rag] {exc}", err=True)
        raise typer.Exit(code=2)
