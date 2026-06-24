"""Gold-set and corpus preparation commands."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app


@app.command("prepare-goldset")
def prepare_goldset_cmd(
    corpus_root: Path = typer.Option(..., help="directory of .md/.txt source docs"),
    model: str = typer.Option(..., help="litellm model id (needs a provider key in .env)"),
    n_per_doc: int = typer.Option(3, min=1, help="draft this many QA pairs per document"),
    out: Path = typer.Option(..., help="output gold set JSONL (items are verified=false)"),
) -> None:
    """Draft review-ready (question, answer, exact span) gold items from a corpus via a frontier LLM."""
    from llb.prep.frontier import prepare_goldset

    items = prepare_goldset(corpus_root, model=model, n_per_doc=n_per_doc, out_path=out)
    typer.echo(
        f"[prepare-goldset] {len(items)} drafted items (verified=false; review before scoring) -> {out}"
    )


@app.command("prepare-synthetic-corpus")
def prepare_synthetic_corpus_cmd(
    topics_file: Path = typer.Option(..., help="text file: one synthetic-doc topic per line"),
    planter: str = typer.Option(..., help="litellm model that PLANTS the labels"),
    judge: str = typer.Option(..., help="the eval judge model (must differ from the planter)"),
    out_dir: Path = typer.Option(..., help="output dir for docs + planted labels"),
    n_labels: int = typer.Option(
        3, min=1, help="planted labels per document (per kind in TA mode)"
    ),
    text_analysis: bool = typer.Option(
        False,
        "--text-analysis/--qa",
        help="emit RICHER per-kind text-analysis PlantedLabelRecords (M5.0) instead of QA labels",
    ),
    kinds: Optional[str] = typer.Option(
        None, help="comma-separated text-analysis kinds (default: the objective sub-tasks)"
    ),
) -> None:
    """Generate synthetic docs with structured planted labels (planter must differ from judge).

    Default emits QA-style key_fact labels (RAG planted set). `--text-analysis` emits the full
    per-kind text-analysis taxonomy (key_fact/entity/topic/trend/risk/decision/...) as
    PlantedLabelRecords for the M5.0 scored text-analysis runner.
    """
    topics = [t.strip() for t in topics_file.read_text(encoding="utf-8").splitlines() if t.strip()]
    if not topics:
        typer.echo(f"[error] no topics found in {topics_file}", err=True)
        raise typer.Exit(code=2)

    if text_analysis:
        from llb.prep.text_analysis_corpus import DEFAULT_KINDS, prepare_text_analysis_corpus

        chosen = tuple(k.strip() for k in kinds.split(",") if k.strip()) if kinds else DEFAULT_KINDS
        try:
            docs, records = prepare_text_analysis_corpus(
                topics,
                planter_model=planter,
                judge_model=judge,
                kinds=chosen,
                n_per_kind=n_labels,
                out_dir=out_dir,
            )
        except ValueError as exc:
            typer.echo(f"[error] {exc}", err=True)
            raise typer.Exit(code=2)
        typer.echo(
            f"[prepare-synthetic-corpus] text-analysis: {len(docs)} docs, {len(records)} planted "
            f"labels across {len(chosen)} kinds (planter={planter} != judge={judge}) -> {out_dir}"
        )
        return

    from llb.prep.frontier import prepare_synthetic_corpus

    docs, items = prepare_synthetic_corpus(
        topics, planter_model=planter, judge_model=judge, n_labels=n_labels, out_dir=out_dir
    )
    typer.echo(
        f"[prepare-synthetic-corpus] {len(docs)} docs, {len(items)} planted items "
        f"(planter={planter} != judge={judge}) -> {out_dir}"
    )


@app.command("cross-check-goldset")
def cross_check_goldset_cmd(
    goldset: Path = typer.Option(..., help="drafted gold set JSONL (verified=false)"),
    corpus: Path = typer.Option(..., help="source corpus dir for the drafted items"),
    model: str = typer.Option(..., help="SECOND-frontier verifier (must differ from the drafter)"),
    out: Optional[Path] = typer.Option(
        None, help="cross-check report JSON (default beside goldset)"
    ),
) -> None:
    """M5.6 verified-data gate: a second frontier re-confirms grounding/support/answerability."""
    import json

    from llb.goldset.schema import load_goldset
    from llb.prep.cross_check import (
        cross_check_goldset,
        load_doc_texts,
        second_frontier_verify,
    )

    items = load_goldset(goldset)
    report = cross_check_goldset(items, load_doc_texts(corpus), second_frontier_verify(model))
    out_path = out or goldset.with_name(f"{goldset.stem}.cross_check.json")
    out_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(
        f"[cross-check] {report.n_passed}/{len(items)} items passed the gate "
        f"(verified=false until MH.5) -> {out_path}"
    )


@app.command("prepare-goldset-draft")
def prepare_goldset_draft_cmd(
    corpus_root: Path = typer.Option(..., help="directory of .md/.txt source docs"),
    model: str = typer.Option(
        ..., help="model id (local endpoint tag, or litellm route for frontier)"
    ),
    endpoint: str = typer.Option(
        "local", help="local (OpenAI-compatible, no egress) | frontier (litellm, opt-in egress)"
    ),
    base_url: Optional[str] = typer.Option(
        None, help="local endpoint base URL (default: Ollama OpenAI-compatible /v1)"
    ),
    max_items: int = typer.Option(60, min=1, help="upper bound on drafted QA items"),
    seed: int = typer.Option(13, help="deterministic sampling/split seed"),
    out_dir: Optional[Path] = typer.Option(
        None, help="output bundle dir (default: $DATA_DIR/prepare-goldset/<timestamp>/)"
    ),
) -> None:
    """M4.4: ontology-assisted DRAFT gold set from a corpus (verified=false; review before scoring)."""
    from llb.prep.ontology import EndpointConfig, draft_goldset

    try:
        cfg = (
            EndpointConfig(kind=endpoint, model=model, base_url=base_url)
            if base_url
            else EndpointConfig(kind=endpoint, model=model)
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    result = draft_goldset(corpus_root, cfg, max_items=max_items, seed=seed, out_dir=out_dir)
    typer.echo(
        f"[prepare-goldset-draft] {len(result.items)} drafted items (verified=false; "
        f"endpoint={endpoint}, egress={cfg.egress}) -> {result.out_dir}"
    )
