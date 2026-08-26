"""The three-lane answer-validation comparison and the committed gate fixture."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED

DEFAULT_LANES = "off,pydantic,pydantic+ontology"


@app.command("compare-answer-validation")
def compare_answer_validation_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    model: Optional[str] = typer.Option(None, help="model name (Ollama tag or HF repo id)"),
    backend: Optional[str] = typer.Option(None, help="ollama | vllm | llamacpp"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    split: str = typer.Option("final", help="gold split(s) to score; comma-separated pools them"),
    limit: Optional[int] = typer.Option(None, help="cap the number of eval items"),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        help="completion budget for EVERY lane. The declared envelope is several times longer "
        "than a short free-text answer, so a run left at a free-text budget measures its own cap "
        "rather than the model; the envelope evidence recommends 768 on this roster",
    ),
    lanes: str = typer.Option(
        DEFAULT_LANES,
        help="comma-separated validation lanes; the FIRST must be `off` (the shipped free-text "
        "path, and the comparison's baseline). `pydantic` is the typed envelope alone; "
        "`pydantic+ontology` adds the signed-axiom answer gate",
    ),
    axioms: Optional[Path] = typer.Option(
        None,
        "--axioms",
        help="SIGNED axiom file the `pydantic+ontology` lane enables; an unsigned file is refused",
    ),
    ledger: Optional[Path] = typer.Option(
        None,
        "--ledger",
        help="the corpus extraction.jsonl (or its draft bundle dir) the answers are checked "
        "against, scoped per case to the retrieved chunks",
    ),
    include_drafted: bool = typer.Option(
        False,
        "--include-drafted",
        help="score a DRAFTED ledger no reviewer has accepted; never use it for a leaderboard run",
    ),
    resamples: int = typer.Option(DEFAULT_RESAMPLES, min=0, help="bootstrap resamples"),
    confidence: float = typer.Option(DEFAULT_CONFIDENCE, min=0.5, max=0.999, help="CI level"),
    seed: int = typer.Option(DEFAULT_SEED, help="bootstrap resampling seed"),
    out_dir: Optional[Path] = typer.Option(
        None, help="artifact dir (default: $DATA_DIR/answer-validation/<timestamp>/)"
    ),
) -> None:
    """Is semantic validation of RAG answers worth its cost on this corpus?

    Scores the identical item set end to end under each validation lane and reports what the gate
    STOPPED against what it wrongly REFUSED: the catch and false-rejection rates per axiom class,
    the objective delta against the ungated baseline on the items every lane answered, and the
    repair round trip's cost in tokens and wall clock.
    """
    from llb.eval.answer_validation.run import parse_lanes, run_answer_validation

    cfg = load_config(
        config, model=model, backend=backend, goldset_path=goldset, max_tokens=max_tokens
    )
    try:
        selected = parse_lanes(lanes)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    splits = [name.strip() for name in split.split(",") if name.strip()]
    if not splits:
        typer.echo("[error] name at least one gold split", err=True)
        raise typer.Exit(code=2)
    try:
        run = run_answer_validation(
            cfg,
            selected,
            axioms=axioms,
            ledger=ledger,
            splits=splits,
            limit=limit,
            resamples=resamples,
            confidence=confidence,
            seed=seed,
            out_dir=out_dir,
            verified_only=not include_drafted,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    for reading in run.report["readings"]:
        typer.echo(f"[compare-answer-validation] {reading['decision']}: {reading['reason']}")
    for verdict in run.report["axiom_classes"]:
        typer.echo(
            f"[compare-answer-validation] {verdict['axiom_class']}: {verdict['decision']} "
            f"-- {verdict['reason']}"
        )
    typer.echo(f"[compare-answer-validation] report -> {run.paths['report']}")


@app.command("check-answer-gate")
def check_answer_gate_cmd(
    fixture: Optional[Path] = typer.Option(
        None, help="fixture JSON (default: the committed samples/benchmarks fixture)"
    ),
) -> None:
    """Run the committed adversarial fixture through the ontology answer gate.

    Reports the catch rate per axiom class over the planted violating answers and the
    false-rejection rate over the correct answers a naive checker would refuse. Exits non-zero when
    a planted violation is missed or a scope case is wrongly refused; the false-rejection rate is
    REPORTED, never gated on, because its right value is a measurement rather than a target.
    """
    from llb.eval.answer_validation.fixture import fixture_report, load_fixture, run_fixture

    outcomes = run_fixture(load_fixture(fixture))
    report = fixture_report(outcomes)
    for line in report.lines():
        typer.echo(line)
    if not report.all_caught:
        typer.echo("[error] a planted violation was not caught by its axiom class", err=True)
        raise typer.Exit(code=1)
    if report.scope_failures:
        typer.echo(
            "[error] the gate refused a contradiction outside the retrieved context", err=True
        )
        raise typer.Exit(code=1)
