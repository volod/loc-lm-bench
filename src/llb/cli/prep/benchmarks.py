"""Benchmark data-prep commands: agentic search tasks and BFCL tooling adaptation."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app


@app.command("prepare-agentic-search")
def prepare_agentic_search_cmd(
    corpus_root: Path = typer.Option(..., help="directory of .md/.txt source docs"),
    out: Path = typer.Option(
        ..., help="output agentic task set JSON (verified=false; review first)"
    ),
    top_k: int = typer.Option(8, min=1, help="max query terms per task kind (count + locate)"),
    limit: Optional[int] = typer.Option(None, help="cap the number of source documents"),
    merge_seed: bool = typer.Option(
        False, help="prepend the committed UA seed (samples/benchmarks/agentic_tasks_uk.json)"
    ),
) -> None:
    """agentic benchmark: build deterministic real-corpus agentic SEARCH tasks (count + locate) from a corpus."""
    import json as _json

    from llb.bench.agentic_tasks import build_from_corpus

    try:
        tasks = build_from_corpus(corpus_root, top_k=top_k, limit=limit)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    if merge_seed:
        seed = _json.loads(
            Path("samples/benchmarks/agentic_tasks_uk.json").read_text(encoding="utf-8")
        )
        tasks = list(seed) + tasks
    out.write_text(_json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(
        f"[prepare-agentic-search] {len(tasks)} tasks (verified=false; human verification gate before headline) -> {out}"
    )


@app.command("adapt-bfcl")
def adapt_bfcl_cmd(
    functions_file: Path = typer.Option(..., help="BFCL function-doc file (.json/.jsonl)"),
    out: Path = typer.Option(..., help="output tooling bundle JSON (verified=false; review first)"),
    answers_file: Optional[Path] = typer.Option(
        None, help="BFCL possible-answer file (.json/.jsonl); without it cases are no-call controls"
    ),
    limit: Optional[int] = typer.Option(None, help="cap the number of adapted cases"),
) -> None:
    """tooling benchmark: adapt the Berkeley Function-Calling Leaderboard (BFCL) cases into a UA tooling bundle."""
    import json as _json

    from llb.prep.tooling_sources import from_bfcl, load_jsonl_or_json

    entries = load_jsonl_or_json(functions_file)
    if limit is not None:
        entries = entries[:limit]
    answers = load_jsonl_or_json(answers_file) if answers_file else None
    bundle = from_bfcl(entries, answers)
    out.write_text(_json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(
        f"[adapt-bfcl] {len(bundle['cases'])} cases over {len(bundle['tools'])} tools "
        f"(verified=false; translate + human verification gate before headline) -> {out}"
    )
