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


@app.command("prepare-agentic-long-transcript")
def prepare_agentic_long_transcript_cmd(
    out: Path = typer.Option(
        ..., help="output multi-step medium-observation agentic task set JSON"
    ),
    from_search_tasks: Optional[Path] = typer.Option(
        None,
        help="optional fat-observation search task JSON to shrink into medium observations "
        "(preferred for CUDA keep_last_n long-transcript evidence)",
    ),
    max_match_docs: int = typer.Option(6, min=1, help="matching docs kept per shrunk search task"),
    max_other_docs: int = typer.Option(6, min=0, help="non-matching filler docs kept per task"),
    max_doc_chars: int = typer.Option(180, min=20, help="max chars kept per planted corpus doc"),
    n_db: int = typer.Option(0, min=0, help="number of pipeline-db tasks (synthetic; default 0)"),
    n_copy: int = typer.Option(0, min=0, help="number of pipeline-copy tasks (synthetic)"),
    n_sum: int = typer.Option(0, min=0, help="number of pipeline-sum tasks (synthetic)"),
    depth: int = typer.Option(4, min=2, help="base pipeline depth for synthetic tasks"),
    pad_chars: int = typer.Option(160, min=0, help="UA filler chars for synthetic file payloads"),
) -> None:
    """Build medium-observation tasks for the keep_last_n long-transcript lane.

    Prefer ``--from-search-tasks`` over a real search set (count/locate) so live models exercise
    the keep window on a shape they already complete; synthetic file/db pipelines are for CI.
    """
    import json as _json

    from llb.bench.agentic_long_transcript import (
        build_long_transcript_from_search_tasks,
        build_long_transcript_tasks,
    )

    tasks: list[dict[str, object]] = []
    if from_search_tasks is not None:
        source = _json.loads(from_search_tasks.read_text(encoding="utf-8"))
        tasks.extend(
            build_long_transcript_from_search_tasks(
                source,
                max_match_docs=max_match_docs,
                max_other_docs=max_other_docs,
                max_doc_chars=max_doc_chars,
            )
        )
    if n_db or n_copy or n_sum:
        tasks.extend(
            build_long_transcript_tasks(
                n_db=n_db, n_copy=n_copy, n_sum=n_sum, depth=depth, pad_chars=pad_chars
            )
        )
    if not tasks:
        typer.echo(
            "[error] no tasks built; pass --from-search-tasks and/or n_db/n_copy/n_sum > 0",
            err=True,
        )
        raise typer.Exit(code=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"[prepare-agentic-long-transcript] {len(tasks)} tasks -> {out}")


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
