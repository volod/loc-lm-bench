"""RAG prompt-system comparison prompt-system CLI -- prepare, review, and compare RAG prompt systems."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config


@app.command("prompt-system-prepare")
def prompt_system_prepare_cmd(
    corpus_root: Path = typer.Option(..., help="directory of .md/.txt source documents"),
    out_dir: Optional[Path] = typer.Option(
        None,
        help="stable output dir for sample/review artifacts (default: DATA_DIR/prompt-system/<ts>)",
    ),
    context_window: int = typer.Option(8192, help="target model context window (tokens)"),
    chunk_tokens: int = typer.Option(1024, help="tokens reserved for retrieved chunks"),
    answer_tokens: int = typer.Option(512, help="tokens reserved for the answer"),
    max_passages: int = typer.Option(12, help="max anthology passages to select from the corpus"),
    role: Optional[str] = typer.Option(None, help="override the generated system role text"),
    instruction: Optional[str] = typer.Option(
        None, help="override the generated grounding instruction text"
    ),
) -> None:
    """Ingest a corpus and generate budget-fitted, reviewable RAG prompt-system candidates."""
    from llb.prompt_system.pipeline import MANIFEST_FILE, prepare_prompt_system
    from llb.prompt_system.template import TemplateFields

    cfg = load_config(None)
    base_fields = None
    if role is not None or instruction is not None:
        defaults = TemplateFields()
        base_fields = TemplateFields(
            role=role or defaults.role,
            instruction=instruction or defaults.instruction,
        )
    run = prepare_prompt_system(
        corpus_root,
        data_dir=cfg.data_dir,
        out_dir=out_dir,
        base_fields=base_fields,
        context_window=context_window,
        chunk_tokens=chunk_tokens,
        answer_tokens=answer_tokens,
        max_passages=max_passages,
    )
    typer.echo(
        f"[prompt-system-prepare] {len(run.candidates)} candidates "
        f"(budget {run.budget.prompt_budget} tok, {len(run.corpus.anthology)} passages) "
        f"-> {run.run_dir / MANIFEST_FILE}"
    )
    for candidate in run.candidates:
        dropped = sum(s["n_dropped"] for s in candidate.dropped_context["sections"])
        typer.echo(
            f"  {candidate.prompt_system_id}  size={candidate.fields.anthology_size} "
            f"meta={candidate.fields.metadata_density} graph={candidate.fields.graph_reference_style} "
            f"used={candidate.used_tokens}tok dropped={dropped}"
        )


@app.command("prompt-system-review")
def prompt_system_review_cmd(
    run_dir: Path = typer.Option(..., help="a prompt-system run dir (holds candidates.json)"),
    action: str = typer.Option(
        "summary", help="summary | approve | pin | reject (the last three need --id)"
    ),
    candidate_id: Optional[str] = typer.Option(None, "--id", help="prompt-system id to act on"),
    note: str = typer.Option("", help="optional reviewer note"),
) -> None:
    """Review prompt-system candidates: summarize, or approve/pin/reject one by id."""
    from llb.prompt_system.pipeline import CANDIDATES_FILE
    from llb.prompt_system.review import (
        approve,
        load_candidates,
        pin,
        reject,
        save_candidates,
        summarize_review,
    )

    path = run_dir / CANDIDATES_FILE
    candidates = load_candidates(path)
    if action == "summary":
        summary = summarize_review(candidates)
        typer.echo(f"[prompt-system-review] n={summary.n} by_status={summary.by_status}")
        if summary.pinned:
            typer.echo(f"  pinned: {', '.join(summary.pinned)}")
        return
    actions = {"approve": approve, "pin": pin, "reject": reject}
    if action not in actions or candidate_id is None:
        typer.echo(
            "[error] approve/pin/reject need --id <prompt-system-id>; or use --action summary",
            err=True,
        )
        raise typer.Exit(code=2)
    target = next((c for c in candidates if c.prompt_system_id == candidate_id), None)
    if target is None:
        typer.echo(f"[error] no candidate with id {candidate_id}", err=True)
        raise typer.Exit(code=2)
    actions[action](target, note)
    save_candidates(candidates, path)
    typer.echo(f"[prompt-system-review] {action} {candidate_id} -> {target.status}")


@app.command("prompt-system-compare")
def prompt_system_compare_cmd(
    model: str = typer.Option(..., help="the candidate model to compare across prompt systems"),
    lane: str = typer.Option("rag", help="rag | agentic"),
    harness: Optional[str] = typer.Option(
        None, help="restrict to one harness (loop/langgraph/...)"
    ),
) -> None:
    """RAG prompt-system comparison: rank ONE model's runs across prompt-system ids."""
    from llb.board.data import prompt_system_comparison, rag_prompt_system_comparison

    cfg = load_config(None)
    if lane == "rag":
        rows, table, ids = rag_prompt_system_comparison(cfg.data_dir, model)
        label = "RAG"
    elif lane == "agentic":
        rows, table, ids = prompt_system_comparison(cfg.data_dir, model, harness)
        label = "agentic"
    else:
        typer.echo("[error] --lane must be rag or agentic", err=True)
        raise typer.Exit(code=2)
    if not rows:
        typer.echo(
            f"[prompt-system-compare] no prompt-system-tagged {label} runs for model '{model}'"
        )
        raise typer.Exit(code=2)
    typer.echo(f"[prompt-system-compare] lane={lane} model={model} prompt_systems={', '.join(ids)}")
    typer.echo(table)
