"""Composite headline, MCP tool server, chain-context, and reliability commands."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import best_effort_gpu_readers, load_config


@app.command("bench-composite")
def bench_composite_cmd(
    allow_unverified: bool = typer.Option(
        False, help="diagnostic only: allow runs not stamped with --data-verified"
    ),
    allow_missing_ci: bool = typer.Option(
        False, help="diagnostic only: allow categories without a reloadable per-case CI series"
    ),
) -> None:
    """Render the guarded category composite headline from persisted category runs."""
    from llb.board.categories import load_category_composite
    from llb.scoring.composite_format import format_composite_issues, format_composite_rows

    cfg = load_config(None)
    rows, issues = load_category_composite(
        cfg.data_dir,
        require_verified=not allow_unverified,
        require_ci=not allow_missing_ci,
    )
    if rows:
        typer.echo(format_composite_rows(rows))
        return
    typer.echo(format_composite_issues(issues) or "[bench-composite] no category runs found")
    raise typer.Exit(code=2)


@app.command("serve-tools-mcp")
def serve_tools_mcp_cmd(
    catalog: Path = typer.Option(
        Path("samples/benchmarks/tooling_cases_uk.json"), help="tooling bundle (tools + cases JSON)"
    ),
    name: str = typer.Option("loc-lm-bench-tools", help="MCP server name"),
) -> None:
    """Serve the same tool catalog over the official MCP SDK (stdio); needs the [mcp] extra."""
    from llb.bench.mcp_server import load_catalog, mcp_tool_specs, serve_stdio

    tool_catalog = load_catalog(catalog)
    typer.echo(
        f"[serve-tools-mcp] serving {len(mcp_tool_specs(tool_catalog))} tools over MCP stdio "
        f"(name={name}); connect an MCP client to its stdin/stdout"
    )
    serve_stdio(tool_catalog, name=name)


@app.command("bench-chain-context")
def bench_chain_context_cmd(
    chains: Path = typer.Option(
        Path("samples/goldsets/chain_context_uk_v1/chains.jsonl"),
        help="verified chain set (JSONL)",
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    corpus: Path = typer.Option(
        Path("samples/goldsets/chain_context_uk_v1/corpus"),
        help="corpus dir; a RAG store is built in-process for per-step retrieval",
    ),
    index_dir: Optional[Path] = typer.Option(
        None, help="load a prebuilt RAG store instead of building one from --corpus"
    ),
    policies: str = typer.Option(
        "fresh,history,summary,roles", help="comma-separated context policies to compare"
    ),
    top_k: int = typer.Option(4, help="retrieved chunks per step"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint (skips launching)"
    ),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM/llama.cpp served context window"),
    data_verified: bool = typer.Option(
        False,
        help="stamp the run as human verification gate-verified for composite-headline eligibility",
    ),
    verification_ref: Optional[str] = typer.Option(
        None,
        help="path or label for the verification worksheet, sample manifest, or accepted ledger",
    ),
) -> None:
    """Rank context-management policies for one fixed model over a verified chain set."""
    from llb.bench.chain_context import ChainContextRun, load_chains_file, run_chain_context
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import drive_with_backend
    from llb.rag.store import RagStore

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    chain_set = load_chains_file(chains)
    policy_list = [p.strip() for p in policies.split(",") if p.strip()]
    retriever = (
        RagStore.load(index_dir)
        if index_dir is not None
        else RagStore.build(corpus, embedding_model=cfg.embedding_model)
    )
    vram_reader, pid_reader = best_effort_gpu_readers()

    def run(complete: LLMComplete) -> ChainContextRun:
        return run_chain_context(
            chain_set,
            model=model,
            backend=backend,
            retriever=retriever,
            complete=complete,
            policies=policy_list,
            k=top_k,
            data_dir=cfg.data_dir,
            data_verified=data_verified,
            verification_ref=verification_ref,
        )

    result = drive_with_backend(
        cfg, run, base_url=base_url, vram_reader=vram_reader, pid_usage_reader=pid_reader
    )
    typer.echo(result.table)
    typer.echo(result.recommendation)
    for report in result.reports:
        if report.paths is not None:
            typer.echo(f"[bench-chain-context]   {report.policy:<8} -> {report.paths['manifest']}")


@app.command("bench-reliability")
def bench_reliability_cmd(
    run_dir: Path = typer.Option(..., help="a run bundle dir (contains scores.jsonl)"),
) -> None:
    """Aggregate a run's typed failure taxonomy into a first-class reliability score."""
    from llb.scoring.reliability import read_case_statuses, reliability_report

    report = reliability_report(read_case_statuses(run_dir))
    typer.echo(
        f"[bench-reliability] reliability={report['reliability']:.3f} ({report['n_ok']}/{report['n']})"
    )
    for status, count in report["failures"].items():
        typer.echo(f"[bench-reliability]   {status:<20} {count}")
