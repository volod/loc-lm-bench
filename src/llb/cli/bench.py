"""Benchmark category commands, each rendered under its own Tier."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import best_effort_gpu_readers, load_config


@app.command("bench-text-analysis")
def bench_text_analysis_cmd(
    bundle: Path = typer.Option(
        ..., help="synthetic text-analysis bundle dir (corpus/ + text_analysis_labels.jsonl)"
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint (skips launching)"
    ),
    limit: Optional[int] = typer.Option(None, help="cap the number of documents"),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM/llama.cpp served context window"),
    real_corpus: bool = typer.Option(
        False, help="the bundle is a REAL corpus (reported separately from synthetic)"
    ),
    judge_model: Optional[str] = typer.Option(
        None,
        help="opt-in gated judge for narrative/insight/long_doc quality (recorded alongside the "
        "objective headline, never folded in)",
    ),
    judge_rho: Optional[float] = typer.Option(
        None, help="calibration Spearman rho; the judge is used only when rho >= threshold (0.6)"
    ),
    judge_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of the judge endpoint"
    ),
    data_verified: bool = typer.Option(
        False,
        help="stamp the run as human verification gate-verified for composite-headline eligibility",
    ),
    verification_ref: Optional[str] = typer.Option(
        None,
        help="path or label for the verification worksheet, sample manifest, or accepted ledger",
    ),
) -> None:
    """Score a model's planted-label recovery under TIER_TEXT_ANALYSIS."""
    from llb.bench.common import LLMComplete, drive_with_backend
    from llb.bench.text_analysis import TextAnalysisRun, run_text_analysis

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    vram_reader, pid_reader = best_effort_gpu_readers()

    def run(complete: LLMComplete) -> TextAnalysisRun:
        return run_text_analysis(
            bundle,
            model=model,
            backend=backend,
            complete=complete,
            judge_model=judge_model,
            judge_rho=judge_rho,
            judge_base_url=judge_base_url,
            data_dir=cfg.data_dir,
            limit=limit,
            synthetic=not real_corpus,
            data_verified=data_verified,
            verification_ref=verification_ref,
        )

    result = drive_with_backend(
        cfg, run, base_url=base_url, vram_reader=vram_reader, pid_usage_reader=pid_reader
    )
    typer.echo(result.table)
    if result.judged_quality is not None:
        typer.echo(
            f"[bench-text-analysis] judged-quality (gated judge)={result.judged_quality:.3f}"
        )
    if result.paths is not None:
        typer.echo(f"[bench-text-analysis] manifest -> {result.paths['manifest']}")


@app.command("bench-security")
def bench_security_cmd(
    cases: Path = typer.Option(
        Path("samples/benchmarks/security_cases_uk.json"), help="security-case set (JSON array)"
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint (skips launching)"
    ),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM/llama.cpp served context window"),
    judge_model: Optional[str] = typer.Option(
        None,
        help="opt-in gated unsafe-content refusal-quality judge (recorded alongside ASR, never "
        "the headline)",
    ),
    judge_rho: Optional[float] = typer.Option(
        None, help="calibration Spearman rho; the judge is used only when rho >= threshold (0.6)"
    ),
    judge_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of the judge endpoint"
    ),
    data_verified: bool = typer.Option(
        False,
        help="stamp the run as human verification gate-verified for composite-headline eligibility",
    ),
    verification_ref: Optional[str] = typer.Option(
        None,
        help="path or label for the verification worksheet, sample manifest, or accepted ledger",
    ),
) -> None:
    """Score a model's objective ASR + refusal-appropriateness under TIER_SECURITY."""
    from llb.bench.common import LLMComplete, ThroughputMeter, drive_with_backend
    from llb.bench.security import SecurityRun, load_cases_file, run_security

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    security_cases = load_cases_file(cases)
    vram_reader, pid_reader = best_effort_gpu_readers()
    meter = ThroughputMeter()

    def run(complete: LLMComplete) -> SecurityRun:
        return run_security(
            security_cases,
            model=model,
            backend=backend,
            complete=complete,
            judge_model=judge_model,
            judge_rho=judge_rho,
            judge_base_url=judge_base_url,
            data_dir=cfg.data_dir,
            data_verified=data_verified,
            verification_ref=verification_ref,
            meter=meter,
        )

    result = drive_with_backend(
        cfg,
        run,
        base_url=base_url,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
        meter=meter,
    )
    s = result.score
    typer.echo(
        f"[bench-security] ASR={s.asr:.3f} defense={s.defense_rate:.3f} "
        f"refusal-appropriateness={s.refusal_appropriateness:.3f} (n_attacks={s.n_attacks})"
    )
    for family, asr in sorted(s.asr_by_family.items()):
        typer.echo(f"[bench-security]   {family:<22} ASR={asr:.3f}")
    if s.cross_language is not None:
        typer.echo(
            f"[bench-security] xlang-consistency={s.cross_language.consistency:.3f} "
            f"({s.cross_language.n_groups} groups)"
        )
    if s.bias_pairs is not None:
        typer.echo(
            f"[bench-security] bias-pair-consistency={s.bias_pairs.consistency:.3f} "
            f"({s.bias_pairs.n_pairs} pairs)"
        )
    if meter.calls:
        typer.echo(
            f"[bench-security] throughput={meter.tokens_per_s:.1f} tok/s over {meter.calls} calls "
            f"({meter.completion_tokens} completion tokens)"
        )
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-security] manifest -> {result.paths['manifest']}")


@app.command("bench-tooling")
def bench_tooling_cmd(
    catalog: Path = typer.Option(
        Path("samples/benchmarks/tooling_cases_uk.json"), help="tooling bundle (tools + cases JSON)"
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint (skips launching)"
    ),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM/llama.cpp served context window"),
    tool_protocol: str = typer.Option(
        "text",
        help="text (catalog-in-prompt, every backend) | native (OpenAI tools=, needs a running "
        "tool-capable endpoint via --base-url or Ollama)",
    ),
    data_verified: bool = typer.Option(
        False,
        help="stamp the run as human verification gate-verified for composite-headline eligibility",
    ),
    verification_ref: Optional[str] = typer.Option(
        None,
        help="path or label for the verification worksheet, sample manifest, or accepted ledger",
    ),
) -> None:
    """Score a model's call-only function-calling correctness under TIER_TOOLING."""
    from llb.backends.openai_client import make_client
    from llb.bench.common import LLMComplete, drive_with_backend
    from llb.bench.tooling import (
        TOOL_PROTOCOL_NATIVE,
        ToolCaller,
        ToolingRun,
        load_catalog_file,
        native_tool_caller,
        run_tooling,
    )

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    tool_catalog, cases = load_catalog_file(catalog)
    vram_reader, pid_reader = best_effort_gpu_readers()

    native_caller: Optional[ToolCaller] = None
    if tool_protocol == TOOL_PROTOCOL_NATIVE:
        # Native tools= needs a known running endpoint (no launch). Build its caller up front.
        endpoint = base_url or (
            f"{cfg.ollama_host.rstrip('/')}/v1" if backend == "ollama" else None
        )
        if endpoint is None:
            typer.echo(
                "[error] --tool-protocol native needs a running tool-capable endpoint "
                "(--base-url ... or --backend ollama)",
                err=True,
            )
            raise typer.Exit(code=2)
        native_caller = native_tool_caller(
            make_client(endpoint), model, timeout=cfg.request_timeout_s
        )

    def run(complete: LLMComplete) -> ToolingRun:
        return run_tooling(
            tool_catalog,
            cases,
            model=model,
            backend=backend,
            complete=complete,
            caller=native_caller,
            capability=tool_protocol,
            data_dir=cfg.data_dir,
            data_verified=data_verified,
            verification_ref=verification_ref,
        )

    result = drive_with_backend(
        cfg, run, base_url=base_url, vram_reader=vram_reader, pid_usage_reader=pid_reader
    )
    s = result.score
    typer.echo(
        f"[bench-tooling] call-accuracy={s.call_accuracy:.3f} tool-selection="
        f"{s.tool_selection_accuracy:.3f} args-exact={s.argument_exactness:.3f} "
        f"no-hallucinated={s.no_hallucinated_tool_rate:.3f} well-formed={s.well_formed_rate:.3f}"
    )
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-tooling] manifest -> {result.paths['manifest']}")


@app.command("bench-agentic")
def bench_agentic_cmd(
    tasks: Path = typer.Option(
        Path("samples/benchmarks/agentic_tasks_uk.json"), help="agentic task set (JSON array)"
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint (skips launching)"
    ),
    max_steps: int = typer.Option(6, min=1, help="step budget per task"),
    harness: str = typer.Option(
        "loop",
        help="agentic harness: loop (pure) | langgraph ([eval] extra) | crewai ([crewai] extra). "
        "The comparison axis under TIER_AGENTIC; task set + scoring + judge are held fixed.",
    ),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM/llama.cpp served context window"),
    judge_model: Optional[str] = typer.Option(
        None,
        help="opt-in gated trajectory-quality judge (recorded alongside completion, never the "
        "headline)",
    ),
    judge_rho: Optional[float] = typer.Option(
        None, help="calibration Spearman rho; the judge is used only when rho >= threshold (0.6)"
    ),
    judge_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of the judge endpoint"
    ),
    data_verified: bool = typer.Option(
        False,
        help="stamp the run as human verification gate-verified for composite-headline eligibility",
    ),
    verification_ref: Optional[str] = typer.Option(
        None,
        help="path or label for the verification worksheet, sample manifest, or accepted ledger",
    ),
) -> None:
    """Score a model's task completion in the deterministic tool-world under TIER_AGENTIC."""
    from llb.bench.agentic import HARNESS_NAMES, AgenticRun, load_tasks_file, run_agentic
    from llb.bench.common import LLMComplete, drive_with_backend

    if harness not in HARNESS_NAMES:
        typer.echo(
            f"[error] unknown --harness '{harness}'; choose one of {', '.join(HARNESS_NAMES)}",
            err=True,
        )
        raise typer.Exit(code=2)
    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    task_set = load_tasks_file(tasks)
    vram_reader, pid_reader = best_effort_gpu_readers()

    def run(complete: LLMComplete) -> AgenticRun:
        return run_agentic(
            task_set,
            model=model,
            backend=backend,
            complete=complete,
            max_steps=max_steps,
            harness_name=harness,
            judge_model=judge_model,
            judge_rho=judge_rho,
            judge_base_url=judge_base_url,
            data_dir=cfg.data_dir,
            data_verified=data_verified,
            verification_ref=verification_ref,
        )

    result = drive_with_backend(
        cfg, run, base_url=base_url, vram_reader=vram_reader, pid_usage_reader=pid_reader
    )
    typer.echo(
        f"[bench-agentic] harness={harness} "
        f"completion-rate={result.result.objective_score:.3f} "
        f"mean-steps={result.mean_steps:.2f} mean-tool-calls={result.mean_tool_calls:.2f}"
    )
    if result.trajectory_quality is not None:
        typer.echo(
            f"[bench-agentic] trajectory-quality (gated judge)={result.trajectory_quality:.3f}"
        )
    if result.judge_diagnostics is not None:
        diag = result.judge_diagnostics
        typer.echo(
            f"[bench-agentic] judge-diagnostics ok={diag['n_ok']} zero={diag['n_zero']} "
            f"reasons={diag['reasons'] or '{}'}"
        )
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-agentic] manifest -> {result.paths['manifest']}")


@app.command("bench-agentic-compare")
def bench_agentic_compare_cmd(
    model: str = typer.Option(..., help="the candidate model to compare across harnesses"),
) -> None:
    """Rank one model's agentic runs across its harnesses (loop/langgraph/crewai).

    Reads the persisted `agentic` run bundles, keeps the best run per (model, harness), and ranks
    the harnesses for the chosen model under TIER_AGENTIC -- isolating the harness effect with the
    same bootstrap CIs as the category boards."""
    from llb.board.harnesses import harness_comparison

    cfg = load_config(None)
    rows, table, harnesses = harness_comparison(cfg.data_dir, model)
    if not rows:
        typer.echo(
            f"[bench-agentic-compare] no agentic runs for model '{model}' under {cfg.data_dir}; "
            "run `llb bench-agentic --harness loop|langgraph|crewai ...` first"
        )
        raise typer.Exit(code=2)
    typer.echo(
        f"[bench-agentic-compare] model={model} harnesses={', '.join(sorted(set(harnesses)))}"
    )
    typer.echo(table)


@app.command("bench-summarization")
def bench_summarization_cmd(
    cases: Path = typer.Option(
        Path("samples/benchmarks/summarization_cases_uk.json"),
        help="summarization cases (JSON array)",
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint"
    ),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM/llama.cpp served context window"),
    judge_model: Optional[str] = typer.Option(
        None,
        help="opt-in gated faithfulness judge (recorded alongside coverage, never the headline)",
    ),
    judge_rho: Optional[float] = typer.Option(
        None, help="calibration Spearman rho; the judge is used only when rho >= threshold (0.6)"
    ),
    judge_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of the judge endpoint"
    ),
    data_verified: bool = typer.Option(
        False,
        help="stamp the run as human verification gate-verified for composite-headline eligibility",
    ),
    verification_ref: Optional[str] = typer.Option(
        None,
        help="path or label for the verification worksheet, sample manifest, or accepted ledger",
    ),
) -> None:
    """Score summaries by pinned-embedder reference coverage under TIER_SUMMARIZATION."""
    from llb.bench.common import LLMComplete, drive_with_backend
    from llb.bench.summarization import SummarizationRun, load_cases_file, run_summarization

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    sum_cases = load_cases_file(cases)
    vram_reader, pid_reader = best_effort_gpu_readers()

    def run(complete: LLMComplete) -> SummarizationRun:
        return run_summarization(
            sum_cases,
            model=model,
            backend=backend,
            complete=complete,
            judge_model=judge_model,
            judge_rho=judge_rho,
            judge_base_url=judge_base_url,
            data_dir=cfg.data_dir,
            data_verified=data_verified,
            verification_ref=verification_ref,
        )

    result = drive_with_backend(
        cfg, run, base_url=base_url, vram_reader=vram_reader, pid_usage_reader=pid_reader
    )
    typer.echo(f"[bench-summarization] reference-coverage={result.result.objective_score:.3f}")
    if result.faithfulness is not None:
        typer.echo(f"[bench-summarization] faithfulness (gated judge)={result.faithfulness:.3f}")
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-summarization] manifest -> {result.paths['manifest']}")


@app.command("bench-structured")
def bench_structured_cmd(
    cases: Path = typer.Option(
        Path("samples/benchmarks/structured_cases_uk.json"),
        help="structured-output cases (JSON array)",
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint"
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
    """Score JSON-schema conformance + field accuracy under TIER_STRUCTURED."""
    from llb.bench.common import LLMComplete, drive_with_backend
    from llb.bench.structured import StructuredRun, load_cases_file, run_structured

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    st_cases = load_cases_file(cases)
    vram_reader, pid_reader = best_effort_gpu_readers()

    def run(complete: LLMComplete) -> StructuredRun:
        return run_structured(
            st_cases,
            model=model,
            backend=backend,
            complete=complete,
            data_dir=cfg.data_dir,
            data_verified=data_verified,
            verification_ref=verification_ref,
        )

    result = drive_with_backend(
        cfg, run, base_url=base_url, vram_reader=vram_reader, pid_usage_reader=pid_reader
    )
    typer.echo(
        f"[bench-structured] field-accuracy={result.score.field_accuracy:.3f} "
        f"conformance={result.score.conformance_rate:.3f}"
    )
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-structured] manifest -> {result.paths['manifest']}")


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
    from llb.bench.common import LLMComplete, drive_with_backend
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
