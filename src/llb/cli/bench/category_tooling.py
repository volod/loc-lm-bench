"""Focused category tooling implementation."""

from pathlib import Path
from typing import Optional
import typer
from llb.cli.app import app
from llb.cli.bench._shared import _echo_throughput
from llb.cli.helpers import best_effort_gpu_readers, load_config


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
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.bench.tooling import (
        ToolingRun,
        load_catalog_file,
        run_tooling,
    )
    from llb.bench.tooling_protocol import TOOL_PROTOCOL_NATIVE, ToolCaller, native_tool_caller

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    tool_catalog, cases = load_catalog_file(catalog)
    vram_reader, pid_reader = best_effort_gpu_readers()
    meter = ThroughputMeter()

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
        f"[bench-tooling] call-accuracy={s.call_accuracy:.3f} tool-selection="
        f"{s.tool_selection_accuracy:.3f} args-exact={s.argument_exactness:.3f} "
        f"no-hallucinated={s.no_hallucinated_tool_rate:.3f} well-formed={s.well_formed_rate:.3f}"
    )
    _echo_throughput("bench-tooling", meter)
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-tooling] manifest -> {result.paths['manifest']}")
