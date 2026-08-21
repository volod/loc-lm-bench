"""Command-driven local-model knowledge-cutoff benchmark."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.bench._shared import _echo_throughput
from llb.cli.helpers import best_effort_gpu_readers, load_config


@app.command("bench-knowledge-cutoff")
def bench_knowledge_cutoff_cmd(
    model: str = typer.Option(..., help="candidate local model id"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible URL of a running local endpoint"
    ),
    events: Optional[Path] = typer.Option(
        None, help="offline/local event JSONL (bypasses Hugging Face)"
    ),
    dataset_id: str = typer.Option(
        "apoorvumang/knowledge-cutoff-benchmark", help="Hugging Face dataset id"
    ),
    dataset_revision: str = typer.Option("main", help="HF branch, tag, or commit to resolve/pin"),
    threshold: float = typer.Option(0.5, min=0.01, max=1.0, help="raw monthly accuracy threshold"),
    optuna_trials: int = typer.Option(200, min=1, help="bounded decay-fit trials"),
    seed: int = typer.Option(42, help="Optuna sampler seed"),
    limit: Optional[int] = typer.Option(
        None, min=1, help="smoke-only event cap spread across the date range"
    ),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM/llama.cpp served context window"),
) -> None:
    """Estimate a local model's effective knowledge cutoff and generate MLOps reports."""
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.bench.knowledge_cutoff.data import LoadedEvents, load_events, select_events
    from llb.bench.knowledge_cutoff.run import KnowledgeCutoffRun, run_knowledge_cutoff

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    loaded = load_events(
        path=events,
        dataset_id=dataset_id,
        revision=dataset_revision,
        cache_dir=cfg.data_dir / "cache" / "huggingface" / "datasets",
    )
    loaded = LoadedEvents(select_events(loaded.events, limit), loaded.source)
    vram_reader, pid_reader = best_effort_gpu_readers()
    meter = ThroughputMeter()

    def run(complete: LLMComplete) -> KnowledgeCutoffRun:
        return run_knowledge_cutoff(
            loaded,
            model=model,
            backend=backend,
            complete=complete,
            data_dir=cfg.data_dir,
            threshold=threshold,
            optuna_trials=optuna_trials,
            seed=seed,
            meter=meter,
        )

    result = drive_with_backend(
        cfg,
        run,
        base_url=base_url,
        max_tokens=16,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
        meter=meter,
    )
    typer.echo(
        f"[bench-knowledge-cutoff] effective-cutoff="
        f"{result.fit.effective_cutoff or 'unavailable'} fit={result.fit.status} "
        f"accuracy={result.summary.eligible_accuracy:.3f} parse-rate={result.summary.parse_rate:.3f}"
    )
    _echo_throughput("bench-knowledge-cutoff", meter)
    if result.report_markdown is not None:
        typer.echo(f"[bench-knowledge-cutoff] report -> {result.report_markdown}")
