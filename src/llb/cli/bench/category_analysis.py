"""Text-analysis, security, and tooling category benchmark commands."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.bench._shared import _echo_throughput
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
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.bench.text_analysis.model import TextAnalysisRun
    from llb.bench.text_analysis.run import run_text_analysis

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    vram_reader, pid_reader = best_effort_gpu_readers()
    meter = ThroughputMeter()

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
    _echo_throughput("bench-text-analysis", meter)
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
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
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
    _echo_throughput("bench-security", meter)
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-security] manifest -> {result.paths['manifest']}")
