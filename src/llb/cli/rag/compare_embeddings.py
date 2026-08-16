"""Embedder bake-off command (`compare-embeddings`): rank encoders on one gold set."""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from llb.core import env
from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.cli.rag.compare_stores import _compare_vector_corpus_root
from llb.cli.rag.compare_embeddings_output import (
    egress_consent,
    resolved_bars,
    write_bakeoff_report,
    write_throughput_summary,
)
from llb.cli.rag.embedding_stores import (
    EncoderThroughputOptions,
    api_store_builder,
    local_store_builder,
)

# Pure, dependency-free defaults (no FAISS / torch pulled in): safe as Typer option defaults.
from llb.rag.embedding_bakeoff_uncertainty import (
    DEFAULT_BARS,
    DEFAULT_BASELINE_MODEL,
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    METRICS,
    METRIC_RECALL,
)
from llb.rag.fusion_evidence.power import DEFAULT_TARGET_POWER

if TYPE_CHECKING:
    from llb.rag.candidate_screen import SkippedCandidate


def _resolve_roster(
    models: str, allow_remote_code: bool
) -> tuple[list[str], list["SkippedCandidate"]]:
    """Screen the requested roster into candidates to build plus visibly declined entries.

    An id with no declared query/passage convention exits the run (scoring it would guess a format
    and understate the encoder); a candidate needing `trust_remote_code` is declined unless the
    operator opted in, and the opt-in is exported process-wide -- like `LLB_EMBED_DEVICE`, because
    the store build, the lazy reload behind `retrieve()`, and the throughput profiler each
    construct their own `Embedder` and all three must agree.
    """
    from llb.rag.embedding_bakeoff_models import DEFAULT_LOCAL_CANDIDATES
    from llb.rag.embedding_bakeoff_roster import UnregisteredCandidateError, screen_candidates

    roster = [m.strip() for m in models.split(",") if m.strip()] or DEFAULT_LOCAL_CANDIDATES
    try:
        local_models, skipped = screen_candidates(roster, allow_remote_code=allow_remote_code)
    except UnregisteredCandidateError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    for row in skipped:
        typer.echo(f"[compare-embeddings] skipping {row['model']}: {row['detail']}", err=True)
    if allow_remote_code:
        os.environ[env.LLB_TRUST_REMOTE_CODE] = "1"
    return local_models, skipped


@app.command("compare-embeddings")
def compare_embeddings_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    corpus_root: Optional[Path] = typer.Option(
        None,
        help="corpus to build for each candidate; defaults to the sibling corpus/ of --goldset",
    ),
    models: str = typer.Option(
        "",
        help="comma-separated local embedding model ids; empty uses the default UA candidate set",
    ),
    k: int = typer.Option(10, help="recall@k / MRR cutoff"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
    api_model: Optional[str] = typer.Option(
        None,
        help="opt-in API embedder row (e.g. cohere/embed-multilingual-v3.0); full corpus EGRESS -- "
        "needs --data-classification open + interactive consent, honors --max-usd",
    ),
    data_classification: Optional[str] = typer.Option(
        None, help="corpus data classification; must be 'open' to enable --api-model"
    ),
    max_usd: Optional[float] = typer.Option(
        None, help="hard budget cap for the --api-model egress lane (USD)"
    ),
    yes: bool = typer.Option(
        False, "--yes", help="skip the interactive egress consent prompt for --api-model"
    ),
    allow_remote_code: bool = typer.Option(
        False,
        "--allow-remote-code",
        help="opt into candidates that ship their own modelling code (trust_remote_code), e.g. "
        "gte-multilingual / jina-embeddings-v3; without it those rows are SKIPPED and recorded",
    ),
    noise_floor: bool = typer.Option(
        False,
        "--noise-floor",
        help="also measure the MEASUREMENT FLOOR per candidate (see compare-retrieval) and state "
        "whether the recommended embedder's lead over the runner-up clears it",
    ),
    noise_floor_replicates: Optional[int] = typer.Option(
        None, help="--noise-floor: jitter replicates per candidate (default 64)"
    ),
    baseline: str = typer.Option(
        DEFAULT_BASELINE_MODEL,
        help="incumbent embedder every candidate is PAIRED against (empty disables the paired "
        "intervals and the adopt-or-retain verdict)",
    ),
    adoption_bars: str = typer.Option(
        ",".join(DEFAULT_BARS),
        "--adoption-bars",
        help="paired metric interval(s) a candidate must clear to be ADOPTED. `recall_at_k` (the "
        "default) is unconditional and always kept; adding `mrr` opts into the scoped first-hit-"
        "rank bar, for configurations where rank binds (small --k, or a cross-encoder reranker)",
    ),
    resamples: int = typer.Option(
        DEFAULT_RESAMPLES, help="paired percentile-bootstrap resamples for the delta intervals"
    ),
    confidence: float = typer.Option(
        DEFAULT_CONFIDENCE, min=0.5, max=0.999, help="paired bootstrap CI level"
    ),
    seed: int = typer.Option(DEFAULT_SEED, help="seed for the shared bootstrap index sets"),
    power_reference: Optional[Path] = typer.Option(
        None,
        help="earlier report.json whose selected paired metric variance prices this item set",
    ),
    power_candidate: Optional[str] = typer.Option(
        None, help="candidate model whose delta versus --baseline the power contract selects"
    ),
    power_metric: str = typer.Option(
        METRIC_RECALL, help=f"paired metric selected for power: {','.join(METRICS)}"
    ),
    minimum_detectable_delta: Optional[float] = typer.Option(
        None, min=0.001, help="smallest material paired metric gain to distinguish"
    ),
    target_power: float = typer.Option(
        DEFAULT_TARGET_POWER,
        min=0.501,
        max=0.999,
        help="target power for the paired normal-approximation item count",
    ),
    out: Optional[Path] = typer.Option(
        None, help="write report.md here (default: $DATA_DIR/compare-embeddings/<ts>/report.md)"
    ),
    encoder_throughput: bool = typer.Option(
        False,
        "--encoder-throughput",
        help="also decompose cold load / first-pass compile+encode / warm steady encode rates "
        "per candidate (writes additive fields + $DATA_DIR/encoder-throughput/<ts>/)",
    ),
    encoder_precision: float = typer.Option(
        0.05,
        help="--encoder-throughput: stop warm passes when IQR/median <= this relative precision",
    ),
    encoder_min_warm: int = typer.Option(
        3, min=1, help="--encoder-throughput: minimum warm encode passes"
    ),
    encoder_max_warm: int = typer.Option(
        10, min=1, help="--encoder-throughput: maximum warm encode passes"
    ),
    encoder_max_warm_seconds: float = typer.Option(
        180.0,
        min=1.0,
        help="--encoder-throughput: wall-clock cap over the warm loop only",
    ),
    encoder_compare_cpu: bool = typer.Option(
        False,
        "--encoder-compare-cpu",
        help="--encoder-throughput: also profile CPU beside the resolved CUDA device",
    ),
) -> None:
    """Rank candidate embedders for Ukrainian RAG on one gold set (recall@k / MRR + throughput).

    Builds one store per candidate over the SAME corpus + chunking, scores the source-span metric,
    and writes a ranked report.md carrying each candidate's PAIRED delta interval against the
    baseline embedder plus an adopt-or-retain verdict. Heavy store builds stay outside quick CI.
    The opt-in --api-model row is corpus egress (bake-off evidence only; scored retrieval stays
    local) and is refused unless --data-classification open plus explicit consent.
    """
    from llb.bench.common import new_run_timestamp
    from llb.backends.telemetry_samplers import nvidia_smi_power_reader
    from llb.cli.helpers import best_effort_gpu_readers
    from llb.executor.cases import spans_as_dicts
    from llb.goldset.schema import load_goldset
    from llb.prep.frontier_telemetry import ProvenanceLog
    from llb.rag.embedding_bakeoff import run_bakeoff
    from llb.rag.embedding_bakeoff_power import prepare_embedding_power, resolve_embedding_power
    from llb.rag.encoder_throughput import ThroughputProfile

    bars = resolved_bars(adoption_bars)
    cfg = load_config(
        config,
        goldset_path=goldset,
        corpus_root=_compare_vector_corpus_root(goldset, corpus_root),
    )
    items = load_goldset(cfg.goldset_path)
    if split:
        items = [it for it in items if it.split == split]
    bakeoff_items = [(it.question, spans_as_dicts(it)) for it in items]
    local_models, skipped = _resolve_roster(models, allow_remote_code)

    _, run_ts = new_run_timestamp()
    run_dir = cfg.data_dir / "compare-embeddings" / run_ts
    stores_dir = run_dir / "stores"
    report_path = out if out is not None else run_dir / "report.md"
    try:
        power_plan = prepare_embedding_power(
            power_reference,
            candidate=power_candidate,
            metric=power_metric,
            minimum_detectable_delta=minimum_detectable_delta,
            target_power=target_power,
            confidence=confidence,
            planned_n=len(items),
            plan_path=report_path.parent / "power-plan.json",
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    stores_dir.mkdir(parents=True, exist_ok=True)

    egress_log = ProvenanceLog()
    throughput_profiles: list[ThroughputProfile] = []
    vram_reader, _pid = best_effort_gpu_readers()
    power_reader = nvidia_smi_power_reader() if encoder_throughput else None
    throughput_opts = EncoderThroughputOptions(
        enabled=encoder_throughput,
        target_relative_precision=encoder_precision,
        min_warm_passes=encoder_min_warm,
        max_warm_passes=encoder_max_warm,
        max_warm_seconds=encoder_max_warm_seconds,
        compare_cpu=encoder_compare_cpu,
    )
    report = run_bakeoff(
        bakeoff_items,
        k,
        corpus_root=str(cfg.corpus_root),
        local_models=local_models,
        build_local=local_store_builder(
            cfg,
            stores_dir,
            throughput=throughput_opts,
            vram_reader=vram_reader,
            power_reader=power_reader,
            profiles_out=throughput_profiles,
        ),
        item_ids=[item.id for item in items],
        skipped=skipped,
        api_model=api_model,
        build_api=api_store_builder(cfg, stores_dir, egress_log, max_usd),
        data_classification=data_classification,
        consent=lambda: egress_consent(cfg, api_model, yes),
        noise_floor=noise_floor,
        noise_floor_replicates=noise_floor_replicates,
        baseline=baseline.strip() or None,
        bars=bars,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    if power_plan is not None:
        try:
            report["power_analysis"] = resolve_embedding_power(report, power_plan)
        except ValueError as exc:
            typer.echo(f"[error] {exc}", err=True)
            raise typer.Exit(code=2) from None
    if throughput_profiles:
        report["encoder_throughput"] = write_throughput_summary(  # type: ignore[typeddict-item]
            throughput_profiles,
            baseline=baseline.strip() or None,
            data_dir=cfg.data_dir,
            run_ts=run_ts,
        )
    write_bakeoff_report(report, report_path)
