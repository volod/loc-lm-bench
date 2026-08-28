"""Reranker bake-off command (`compare-rerankers`): rank cross-encoders on one gold set."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.cli.rag.compare_stores import _compare_vector_corpus_root

# Pure, dependency-free defaults (no torch / FAISS pulled in): safe as Typer option defaults.
from llb.core.config_validation import DEFAULT_RERANK_CANDIDATES
from llb.rag.embedding_bakeoff.uncertainty import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
)
from llb.rag.rerank_bakeoff.lane import DEFAULT_RERANK_BARS
from llb.rag.rerank_bakeoff.loader import DEFAULT_BATCH_SIZE, DTYPE_AUTO
from llb.rag.rerank_bakeoff.models import BAKEOFF_METHOD, DEFAULT_BASELINE_RERANKER

if TYPE_CHECKING:
    from llb.rag.encoders.candidate_screen import SkippedCandidate
    from llb.rag.rerank_bakeoff.models import VramHeadroom

# Headroom the host keeps for fragmentation and the CUDA context, on top of the generator's own
# residency. A reranker that only fits by consuming this is not a reranker that fits.
DEFAULT_VRAM_RESERVE_MB = 512.0


def _resolve_roster(
    models: str, allow_remote_code: bool
) -> tuple[list[str], list["SkippedCandidate"]]:
    """Screen the requested roster into candidates to load plus visibly declined entries."""
    from llb.rag.encoders.candidate_screen import UnregisteredCandidateError
    from llb.rag.rerank_bakeoff.models import DEFAULT_RERANK_CANDIDATES_ROSTER
    from llb.rag.encoders.model_stack import installed_transformers_major
    from llb.rag.rerank_bakeoff.roster import screen_rerankers

    roster = [m.strip() for m in models.split(",") if m.strip()] or DEFAULT_RERANK_CANDIDATES_ROSTER
    try:
        candidates, skipped = screen_rerankers(
            roster,
            allow_remote_code=allow_remote_code,
            transformers_major=installed_transformers_major(),
        )
    except UnregisteredCandidateError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    for row in skipped:
        typer.echo(f"[compare-rerankers] skipping {row['model']}: {row['detail']}", err=True)
    return candidates, skipped


def _headroom(generator_vram_mb: float | None, reserve_mb: float) -> "VramHeadroom":
    """The VRAM budget the fit gate reads: device total minus the declared generator residency."""
    from llb.backends.hardware import detect_gpus, max_vram_mb

    total = float(max_vram_mb(detect_gpus())) or None
    if generator_vram_mb is None or total is None:
        return {
            "total_mb": total,
            "generator_mb": generator_vram_mb,
            "reserve_mb": reserve_mb,
            "headroom_mb": None,
        }
    return {
        "total_mb": total,
        "generator_mb": generator_vram_mb,
        "reserve_mb": reserve_mb,
        "headroom_mb": max(total - generator_vram_mb - reserve_mb, 0.0),
    }


@app.command("compare-rerankers")
def compare_rerankers_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    corpus_root: Optional[Path] = typer.Option(
        None, help="corpus to index once; defaults to the sibling corpus/ of --goldset"
    ),
    models: str = typer.Option(
        "", help="comma-separated reranker ids; empty uses the default UA candidate set"
    ),
    k: int = typer.Option(10, help="recall@k / MRR cutoff -- what the reranker keeps"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
    rerank_candidates: int = typer.Option(
        DEFAULT_RERANK_CANDIDATES,
        min=1,
        help="candidate pool depth retrieved ONCE and re-sorted by every candidate",
    ),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE, min=1, help="cross-encoder predict batch size (recorded in the report)"
    ),
    dtype: str = typer.Option(
        DTYPE_AUTO, help="load dtype passed to every candidate ('auto' keeps each card's own)"
    ),
    allow_remote_code: bool = typer.Option(
        False,
        "--allow-remote-code",
        help="opt into candidates that ship their own modelling code (trust_remote_code), e.g. "
        "jina-reranker-v2 / gte-multilingual-reranker; without it those rows are SKIPPED and "
        "recorded",
    ),
    generator_vram_mb: Optional[float] = typer.Option(
        None,
        help="VRAM the GENERATOR holds while serving; declares the budget a reranker must fit "
        "beside it. Without it footprints are still measured and the fit gate does not run",
    ),
    vram_reserve_mb: float = typer.Option(
        DEFAULT_VRAM_RESERVE_MB, help="VRAM kept free on top of the generator residency"
    ),
    baseline: str = typer.Option(
        DEFAULT_BASELINE_RERANKER,
        help="incumbent reranker every candidate is PAIRED against (empty disables the paired "
        "intervals and the keep-or-swap verdict)",
    ),
    adoption_bars: str = typer.Option(
        ",".join(DEFAULT_RERANK_BARS),
        "--adoption-bars",
        help="paired metric interval(s) a candidate must clear to be adopted. This lane defaults "
        "to BOTH recall_at_k and mrr: a cross-encoder can only re-sort the pool it is handed, so "
        "first-hit rank is the quantity it moves",
    ),
    resamples: int = typer.Option(
        DEFAULT_RESAMPLES, help="paired percentile-bootstrap resamples for the delta intervals"
    ),
    confidence: float = typer.Option(
        DEFAULT_CONFIDENCE, min=0.5, max=0.999, help="paired bootstrap CI level"
    ),
    seed: int = typer.Option(DEFAULT_SEED, help="seed for the shared bootstrap index sets"),
    noise_floor: bool = typer.Option(
        False,
        "--noise-floor",
        help="also measure the MEASUREMENT FLOOR per candidate over its OWN rerank scores and "
        "state whether the recommended lead clears it",
    ),
    noise_floor_replicates: Optional[int] = typer.Option(
        None, help="--noise-floor: jitter replicates per candidate (default 64)"
    ),
    in_process: bool = typer.Option(
        False,
        "--in-process",
        help="load every candidate in THIS process instead of isolating each in its own. Faster "
        "by one spawn per candidate, but a candidate whose modelling code raises a device-side "
        "assert then poisons the CUDA context and every later candidate reads as unloadable",
    ),
    out: Optional[Path] = typer.Option(
        None, help="write report.md here (default: $DATA_DIR/compare-rerankers/<ts>/report.md)"
    ),
) -> None:
    """Rank candidate cross-encoder rerankers on one gold set (rank quality + latency + VRAM).

    Retrieves ONE candidate pool per item at a fixed encoder and chunking, then re-sorts that same
    pool with every candidate, so the rows differ only by the reranker. The reranker-OFF row rides
    along, each row carries its paired delta against the incumbent, and the run ends in a
    keep-or-swap verdict beside the cost the swap is paid in. Heavy loads stay outside quick CI.
    """
    import json

    from llb.bench.common import new_run_timestamp
    from llb.cli.helpers import best_effort_gpu_readers
    from llb.cli.rag.embedding_stores import local_store_builder
    from llb.executor.cases import spans_as_dicts
    from llb.goldset.schema import load_goldset
    from llb.rag.embedding_bakeoff.verdict import resolve_bars
    from llb.rag.rerank_bakeoff.lane import run_rerank_bakeoff
    from llb.rag.rerank_bakeoff.loader import cross_encoder_loader
    from llb.rag.rerank_bakeoff.report import format_report, render_markdown
    from llb.rag.rerank_bakeoff.worker import isolated_loader

    try:
        bars = resolve_bars(adoption_bars, default=DEFAULT_RERANK_BARS)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    cfg = load_config(
        config,
        goldset_path=goldset,
        corpus_root=_compare_vector_corpus_root(goldset, corpus_root),
    )
    items = load_goldset(cfg.goldset_path)
    if split:
        items = [it for it in items if it.split == split]
    bakeoff_items = [(it.question, spans_as_dicts(it)) for it in items]
    candidates, skipped = _resolve_roster(models, allow_remote_code)

    _, run_ts = new_run_timestamp()
    run_dir = cfg.data_dir / BAKEOFF_METHOD / run_ts
    report_path = out if out is not None else run_dir / "report.md"
    stores_dir = run_dir / "stores"
    stores_dir.mkdir(parents=True, exist_ok=True)

    built = local_store_builder(cfg, stores_dir)(cfg.embedding_model)
    typer.echo(
        f"[compare-rerankers] retrieving {rerank_candidates} candidates/item from "
        f"{cfg.embedding_model} ({len(items)} items)"
    )
    pools = [
        built.store.retrieve(question, rerank_candidates) for question, _spans in bakeoff_items
    ]
    # Free the encoder before the first candidate loads, so a reranker's measured footprint is its
    # own rather than the encoder's plus its own.
    release = getattr(getattr(built.store, "embedder", None), "release", None)
    if callable(release):
        release()

    vram_reader, _pid = best_effort_gpu_readers()
    load_scorer = (
        cross_encoder_loader(batch_size=batch_size, dtype=dtype, vram_reader=vram_reader)
        if in_process
        else isolated_loader(batch_size=batch_size, dtype=dtype)
    )
    report = run_rerank_bakeoff(
        bakeoff_items,
        pools,
        k,
        corpus_root=str(cfg.corpus_root),
        embedding_model=cfg.embedding_model,
        chunking=f"{cfg.strategy}@{cfg.chunk_size}/{cfg.chunk_overlap}",
        pool_depth=rerank_candidates,
        batch_size=batch_size,
        candidates=candidates,
        load_scorer=load_scorer,
        dtype=dtype,
        item_ids=[item.id for item in items],
        skipped=skipped,
        headroom=_headroom(generator_vram_mb, vram_reserve_mb),
        baseline=baseline.strip() or None,
        bars=bars,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
        noise_floor=noise_floor,
        noise_floor_replicates=noise_floor_replicates,
    )
    typer.echo(format_report(report))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(report), encoding="utf-8")
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(f"[compare-rerankers] wrote report -> {report_path} ; {json_path}")
