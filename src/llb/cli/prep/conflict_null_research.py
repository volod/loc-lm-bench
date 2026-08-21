"""CLI for the corpus-conflict independent-null research matrix."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.prep.conflict_null_research_support import (
    echo_summary,
    shared_embedding_model,
    validate_generation,
)
from llb.conflicts.constants import (
    DEFAULT_ADJUDICATION_BUDGET,
    DEFAULT_CROSS_ENCODER_ROWS,
    DEFAULT_ROLE_SAMPLES_PER_TYPE,
    DEFAULT_SYNTHESIS_PER_DOCUMENT,
    RESEARCH_GENERATION_FOURTH,
    RESEARCH_GENERATION_INITIAL,
    RESEARCH_GENERATION_THIRD,
)
from llb.conflicts.null_research.evaluation import (
    DEFAULT_MAX_GOODS_CANDIDATES,
    DEFAULT_RESEARCH_FPR,
    DEFAULT_RESEARCH_RANK_BUDGET,
    DEFAULT_TRANSFER_THRESHOLD,
)
from llb.rag.rerank import DEFAULT_RERANKER


@app.command("research-conflict-nulls")
def research_conflict_nulls_cmd(
    fixture_corpus: Path = typer.Option(..., help="planted conflict fixture corpus"),
    fixture_store: Path = typer.Option(..., help="real-embedder store for the planted fixture"),
    hr_corpus: Path = typer.Option(..., help="high-recall quickstart corpus"),
    hr_store: Path = typer.Option(..., help="store for the high-recall quickstart corpus"),
    goods_corpus: Path = typer.Option(..., help="goods quickstart corpus"),
    goods_store: Path = typer.Option(..., help="store for the goods quickstart corpus"),
    reference_corpus: Optional[Path] = typer.Option(
        None, help="unrelated Ukrainian reference corpus (every generation except fourth)"
    ),
    reference_store: Optional[Path] = typer.Option(
        None, help="store for the unrelated reference corpus"
    ),
    domain_reference_corpus: Optional[Path] = typer.Option(
        None, help="second unrelated, domain-matched Ukrainian reference corpus"
    ),
    domain_reference_store: Optional[Path] = typer.Option(
        None, help="store for the domain-matched reference corpus"
    ),
    generation: str = typer.Option(
        RESEARCH_GENERATION_INITIAL,
        help="which matrix to run: initial (cross-corpus/permutation/held-out/labelled), "
        "next (matched, residual, cluster-FDR, counterfactual), third (feasibility, "
        "propensity-balanced control, mixture identifiability, geometry variants, verified "
        "control roles, claim-tier precision), or fourth (in-support control synthesis, "
        "cross-encoder relation scoring, group-split conformal tail inference)",
    ),
    conflict_model: Optional[str] = typer.Option(
        None,
        help="local model adjudicating claim pairs, control roles, and generated controls "
        "(--generation third and fourth)",
    ),
    conflict_backend: str = typer.Option(
        "ollama", help="local backend for the adjudicator: ollama | vllm | openai"
    ),
    conflict_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL for the adjudicator"
    ),
    out: Optional[Path] = typer.Option(
        None,
        help="artifact directory (default: $DATA_DIR/corpus-conflicts/null-research/<run>)",
    ),
    fpr: float = typer.Option(
        DEFAULT_RESEARCH_FPR,
        min=0.000001,
        max=0.5,
        help="nominal upper-tail probability resolved independently by each null",
    ),
    rank_budget: int = typer.Option(
        DEFAULT_RESEARCH_RANK_BUDGET,
        min=1,
        help="current candidate-budget baseline on the planted fixture",
    ),
    transfer_threshold: float = typer.Option(
        DEFAULT_TRANSFER_THRESHOLD,
        min=0.0,
        max=1.0,
        help="swept HR baseline cosine whose pairs must be recovered",
    ),
    max_goods_candidates: int = typer.Option(
        DEFAULT_MAX_GOODS_CANDIDATES,
        min=0,
        help="largest non-flooding candidate count on the goods corpus",
    ),
    permutations: int = typer.Option(
        3,
        min=1,
        help="minimum deterministic shuffles per chunk; small corpora repeat until the null "
        "tail is statistically resolved",
    ),
    matches_per_reference: int = typer.Option(
        2,
        min=1,
        help="nearest surface/encoder-neighborhood controls selected from each reference",
    ),
    adjudication_budget: int = typer.Option(
        DEFAULT_ADJUDICATION_BUDGET,
        min=1,
        help="ranked candidate rows adjudicated per corpus for the claim-tier precision curve",
    ),
    role_samples_per_type: int = typer.Option(
        DEFAULT_ROLE_SAMPLES_PER_TYPE,
        min=1,
        help="traced control edits per corpus and edit type sent to the relation verifier",
    ),
    synthesis_per_document: int = typer.Option(
        DEFAULT_SYNTHESIS_PER_DOCUMENT,
        min=1,
        help="in-support control claims generated per source document (--generation fourth)",
    ),
    cross_encoder_rows: int = typer.Option(
        DEFAULT_CROSS_ENCODER_ROWS,
        min=1,
        help="top-cosine rows per corpus the cross-encoder re-scores (--generation fourth)",
    ),
    cross_encoder: str = typer.Option(
        DEFAULT_RERANKER, help="cross-encoder scoring the relation lane (--generation fourth)"
    ),
    cross_encoder_device: Optional[str] = typer.Option(
        None, help="sentence-transformers device for the cross-encoder (for example cuda)"
    ),
    seed: int = typer.Option(0, help="deterministic research seed"),
    embedding_device: Optional[str] = typer.Option(
        None, help="sentence-transformers device for permutation embeddings (for example cuda)"
    ),
) -> None:
    """Compare independent-null candidates against fixture and cross-corpus transfer gates."""
    from llb.conflicts.claim.adjudicator import build_adjudicator
    from llb.conflicts.null_research.run import run_null_research
    from llb.conflicts.null_research.report.render import write_null_research
    from llb.conflicts.store_access import load_store_view
    from llb.core.paths import resolve_data_dir
    from llb.core.store_generations import generation_timestamp
    from llb.rag.encoders.embedder import Embedder
    from llb.rag.rerank import CrossEncoderReranker

    validate_generation(
        generation,
        reference_corpus,
        reference_store,
        domain_reference_corpus,
        domain_reference_store,
        conflict_model,
    )
    fixture_view = load_store_view(fixture_store)
    hr_view = load_store_view(hr_store)
    goods_view = load_store_view(goods_store)
    reference_view = load_store_view(reference_store) if reference_store is not None else None
    domain_reference_view = (
        load_store_view(domain_reference_store) if domain_reference_store is not None else None
    )
    model = shared_embedding_model(
        [fixture_view, hr_view, goods_view, reference_view, domain_reference_view]
    )
    complete = build_adjudicator(conflict_model, conflict_backend, conflict_base_url)
    # The third generation scores stored vectors and constructed edits only, so it never loads an
    # encoder -- which also leaves the GPU to the adjudicating model.
    embedder = (
        None
        if generation == RESEARCH_GENERATION_THIRD
        else Embedder(model, device=embedding_device)
    )
    scorer = (
        CrossEncoderReranker(cross_encoder, device=cross_encoder_device)
        if generation == RESEARCH_GENERATION_FOURTH
        else None
    )
    try:
        summary = run_null_research(
            fixture=(fixture_corpus, fixture_view),
            hr=(hr_corpus, hr_view),
            goods=(goods_corpus, goods_view),
            reference=(reference_corpus, reference_view)
            if reference_corpus is not None and reference_view is not None
            else None,
            embed=embedder.encode_passages if embedder is not None else None,
            domain_reference=(domain_reference_corpus, domain_reference_view)
            if domain_reference_corpus is not None and domain_reference_view is not None
            else None,
            generation=generation,
            complete=complete,
            scorer=scorer,
            fpr=fpr,
            rank_budget=rank_budget,
            transfer_threshold=transfer_threshold,
            max_goods_candidates=max_goods_candidates,
            permutations=permutations,
            matches_per_reference=matches_per_reference,
            adjudication_budget=adjudication_budget,
            adjudicator_model=conflict_model or "",
            cross_encoder_model=cross_encoder if scorer is not None else "",
            role_samples_per_type=role_samples_per_type,
            synthesis_per_document=synthesis_per_document,
            cross_encoder_rows=cross_encoder_rows,
            seed=seed,
        )
    finally:
        if embedder is not None:
            embedder.release()
    out_dir = out or (
        resolve_data_dir() / "corpus-conflicts" / "null-research" / generation_timestamp()
    )
    paths = write_null_research(out_dir, summary)
    echo_summary(generation, summary, paths)
