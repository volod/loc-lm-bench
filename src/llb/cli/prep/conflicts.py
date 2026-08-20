"""Corpus-conflict audit command: tiered duplicate / staleness / contradiction detection."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.prep.conflicts_output import echo_summary, projected_policies
from llb.conflicts.claim.calibration import DEFAULT_CALIBRATION_PROBE
from llb.conflicts.constants import (
    CONFLICTS_METHOD,
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_JACCARD_THRESHOLD,
    DEFAULT_LEAF_SIZE,
    MIN_CLAIM_TOKENS,
    TIER_CLAIM,
    TIER_HASH,
    TIER_SEMANTIC,
    TIERS,
    tiers_up_to,
)
from llb.conflicts.bundle.candidate_record import DEFAULT_CANDIDATE_RECORD_PAIRS
from llb.conflicts.calibration.distribution import (
    SUGGESTED_MAX_CANDIDATE_PAIRS,
    DEFAULT_NULL_SAMPLE_PAIRS,
    DEFAULT_NULL_SEED,
)
from llb.conflicts.resolution.policy import POLICIES


@app.command("audit-corpus-conflicts")
def audit_corpus_conflicts_cmd(
    corpus: Path = typer.Option(..., help="corpus root to audit (never modified)"),
    effort: str = typer.Option(
        TIER_HASH,
        help=f"computational effort, cumulative: {' < '.join(TIERS)}",
    ),
    store: Optional[Path] = typer.Option(
        None, help="built RAG store (required from --effort semantic upward)"
    ),
    goldset: Optional[Path] = typer.Option(
        None, help="gold set JSONL; enables the needle-ambiguity lane"
    ),
    out: Optional[Path] = typer.Option(
        None, help="report directory (default: $DATA_DIR/corpus-conflicts/<run>/)"
    ),
    conflict_model: Optional[str] = typer.Option(
        None, help="local model adjudicating claim pairs (required for --effort claim)"
    ),
    conflict_backend: str = typer.Option(
        "ollama", help="local backend for the claim tier: ollama | vllm | openai"
    ),
    conflict_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL for the claim tier"
    ),
    jaccard_threshold: float = typer.Option(
        DEFAULT_JACCARD_THRESHOLD, min=0.0, max=1.0, help="lexical near-duplicate cutoff"
    ),
    containment_threshold: float = typer.Option(
        DEFAULT_CONTAINMENT_THRESHOLD, min=0.0, max=1.0, help="lexical subsumption cutoff"
    ),
    cos_threshold: Optional[float] = typer.Option(
        None,
        min=0.0,
        max=1.0,
        help=f"semantic same-claim cosine cutoff, absolute (default {DEFAULT_COSINE_THRESHOLD}); "
        "overrides --cos-quantile when both are given",
    ),
    max_candidate_pairs: Optional[int] = typer.Option(
        None,
        min=1,
        help="calibrate the cosine cutoff from the corpus's OWN similarity distribution so the "
        f"tier returns at most this many candidate pairs (try {SUGGESTED_MAX_CANDIDATE_PAIRS}); "
        "portable across corpora and corpus sizes, unlike an absolute cosine or a bare quantile. "
        "A rank cutoff, not a false-positive guarantee -- see the data-prep known limitation",
    ),
    max_candidate_record_pairs: Optional[int] = typer.Option(
        None,
        min=1,
        help="how many DOCUMENT pairs of the ranked candidate list the bundle records, which "
        "bounds how deep a later `recompute-conflict-stage --budget` can reach "
        f"(default {DEFAULT_CANDIDATE_RECORD_PAIRS}, or this run's --max-candidate-pairs when it "
        "sets one); costs about 24 bytes per pair and answers every budget up to at least that "
        "rank",
    ),
    cos_quantile: Optional[float] = typer.Option(
        None,
        min=0.0,
        max=1.0,
        help="advanced: set the per-PAIR quantile directly instead of a candidate budget; "
        "note the flag count it admits grows with the corpus's pair count",
    ),
    null_sample_pairs: int = typer.Option(
        DEFAULT_NULL_SAMPLE_PAIRS,
        min=1,
        help="pairs sampled to estimate the null distribution (--cos-quantile only)",
    ),
    null_seed: int = typer.Option(
        DEFAULT_NULL_SEED,
        help="seed for null-distribution sampling and the clustered precision bootstrap; "
        "fixed for reproducibility",
    ),
    leaf_size: int = typer.Option(
        DEFAULT_LEAF_SIZE, min=1, help="semantic prefix tree leaf capacity"
    ),
    max_claim_pairs: int = typer.Option(
        0, min=0, help="cap adjudicated pairs (0 = every candidate pair)"
    ),
    calibrate_adjudicator: bool = typer.Option(
        True,
        "--calibrate-adjudicator/--no-calibrate-adjudicator",
        help="adjudicate the frozen calibration probe before measuring claim-tier precision; "
        "without it the precision block is suppressed rather than printed uncalibrated",
    ),
    calibration_probe: Optional[Path] = typer.Option(
        None,
        help="frozen-label probe for adjudicator calibration "
        f"(default {DEFAULT_CALIBRATION_PROBE})",
    ),
    min_claim_tokens: int = typer.Option(
        MIN_CLAIM_TOKENS,
        min=0,
        help="skip chunks with fewer content tokens (PDF page markers, bare headings)",
    ),
    center_vectors: bool = typer.Option(
        True,
        "--center-vectors/--no-center-vectors",
        help="remove the corpus mean direction before comparing; encoder spaces are anisotropic, "
        "so without it two unrelated chunks already score ~0.83 and the threshold means little",
    ),
    project_dims: int = typer.Option(
        0,
        min=0,
        help="PCA dimensions for exact Euclidean tree blocking (0 = blocked all-pairs scan)",
    ),
    project_policy: Optional[str] = typer.Option(
        None,
        help="also PROJECT the `to review` count under these resolution policies, comma-separated "
        f"({' | '.join(POLICIES)}): the rows `resolve-corpus-conflicts --policy <p>` would leave "
        "at `review_required`, one column per policy per decision group, plus the DELTA between "
        "them so the cost of the policy CHOICE is visible. Projections under named policies, "
        "never measurements -- off by default, and off leaves the report unchanged",
    ),
) -> None:
    """Report duplicated, stale, and contradictory knowledge in a corpus. Never edits the corpus."""
    from llb.conflicts.claim.adjudicator import build_adjudicator
    from llb.conflicts.audit import AuditParams, run_audit
    from llb.conflicts.report.render import write_audit
    from llb.conflicts.store_access import load_store_view
    from llb.core.paths import resolve_data_dir
    from llb.core.store_generations import generation_timestamp

    if effort not in TIERS:
        raise typer.BadParameter(f"unknown effort {effort!r}; choose one of {', '.join(TIERS)}")
    policies = projected_policies(project_policy)
    tiers = tiers_up_to(effort)

    view = None
    if TIER_SEMANTIC in tiers:
        index_dir = store if store is not None else resolve_data_dir() / "llb" / "rag"
        view = load_store_view(index_dir)

    complete = None
    if TIER_CLAIM in tiers:
        if not conflict_model:
            raise typer.BadParameter(
                "--effort claim needs --conflict-model (the local model adjudicating claim pairs)"
            )
        complete = build_adjudicator(conflict_model, conflict_backend, conflict_base_url)

    items = None
    if goldset is not None:
        from llb.goldset.schema import load_goldset

        items = load_goldset(goldset)

    result = run_audit(
        corpus,
        AuditParams(
            effort=effort,
            jaccard_threshold=jaccard_threshold,
            containment_threshold=containment_threshold,
            cos_threshold=cos_threshold,
            cos_quantile=cos_quantile,
            max_candidate_pairs=max_candidate_pairs,
            max_candidate_record_pairs=max_candidate_record_pairs,
            null_sample_pairs=null_sample_pairs,
            null_seed=null_seed,
            leaf_size=leaf_size,
            max_claim_pairs=max_claim_pairs,
            min_claim_tokens=min_claim_tokens,
            center_vectors=center_vectors,
            project_dims=project_dims,
            calibrate_adjudicator=calibrate_adjudicator,
            calibration_probe=calibration_probe,
        ),
        store=view,
        goldset=items,
        complete=complete,
    )

    # The projection is composed HERE, above both layers: it needs the resolution vocabulary, and
    # the report must not. `write_audit` renders whatever plain data it is handed.
    if policies:
        from llb.conflicts.resolution.projection import project_policies

        result.policy_projection = project_policies(result.rows(), policies)

    out_dir = (
        out if out is not None else resolve_data_dir() / CONFLICTS_METHOD / generation_timestamp()
    )
    paths = write_audit(out_dir, result)
    echo_summary(result, paths)
