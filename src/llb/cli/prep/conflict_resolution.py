"""Plan, apply, measure, and roll back corpus-conflict resolutions."""

from pathlib import Path
from typing import Any, Optional

import typer

from llb.cli.app import app
from llb.conflicts.resolution_policy import POLICIES, POLICY_CONSERVATIVE


@app.command("resolve-corpus-conflicts")
def resolve_corpus_conflicts_cmd(
    findings: Optional[Path] = typer.Option(None, help="audit findings.jsonl"),
    corpus: Optional[Path] = typer.Option(
        None, help="corpus root; inferred from adjacent summary.json when possible"
    ),
    out: Optional[Path] = typer.Option(None, help="artifact directory (default: findings parent)"),
    policy: str = typer.Option(
        POLICY_CONSERVATIVE, help=f"resolution policy: {' | '.join(POLICIES)}"
    ),
    apply: bool = typer.Option(False, "--apply", help="install the validated additive overlay"),
    rollback: bool = typer.Option(
        False, "--rollback", help="remove the installed overlay; source documents stay untouched"
    ),
    reviewed: Optional[Path] = typer.Option(
        None, help="reviewed resolution_review.jsonl whose decisions override escalations"
    ),
    store: Optional[Path] = typer.Option(
        None, help="RAG store to refresh and measure after apply/rollback"
    ),
    goldset: Optional[Path] = typer.Option(None, help="gold set for before/after recall@k and MRR"),
    k: int = typer.Option(10, min=1, help="retrieval cutoff for effect.md"),
    before_run: Optional[Path] = typer.Option(
        None, help="pre-overlay run directory or manifest.json for objective delta"
    ),
    after_run: Optional[Path] = typer.Option(
        None, help="post-overlay run directory or manifest.json for objective delta"
    ),
) -> None:
    """Turn findings into a reversible resolution overlay and measure its retrieval effect."""
    from llb.conflicts.resolution_io import (
        create_resolution_artifacts,
        infer_corpus_root,
        install_overlay,
        rollback_overlay,
    )
    from llb.core.paths import resolve_data_dir
    from llb.core.store_generations import generation_timestamp

    if policy not in POLICIES:
        raise typer.BadParameter(f"unknown policy {policy!r}; choose one of {', '.join(POLICIES)}")
    if rollback:
        if corpus is None:
            raise typer.BadParameter("--rollback requires --corpus")
        out_dir = out or resolve_data_dir() / "corpus-conflicts" / generation_timestamp()
        removed = rollback_overlay(corpus)
        typer.echo(
            f"[resolve-conflicts] overlay removed: {removed}"
            if removed is not None
            else "[resolve-conflicts] no applied overlay to remove"
        )
        _refresh_and_report(
            corpus,
            store,
            goldset,
            k,
            out_dir / "effect.md",
            None,
            before_run,
            after_run,
            0,
            "rollback",
        )
        return
    if findings is None:
        raise typer.BadParameter("--findings is required unless --rollback is used")

    corpus_root = infer_corpus_root(findings, corpus)
    out_dir = out or findings.parent
    plan, overlay, paths = create_resolution_artifacts(
        findings, out_dir, policy=policy, corpus_root=corpus_root, reviewed=reviewed
    )
    typer.echo(f"[resolve-conflicts] plan: {paths['plan']}")
    typer.echo(f"[resolve-conflicts] overlay: {paths['overlay']}")
    review_count = sum(item.get("status") == "review_required" for item in plan.get("items", []))
    if review_count:
        typer.echo(f"[resolve-conflicts] review required: {review_count} -> {paths['review']}")
    applied_path = None
    if apply:
        applied_path = install_overlay(corpus_root, overlay, plan)
        typer.echo(f"[resolve-conflicts] applied overlay: {applied_path}")
    _refresh_and_report(
        corpus_root,
        store if apply else None,
        goldset,
        k,
        paths["effect"],
        applied_path,
        before_run,
        after_run,
        review_count,
        _effect_key(overlay),
    )


def _refresh_and_report(
    corpus: Path,
    store: Optional[Path],
    goldset: Optional[Path],
    k: int,
    effect_path: Path,
    overlay_path: Optional[Path],
    before_run: Optional[Path],
    after_run: Optional[Path],
    unresolved_reviews: int,
    effect_key: str,
) -> None:
    from llb.conflicts.resolution_effect import objective_from_manifest, write_effect
    from llb.core.store_generations import generation_timestamp
    from llb.goldset.schema import load_goldset
    from llb.rag.refresh.drift import measure_drift
    from llb.rag.refresh.store_refresh import refresh_vector_store

    drift = None
    if store is not None:
        result = refresh_vector_store(store, corpus, timestamp=generation_timestamp())
        if result.refreshed:
            typer.echo(f"[resolve-conflicts] refreshed store -> {result.generation_dir}")
            if goldset is not None:
                drift = measure_drift(
                    result.old_store, result.new_store, load_goldset(goldset), k=k
                )
        else:
            typer.echo("[resolve-conflicts] store already reflects the overlay state")
    write_effect(
        effect_path,
        drift,
        before_objective=objective_from_manifest(before_run),
        after_objective=objective_from_manifest(after_run),
        overlay_path=overlay_path,
        unresolved_reviews=unresolved_reviews,
        effect_key=effect_key,
    )
    typer.echo(f"[resolve-conflicts] effect: {effect_path}")


def _effect_key(overlay: dict[str, Any]) -> str:
    from llb.conflicts.overlay import overlay_fingerprint

    value = overlay_fingerprint(overlay)
    assert value is not None
    return value
