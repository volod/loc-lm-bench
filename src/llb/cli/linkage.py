"""Record-linkage seam: fit or replay an identity decision over a JSONL record table."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app


@app.command("link-records")
def link_records_cmd(
    records: Path = typer.Option(..., help="JSONL record table, one record per line"),
    spec: Optional[Path] = typer.Option(
        None, help="JSON comparison/blocking specification (required unless --replay-from)"
    ),
    labels: Optional[Path] = typer.Option(
        None, help="JSONL reviewer labels ({left, right, match}); fits m and scores the curve"
    ),
    replay_from: Optional[Path] = typer.Option(
        None, help="re-score from a previous run bundle's saved model instead of fitting"
    ),
    data_dir: Optional[Path] = typer.Option(None, help="artifact root (default: $DATA_DIR)"),
    method: Optional[str] = typer.Option(None, help="artifact method directory name"),
    run: Optional[str] = typer.Option(None, help="run directory name (default: a fresh stamp)"),
    examples: int = typer.Option(5, help="how many clusters and pairs to print"),
) -> None:
    """Decide which records denote the same thing, and how sure the model is.

    Fits a Fellegi-Sunter model over the specification's comparisons: non-match parameters from
    randomly drawn pairs, match parameters from expectation-maximisation or from a reviewer label
    table when one is supplied. Publishes the specification, the blocking counts taken before the
    fit, the fitted parameters, every scored pair, and the identity clusters at the run's
    threshold -- plus the trained model, so `--replay-from` reproduces the same probabilities
    without re-fitting.

    Clusters are PROPOSED, never applied: nothing here rewrites a corpus, a gold set, or a graph.
    A match probability answers "same thing", never "these contradict"; it is not a conflict
    verdict and must not be reported as one.
    """
    from llb.core.paths import resolve_data_dir
    from llb.linkage.constants import METHOD
    from llb.linkage.run import (
        format_accuracy,
        format_pairs,
        format_summary,
        link_records,
        replay_records,
    )

    if spec is None and replay_from is None:
        typer.echo("[error] pass --spec <json> or --replay-from <run-bundle>", err=True)
        raise typer.Exit(code=2)
    root = resolve_data_dir(data_dir)
    if replay_from is not None:
        published = replay_records(records, replay_from, root, method=method or METHOD, run=run)
    else:
        assert spec is not None  # guarded above; keeps the type checker honest
        published = link_records(
            records, spec, root, labels_path=labels, method=method or METHOD, run=run
        )
    typer.echo(format_summary(published.result, examples))
    typer.echo(format_pairs(published.result, examples))
    typer.echo(format_accuracy(published.result))
    typer.echo(f"[link-records] wrote bundle -> {published.out_dir}")
