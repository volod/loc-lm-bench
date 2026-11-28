"""The fixed-settings book mode of compare-graph-fusion."""

from pathlib import Path

import typer
from click.core import ParameterSource

from llb.cli.helpers import cli_error, load_config
from llb.core.paths import resolve_project_path
from llb.rag.book_comparison.graph_run import GraphReadingOptions, run_book_graph_reading
from llb.rag.book_comparison.metrics import ReadingSettings

_BOOK_OPTIONS = frozenset(
    {
        "book_corpus",
        "config",
        "goldset",
        "k",
        "split",
        "minimum_detectable_delta",
        "confidence",
        "resamples",
        "seed",
        "noise_floor",
        "noise_floor_replicates",
        "out_dir",
    }
)


def compare_book_graph(
    ctx: typer.Context,
    config: Path | None,
    goldset: Path | None,
    corpus: Path,
    settings: ReadingSettings,
    options: GraphReadingOptions,
) -> None:
    for name in ctx.params.keys() - _BOOK_OPTIONS:
        if ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE:
            cli_error(
                f"--{name.replace('_', '-')} is for the fusion sweep, not the fixed book reading"
            )
    if not 0 < settings.minimum_detectable_delta <= 1 or settings.resamples < 1:
        cli_error(
            "the book reading needs 0 < --minimum-detectable-delta <= 1 and positive --resamples"
        )
    cfg = load_config(config, goldset_path=goldset, corpus_root=resolve_project_path(corpus))
    try:
        target, slices = run_book_graph_reading(cfg, settings, options)
    except (OSError, ValueError, SystemExit) as exc:
        cli_error(str(exc))
    for name, reading in slices.items():
        typer.echo(f"[compare-graph-fusion] {name} n={reading['n']}")
    typer.echo(f"[compare-graph-fusion] book report -> {target / 'comparison.json'}")
