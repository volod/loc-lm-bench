"""Eval, screen, pipeline, and judge experiment commands.

Importing each submodule registers its @app.command handlers on the shared Typer app.
"""

from llb.cli.eval import (  # noqa: F401
    analysis,
    adoption_screen,
    answer_envelope,
    answer_quality,
    answer_validation,
    context_ablation,
    embedder_adoption,
    frontier_judge,
    judge,
    paired_reading_audit,
    query_robustness,
    restoration_sweep,
    run,
    screen,
    verbosity,
)
