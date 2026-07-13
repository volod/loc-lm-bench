"""Benchmark category commands, each rendered under its own Tier.

Importing each submodule registers its @app.command handlers on the shared Typer app.
"""

from llb.cli.bench import (  # noqa: F401
    category_analysis,
    category_tasks,
    knowledge_cutoff,
    misc,
)
