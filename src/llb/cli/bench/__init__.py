"""Benchmark category commands, each rendered under its own Tier.

Importing each submodule registers its @app.command handlers on the shared Typer app.
"""

from llb.cli.bench import (  # noqa: F401
    category_agentic,
    category_agentic_context_sweep,
    category_analysis,
    category_structured,
    category_tasks,
    category_tooling,
    knowledge_cutoff,
    knowledge_cutoff_ua,
    misc,
)
