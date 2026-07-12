"""Gold-set and corpus preparation commands.

Split by functional area; importing each submodule registers its @app.command handlers on the
shared Typer app (same registration contract as the former single prep.py module).
"""

from llb.cli.prep import (  # noqa: F401
    benchmarks,
    corpus,
    curation,
    draft,
    goldset,
    security,
)
