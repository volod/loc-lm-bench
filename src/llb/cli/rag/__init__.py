"""RAG index build and retrieval validation commands.

Importing each submodule registers its @app.command handlers on the shared Typer app.
"""

from llb.cli.rag import (  # noqa: F401
    compare_embeddings,
    compare_retrieval,
    compare_stores,
    duplicate_residue,
    fusion_calibration,
    fusion_evidence,
    graph_index,
    index,
    refresh,
    validate,
)
