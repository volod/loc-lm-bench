"""Fine-tuning, adapter lifecycle, and local self-improvement commands.

Importing each submodule registers its @app.command handlers on the shared Typer app.
"""

from llb.cli.finetune import adapters, improve, training  # noqa: F401
