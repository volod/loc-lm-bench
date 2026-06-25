"""loc-lm-bench CLI command modules (Typer).

Import submodules so their @app.command handlers register on the shared Typer app.
"""

from llb.cli.app import app

import llb.cli.bench  # noqa: F401
import llb.cli.eval  # noqa: F401
import llb.cli.inference  # noqa: F401
import llb.cli.models  # noqa: F401
import llb.cli.prep  # noqa: F401
import llb.cli.rag  # noqa: F401
import llb.cli.ui  # noqa: F401

__all__ = ["app"]
