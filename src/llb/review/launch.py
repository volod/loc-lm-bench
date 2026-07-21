"""Optional-dependency bridge used by established review commands."""

from pathlib import Path

from llb.review.core import ReviewAdapter


def try_workbench(path: Path | str, *, start: int | None = None) -> ReviewAdapter | None:
    """Run Textual when installed; otherwise let the established terminal loop take over."""
    try:
        from llb.review.workbench import run_workbench
    except ModuleNotFoundError as exc:
        if exc.name == "textual" or (exc.name or "").startswith("textual."):
            return None
        raise
    return run_workbench(path, start=start)
