"""Shared CLI runtime for every llb entrypoint: graceful Ctrl-C + crash logging.

The `llb` Typer app (`run_typer`) and the `python -m llb.*` module CLIs (`run`) both route
through here, so a Ctrl-C shuts down cleanly with one message and exit code 130, and an
unexpected error is logged with a traceback (set `LLB_LOG=debug` for more) instead of dumping
a raw stack trace. Long-running commands clean up their own resources (e.g. `run-eval` kills
the backend via the launcher context manager) as the interrupt propagates through here.
"""

import logging
import os
import sys

from llb import env
from typing import Any, Callable, Optional

INTERRUPT_EXIT = 130  # 128 + SIGINT -- the conventional exit code for Ctrl-C
_LOG = logging.getLogger("llb")


# Chatty third-party loggers (per-request HTTP, model loading, faiss probing) are pinned to
# WARNING so the pipeline log stays readable -- raise everything with LLB_LOG=debug.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "filelock",
    "faiss",
    "faiss.loader",
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
)


def configure_logging() -> None:
    """Set up root logging once (idempotent). `LLB_LOG=debug` raises the level + the noise."""
    if logging.getLogger().handlers:
        return
    debug = os.environ.get(env.LLB_LOG, "").lower() in ("debug", "1", "true")
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not debug:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


def _announce_interrupt() -> None:
    sys.stderr.write("\n[llb] interrupted (Ctrl-C) -- shutting down cleanly.\n")
    sys.stderr.flush()


def _report_crash() -> None:
    _LOG.exception("unhandled error")
    sys.stderr.write("[llb] FAILED -- see the traceback above (set LLB_LOG=debug for more).\n")
    sys.stderr.flush()


def run(entry: Callable[[], Optional[int]]) -> int:
    """Run an argparse-style `main() -> int|None` with shared Ctrl-C + crash handling."""
    configure_logging()
    try:
        return entry() or 0
    except KeyboardInterrupt:
        _announce_interrupt()
        return INTERRUPT_EXIT
    except SystemExit as exc:  # argparse usage errors etc. -- preserve the code
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else 1
    except Exception:
        _report_crash()
        return 1


def run_typer(app: Any) -> None:
    """Run a Typer app with shared Ctrl-C + crash handling. Click swallows SIGINT in its
    standalone mode, so we drive the app in non-standalone mode and translate the result."""
    import click

    configure_logging()
    try:
        app(standalone_mode=False)
    except (KeyboardInterrupt, click.exceptions.Abort):
        _announce_interrupt()
        raise SystemExit(INTERRUPT_EXIT) from None
    except click.exceptions.Exit as exc:  # typer.Exit(code) and --help
        raise SystemExit(exc.exit_code) from None
    except click.ClickException as exc:  # usage errors
        exc.show()
        raise SystemExit(exc.exit_code) from None
    except Exception:
        _report_crash()
        raise SystemExit(1) from None
