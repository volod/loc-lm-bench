"""Prompt I/O primitives and the internal end-of-session control signal."""

import sys
from collections.abc import Callable, Iterator

from llb.scoring.external_rag_session.commands import _ESC, Command


class _Quit(Exception):
    """Internal end-of-session signal."""


def _read(prompt: str, it: Iterator[str] | None, emit: Callable[[str], None]) -> str:
    if it is None:
        return input(prompt)
    emit(prompt)
    try:
        return next(it)
    except StopIteration as exc:
        raise _Quit from exc


def _default_output(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _emit_unknown(cmd: Command, emit: Callable[[str], None]) -> None:
    if cmd.raw.startswith(_ESC):
        emit("[score-external-rag] arrow keys garbled -- use n (next) / b (prev).")
    else:
        emit(f"[score-external-rag] not a command: {cmd.raw!r} (? for help).")
