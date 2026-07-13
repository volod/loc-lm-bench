"""Prompt-command vocabulary and parsing for the external-RAG review session."""

from dataclasses import dataclass

from llb.scoring.external_rag import (
    HUMAN_DECISION_ACCEPT,
    HUMAN_DECISION_PARTIAL,
    HUMAN_DECISION_REJECT,
)

NEXT = "next"
PREV = "prev"
JUMP = "jump"
UNSCORED = "unscored"
CLEAR = "clear"
HELP = "help"
QUIT = "quit"
NOTE = "note"
CORRECT = "correct"
DECISION = "decision"
SCORE = "score"
UNKNOWN = "unknown"

_ESC = "\x1b"
_ARROWS = {f"{_ESC}[A": PREV, f"{_ESC}[D": PREV, f"{_ESC}[B": NEXT, f"{_ESC}[C": NEXT}
_SIMPLE_COMMANDS = {
    "": NEXT,
    "n": NEXT,
    "b": PREV,
    "u": UNSCORED,
    "c": CLEAR,
    "q": QUIT,
    "quit": QUIT,
    "?": HELP,
    "h": HELP,
    "help": HELP,
    "o": NOTE,
    "note": NOTE,
    "w": CORRECT,
    "corr": CORRECT,
    "correct": CORRECT,
    "a": DECISION,
    "accept": DECISION,
    "p": DECISION,
    "partial": DECISION,
    "r": DECISION,
    "reject": DECISION,
}

PROMPT_HINT = (
    "score: a=accept(1.0), p=partial(0.5), r=reject(0.0), s 0..1=custom score\n"
    "edit/nav: o=note, w=corrected answer, c=clear, Enter/n=next, "
    "b=prev, u=unscored, j<N>=jump, ?=help, q=quit"
)


@dataclass(frozen=True)
class Command:
    """A parsed prompt line."""

    kind: str
    value: str | float | int | None = None
    raw: str = ""


def parse_command(raw: str) -> Command:
    """Parse one prompt command."""
    s = raw.strip()
    if s in _ARROWS:
        return Command(_ARROWS[s])
    lower = s.lower()
    if lower.startswith("j"):
        rest = lower[1:].strip()
        if rest.isdigit():
            return Command(JUMP, int(rest))
        return Command(UNKNOWN, raw=s)
    if lower.startswith("s "):
        return _parse_score(lower[2:].strip(), s)
    if lower in _SIMPLE_COMMANDS:
        if _SIMPLE_COMMANDS[lower] == DECISION:
            return Command(DECISION, _decision_from_text(lower))
        return Command(_SIMPLE_COMMANDS[lower])
    score_command = _parse_score(lower, s)
    if score_command.kind != UNKNOWN:
        return score_command
    return Command(UNKNOWN, raw=s)


def _parse_score(value: str, raw: str) -> Command:
    try:
        score = float(value)
    except ValueError:
        return Command(UNKNOWN, raw=raw)
    if 0.0 <= score <= 1.0:
        return Command(SCORE, score)
    return Command(UNKNOWN, raw=raw)


def _decision_from_text(value: str) -> str:
    if value in ("a", "accept"):
        return HUMAN_DECISION_ACCEPT
    if value in ("p", "partial"):
        return HUMAN_DECISION_PARTIAL
    return HUMAN_DECISION_REJECT


def help_text() -> str:
    """Command reference for the review prompt."""
    return "\n".join(
        [
            "score commands:",
            "  a        accept, score=1.0",
            "  p        partial, score=0.5",
            "  r        reject, score=0.0",
            "  s 0..1   set an explicit human score, for example: s 0.8",
            "edits:",
            "  o        edit human_notes",
            "  w        edit human_corrected_answer",
            "  c        clear this row's human fields",
            "navigation:",
            "  n/Enter  next",
            "  b        previous",
            "  j <N>    jump to row N",
            "  u        first unscored row",
            "  q        save and quit",
            "The JSONL is saved after each edit. CSV/report are written only after all rows are "
            "scored.",
        ]
    )
