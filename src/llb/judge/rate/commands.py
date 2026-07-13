"""Calibration-rater command vocabulary and parsing."""

from dataclasses import dataclass

RATING_MIN = 1
RATING_MAX = 5
RATING_ANCHORS: dict[int, str] = {
    1: "wrong / unfaithful",
    2: "mostly wrong",
    3: "partially correct",
    4: "mostly correct",
    5: "fully correct + faithful",
}

RATE = "rate"
ANSWER = "answer"
NOTE = "note"
NEXT = "next"
PREV = "prev"
JUMP = "jump"
UNRATED = "unrated"
CLEAR = "clear"
HELP = "help"
QUIT = "quit"
UNKNOWN = "unknown"

_ESC = "\x1b"
_ARROWS = {f"{_ESC}[A": PREV, f"{_ESC}[D": PREV, f"{_ESC}[B": NEXT, f"{_ESC}[C": NEXT}

PROMPT_HINT = "[1-5]=rate (1=wrong..5=correct)  a=answer  n=next  p=prev  j<N>=jump  ?=help  q=quit"


@dataclass(frozen=True)
class Command:
    """A parsed rating command with an optional rating or jump target."""

    kind: str
    value: int | None = None
    raw: str = ""


_SIMPLE_COMMANDS: dict[str, str] = {
    "n": NEXT,
    "p": PREV,
    "b": PREV,
    "u": UNRATED,
    "c": CLEAR,
    "q": QUIT,
    "quit": QUIT,
    "?": HELP,
    "h": HELP,
    "help": HELP,
    "a": ANSWER,
    "answer": ANSWER,
    "note": NOTE,
}


def parse_command(raw: str) -> Command:
    """Parse one prompt line into a command without performing I/O."""
    text = raw.strip()
    if not text:
        return Command(NEXT)
    if text in _ARROWS:
        return Command(_ARROWS[text])
    kind = _SIMPLE_COMMANDS.get(text.lower())
    if kind is not None:
        return Command(kind)
    if text.lower().startswith("j"):
        target = text[1:].strip()
        return Command(JUMP, value=int(target)) if target.isdigit() else Command(UNKNOWN, raw=text)
    if text.isdigit():
        value = int(text)
        if RATING_MIN <= value <= RATING_MAX:
            return Command(RATE, value=value)
    return Command(UNKNOWN, raw=text)
