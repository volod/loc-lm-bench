"""Focused verify commands implementation."""

from dataclasses import dataclass
from llb.goldset.verify_base import REJECT_CODES

CHECK_KEYS: dict[str, str] = {
    "g": "chk_grounded",
    "a": "chk_answerable",
    "r": "chk_reference",
    "p": "chk_planted",
}

CHECK_LABEL: dict[str, str] = {
    "chk_grounded": "span grounded (offsets really support the answer/label)",
    "chk_answerable": "answerable + non-circular (question doesn't leak its answer)",
    "chk_reference": "reference answer correct",
    "chk_planted": "planted labels match the doc (synthetic only)",
}

CHECK = "check"

ACCEPT_CMD = "accept"

REJECT_CMD = "reject"

EDIT = "edit"

NOTE = "note"

NEXT = "next"

PREV = "prev"

JUMP = "jump"

UNDECIDED = "undecided"

CLEAR = "clear"

HELP = "help"

QUIT = "quit"

UNKNOWN = "unknown"

_ESC = "\x1b"

_ARROWS = {f"{_ESC}[A": PREV, f"{_ESC}[D": PREV, f"{_ESC}[B": NEXT, f"{_ESC}[C": NEXT}

_SIMPLE_COMMANDS = {
    "": NEXT,
    "n": NEXT,
    "b": PREV,
    "u": UNDECIDED,
    "c": CLEAR,
    "y": ACCEPT_CMD,
    "x": REJECT_CMD,
    "e": EDIT,
    "w": EDIT,
    "q": QUIT,
    "quit": QUIT,
    "?": HELP,
    "h": HELP,
    "help": HELP,
    "o": NOTE,
    "note": NOTE,
}

PROMPT_HINT = (
    "decide: y=accept, x=reject (code inferred), x <code>=coded reject; "
    "checks: g/a/r/p=PASS, G/A/R/P=FAIL\n"
    "edit/nav: e/w=edit answer, o=note, c=clear, Enter/n=next, b=prev, u=undecided, "
    "j<N>=jump, ?=help, q=quit"
)


@dataclass(frozen=True)
class Command:
    """A parsed prompt line. For CHECK, `field` is the column and `value` is pass/fail; for JUMP,
    `value` is the target; `raw` carries offending text for UNKNOWN so the loop can echo it."""

    kind: str
    field: str = ""
    value: bool | int | None = None
    raw: str = ""


def _parse_simple_command(s: str) -> Command | None:
    kind = _SIMPLE_COMMANDS.get(s.lower())
    return Command(kind) if kind is not None else None


def _parse_check_command(s: str) -> Command | None:
    if s in CHECK_KEYS:
        return Command(CHECK, field=CHECK_KEYS[s], value=True)
    if s.lower() in CHECK_KEYS and s.isupper():
        return Command(CHECK, field=CHECK_KEYS[s.lower()], value=False)
    return None


def _parse_jump_command(s: str) -> Command | None:
    if not s.lower().startswith("j"):
        return None
    rest = s[1:].strip()
    if rest.isdigit():
        return Command(JUMP, value=int(rest))
    return Command(UNKNOWN, raw=s)


def _parse_reject_code_command(s: str) -> Command | None:
    """`x <code>` -- reject with an explicit coded reason (bare `x` stays a simple command)."""
    if not s.lower().startswith("x "):
        return None
    return Command(REJECT_CMD, field=s[2:].strip().lower())


def parse_command(raw: str) -> Command:
    """Parse one prompt line into a `Command` (pure; no I/O, no state).

    Empty / `n` = next; `b`/up/left arrow = previous; `g/a/r/p` mark a check PASS, the uppercase
    forms FAIL; `y` = accept, `x` = reject (code inferred from failed checks), `x <code>` = reject
    with an explicit coded reason; `e` = edit the reference answer (re-grounded immediately);
    `note` = edit a note; `j <N>`/`jN` = jump; `u` = next undecided; `c` = clear this item;
    `?`/`h` = help; `q` = save + quit.
    """
    s = raw.strip()
    if s in _ARROWS:
        return Command(_ARROWS[s])
    parsers = (
        _parse_simple_command,
        _parse_check_command,
        _parse_reject_code_command,
        _parse_jump_command,
    )
    for parser in parsers:
        cmd = parser(s)
        if cmd is not None:
            return cmd
    return Command(UNKNOWN, raw=s)


def help_text() -> str:
    """The command + check reference shown by `?`."""
    checks = "\n".join(
        f"  {key} / {key.upper()}  {CHECK_LABEL[CHECK_KEYS[key]]}" for key in CHECK_KEYS
    )
    return "\n".join(
        [
            "checks (lowercase = PASS, uppercase = FAIL):",
            checks,
            "decision:",
            "  y        accept (within tolerance)",
            "  x        reject (code inferred from failed checks)",
            f"  x <code> reject with an explicit code: {', '.join(REJECT_CODES)}",
            "edits:",
            "  e / w    edit the reference answer (accept-with-edit; re-grounded immediately)",
            "  o / note edit a note",
            "  c        clear this item's marks",
            "navigation:",
            "  n/Enter  next                           b        previous",
            "  j <N>    jump to item N                 u        next undecided",
            "  ?/h      this help                      q        save + quit",
            "verify INDEPENDENTLY against the corpus window -- do not anchor to the cross-check.",
        ]
    )
