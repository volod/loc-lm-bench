"""Review-card rendering + command parsing for the interactive verification session.

The pure presentation half of `verify_session.py`: parse one prompt line into a `Command`, render
a sampled item as a terminal review card (standard QA layout or the dense chain layout), and the
`?` help text. No I/O, no session state, so it is fully unit-testable. `verify_session.py` (the
session loop) builds on these; both share the worksheet schema from `verify.py`.
"""

import json
from dataclasses import dataclass

from llb.goldset.verify import CHECK_COLS, KIND_CHAINS, REJECT_CODES

_CHAIN_SEPARATOR = "+" * 64
_CHAIN_DEFAULT_WIDTH = 120
_CHAIN_MIN_VALUE_WIDTH = 24
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_QUESTION = "\033[1;36m"
_ANSI_ANSWER = "\033[1;32m"
_ANSI_SOURCE = "\033[33m"
_ANSI_DEPENDENCY = "\033[35m"

# The four checks, in card order, mapped to the keystroke that marks them. Lowercase = PASS,
# uppercase = FAIL. `planted` only applies to synthetic items (blank/N/A for real ones).
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

# Command kinds returned by `parse_command`.
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
# Command keys mirror the external-RAG review session (`llb.scoring.external_rag_session`) where
# the two tools do the same thing: o=note and w=edit-the-answer are shared aliases, and the
# navigation row (Enter/n, b, u, j<N>, ?, q) is identical. The decision keys necessarily differ:
# here a/r/p mark CHECKS (answerable/reference/planted), so accept/reject are y/x.
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


def _field(row: dict[str, str], name: str, blank: str) -> str:
    value = (row.get(name) or "").strip()
    return value if value else blank


def _is_synthetic_row(row: dict[str, str]) -> bool:
    return (row.get("synthetic") or "").strip().lower() == "true"


def _is_chain_row(row: dict[str, str]) -> bool:
    return (row.get("item_kind") or "").strip() == KIND_CHAINS


def _indent(text: str, prefix: str = "    ") -> str:
    if not text:
        return prefix.rstrip()
    return "\n".join(prefix + line for line in text.splitlines())


def _one_line(value: object) -> str:
    return " ".join(str(value or "").split())


def _truncate(value: object, limit: int, *, blank: str = "(none)") -> str:
    text = _one_line(value) or blank
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)].rstrip() + "..."


def _context_excerpt(value: object, limit: int) -> str:
    text = _one_line(value) or "(missing)"
    cited_start = text.find(">>>")
    cited_end = text.find("<<<", cited_start + 3)
    if cited_start < 0 or cited_end < 0:
        return _truncate(text, limit, blank="(missing)")
    cited_end += 3
    cited = text[cited_start:cited_end]
    if len(cited) >= limit:
        return _truncate(cited, limit, blank="(missing)")
    context_budget = max(0, limit - len(cited) - 6)
    before = context_budget // 2
    after = context_budget - before
    start = max(0, cited_start - before)
    end = min(len(text), cited_end + after)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix


def _color(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{_ANSI_RESET}" if enabled else text


def _chain_field(
    label: str,
    value: object,
    *,
    width: int,
    color: str = "",
    use_color: bool = False,
    blank: str = "(none)",
) -> str:
    available = max(_CHAIN_MIN_VALUE_WIDTH, width - len(label) - 1)
    line = f"{label} {_truncate(value, available, blank=blank)}"
    return _color(line, color, use_color and bool(color))


def _format_chain_steps(
    row: dict[str, str], *, color: bool = False, width: int = _CHAIN_DEFAULT_WIDTH
) -> list[str]:
    raw = (row.get("chain_steps") or "").strip()
    if not raw:
        return []
    try:
        steps = json.loads(raw)
    except json.JSONDecodeError:
        return ["== chain_steps: (invalid JSON)"]
    if not isinstance(steps, list):
        return ["== chain_steps: (invalid JSON)"]
    valid_steps = [step for step in steps if isinstance(step, dict)]
    lines: list[str] = []
    for index, step in enumerate(valid_steps, start=1):
        order = str(step.get("order", ""))
        page = str(step.get("page_citation") or "(none)")
        doc = _truncate(step.get("span_doc_id"), max(_CHAIN_MIN_VALUE_WIDTH, width // 2))
        header = f"STEP {order or index}/{len(valid_steps)} | doc={doc} | page={page}"
        lines.append(_color(header, _ANSI_BOLD, color))
        lines.append(
            _chain_field(
                "Q:", step.get("question"), width=width, color=_ANSI_QUESTION, use_color=color
            )
        )
        lines.append(
            _chain_field(
                "A:",
                step.get("reference_answer"),
                width=width,
                color=_ANSI_ANSWER,
                use_color=color,
            )
        )
        lines.append(
            _chain_field(
                "SOURCE:",
                step.get("span_text"),
                width=width,
                color=_ANSI_SOURCE,
                use_color=color,
                blank="(missing)",
            )
        )
        dependency = str(step.get("dependency_note") or "").strip()
        if dependency:
            lines.append(
                _chain_field(
                    "DEPENDENCY:",
                    dependency,
                    width=width,
                    color=_ANSI_DEPENDENCY,
                    use_color=color,
                )
            )
        context = str(step.get("context") or "").strip()
        if context:
            available = max(_CHAIN_MIN_VALUE_WIDTH, width - len("CONTEXT:") - 1)
            lines.append(f"CONTEXT: {_context_excerpt(context, available)}")
    return lines


def _chain_checks(row: dict[str, str]) -> str:
    checks = [
        f"{col.removeprefix('chk_')}={_field(row, col, '(unchecked)')}"
        for col in CHECK_COLS
        if col != "chk_planted"
    ]
    return "CHECKS: " + " | ".join(checks)


def _chain_card_lines(
    row: dict[str, str], position: int, total: int, decided: int, *, color: bool, width: int
) -> list[str]:
    """Dense one-line-per-field chain layout."""
    decision = _field(row, "decision", "(undecided)")
    lines = [
        _CHAIN_SEPARATOR,
        f"CHAIN {position}/{total} | id={row.get('item_id', '')} | decided={decided} "
        f"remaining={total - decided} | decision={decision}",
        "REVIEW: compare A with SOURCE; confirm Q is answered and later steps add context.",
    ]
    lines.extend(_format_chain_steps(row, color=color, width=width))
    lines.append(_chain_checks(row))
    return lines


def _standard_card_lines(row: dict[str, str], position: int, total: int, decided: int) -> list[str]:
    """Classic multi-line QA card: header, meta, question/answer/span, context, checks."""
    rank = (row.get("retrieval_rank") or "").strip() or "(none)"
    page = (row.get("page_citation") or "").strip() or "(none)"
    lines = [
        "===== goldset verification review =====",
        f"item {position}/{total} (decided {decided}, remaining {total - decided})",
        f"== id: {row.get('item_id', '')}",
        f"== meta: stratum={row.get('stratum', '')} synthetic={row.get('synthetic', '')} "
        f"retrieval_rank={rank}",
    ]
    reviewer = (row.get("reviewer_id") or "").strip()
    if reviewer:
        lines.append(f"== reviewer: {reviewer}")
    priors = (row.get("prior_decisions") or "").strip()
    if priors:
        lines.append(f"== prior_decisions: {priors} -- decide independently, then compare")
    lines += [
        "",
        f"== question: {row.get('question', '')}",
        f"== reference_answer: {row.get('reference_answer', '')}",
        f"== span: doc={row.get('span_doc_id', '')} page={page}",
        "== context (cited span between >>> <<<)",
        _indent(row.get("context", "") or "(missing)"),
        "== checks:",
    ]
    synthetic = _is_synthetic_row(row)
    for col in CHECK_COLS:
        if col == "chk_planted" and not synthetic:
            continue
        lines.append(f"    {col:<15}: {_field(row, col, '(unchecked)')}  -- {CHECK_LABEL[col]}")
    lines.append(f"== decision: {_field(row, 'decision', '(undecided)')}")
    return lines


def _card_annotation_lines(row: dict[str, str], show_crosscheck: bool) -> list[str]:
    """Optional trailing lines: edit, reject code, note, cross-check summary."""
    lines: list[str] = []
    for column, label in (
        ("edited_answer", "edited_answer"),
        ("reject_code", "reject_code"),
        ("human_note", "human_note"),
    ):
        value = (row.get(column) or "").strip()
        if value:
            lines.append(f"== {label}: {value}")
    if show_crosscheck:
        cc = "  ".join(
            f"{key}={_field(row, f'cc_{key}', '?')}"
            for key in ("grounded", "non_circular", "supported", "answerable")
        )
        lines.append(f"== crosscheck: {cc}  note={_field(row, 'cc_note', '')}")
    return lines


def format_card(
    row: dict[str, str],
    position: int,
    total: int,
    decided: int,
    *,
    show_crosscheck: bool = False,
    color: bool = False,
    width: int = _CHAIN_DEFAULT_WIDTH,
) -> str:
    """Render a review card; chain rows use a dense one-line-per-field terminal layout."""
    if _is_chain_row(row):
        lines = _chain_card_lines(row, position, total, decided, color=color, width=width)
    else:
        lines = _standard_card_lines(row, position, total, decided)
    lines.extend(_card_annotation_lines(row, show_crosscheck))
    return "\n".join(lines)


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
