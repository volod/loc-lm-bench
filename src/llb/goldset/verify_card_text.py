"""Focused verify card text implementation."""

_ANSI_RESET = "\033[0m"

_ANSI_BOLD = "\033[1m"

_ANSI_QUESTION = "\033[1;36m"

_ANSI_ANSWER = "\033[1;32m"

_ANSI_SOURCE = "\033[33m"

_ANSI_DEPENDENCY = "\033[35m"


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
