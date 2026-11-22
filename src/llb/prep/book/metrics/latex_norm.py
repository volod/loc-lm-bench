"""One canonical spelling for the LaTeX equivalences the harness treats as the same formula.

Optional braces on a one-token script, ``\\left`` and ``\\right``, spacing commands, and the
aliases below collapse before any comparison. ``plain_form`` is the reading a text-only lane
prints, and ``script_tokens`` is the structure the script confusion class counts.
"""

import re

_TOKEN = re.compile(r"\\[a-zA-Z]+|\\.|\s+|[{}^_=+\-*/()'\.,]|[0-9]+(?:\.[0-9]+)?|[A-Za-z]|[^\s]")
_COMMENT = re.compile(r"(?<!\\)%.*")
_TAG = re.compile(r"\\tag\*?\{[^{}]*\}")
_BEGIN = re.compile(r"\\(?:begin|end)\{[^{}]*\}")
_DELIMITER = (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
_LEFT_RIGHT = re.compile(r"\\(?:left|right|middle|[bB]igg?[lLr])\b")
_UNWRAP = frozenset({r"\mathrm", r"\text", r"\operatorname", r"\mathbf", r"\mathit", r"\textrm"})
_ALIAS = {
    r"\le": r"\leq",
    r"\ge": r"\geq",
    r"\ne": r"\neq",
    r"\to": r"\rightarrow",
    r"\cdotp": r"\cdot",
    r"\dfrac": r"\frac",
    r"\tfrac": r"\frac",
    r"\ldots": r"\dots",
}
_SPACING = frozenset({r"\,", r"\;", r"\:", r"\!", r"\ ", r"\quad", r"\qquad", "~"})
_DASHES = str.maketrans({"\u2212": "-", "\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2014": "-"})
_PLAIN = {
    "alpha": "\u03b1",
    "beta": "\u03b2",
    "gamma": "\u03b3",
    "delta": "\u03b4",
    "Delta": "\u0394",
    "omega": "\u03c9",
    "Omega": "\u03a9",
    "mu": "\u03bc",
    "nu": "\u03bd",
    "pi": "\u03c0",
    "theta": "\u03b8",
    "lambda": "\u03bb",
    "ln": "ln",
    "log": "log",
    "sin": "sin",
    "cos": "cos",
    "cdot": "",
    "times": "x",
    "left": "",
    "right": "",
}
_FRAC = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_SCRIPT_ARG = re.compile(r"[_^]\{([^{}]*)\}")


class LatexError(ValueError):
    """The string is not a complete formula in the subset this harness parses."""


def canonicalize(source: str) -> str:
    """One spelling for the equivalences this harness treats as the same formula."""
    text = _strip_wrappers(_COMMENT.sub("", source).translate(_DASHES))
    text = _TAG.sub("", text)
    text = _BEGIN.sub("", text)
    text = text.replace("&", "").replace(r"\\", "")
    text = _LEFT_RIGHT.sub("", text)
    tokens = [
        token for token in _TOKEN.findall(text) if not token.isspace() and token not in _SPACING
    ]
    tokens = [_ALIAS.get(token, token) for token in tokens]
    return _join(_normalize_scripts(_unwrap(tokens)))


def looks_like_latex(text: str) -> bool:
    """True when the reading uses a command or a script, so it is compared as LaTeX."""
    return bool(re.search(r"\\|[_^]", text))


def plain_form(source: str) -> str:
    """The reading a text-only lane would print for one canonical term or a plain string."""
    text = _FRAC.sub(lambda match: f"{match.group(1)}/{match.group(2)}", source)

    def command(match: re.Match[str]) -> str:
        return _PLAIN.get(match.group(1), "")

    text = re.sub(r"\\([a-zA-Z]+)", command, text)
    text = _SCRIPT_ARG.sub(lambda match: match.group(1), text)
    return re.sub(r"[\s{}\\]", "", text)


def script_tokens(source: str) -> list[tuple[str, str, str]]:
    """``(base, subscript, superscript)`` for every scripted atom in the canonical form."""
    tokens = _tokens(canonicalize(source))
    found: list[tuple[str, str, str]] = []
    index = 0
    while index < len(tokens):
        if not _is_base(tokens[index]):
            index += 1
            continue
        base = tokens[index]
        sub = sup = ""
        index += 1
        while index < len(tokens) and tokens[index] in "_^":
            kind = tokens[index]
            group, index = _script_group(tokens, index + 1)
            if kind == "_":
                sub = group
            else:
                sup = group
        if sub or sup:
            found.append((base, sub, sup))
    return found


def _tokens(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text) if not token.isspace()]


def _strip_wrappers(text: str) -> str:
    current = text.strip()
    changed = True
    while changed:
        changed = False
        for opener, closer in _DELIMITER:
            if (
                current.startswith(opener)
                and current.endswith(closer)
                and len(current) > len(opener) + len(closer)
            ):
                current = current[len(opener) : -len(closer)].strip()
                changed = True
                break
    return current


def _unwrap(tokens: list[str]) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index] in _UNWRAP and index + 1 < len(tokens) and tokens[index + 1] == "{":
            inner, index = _group_tokens(tokens, index + 1)
            output.extend(_unwrap(inner))
            continue
        output.append(tokens[index])
        index += 1
    return output


def _normalize_scripts(tokens: list[str]) -> list[str]:
    """``x_a`` and ``x_{{a}}`` both become ``x_{a}``. An unbraced script takes one TeX token."""
    output: list[str] = []
    index = 0
    while index < len(tokens):
        output.append(tokens[index])
        if tokens[index] not in "_^" or index + 1 >= len(tokens):
            index += 1
            continue
        if tokens[index + 1] != "{":
            output.extend(["{", tokens[index + 1], "}"])
            index += 2
            continue
        inner, index = _group_tokens(tokens, index + 1)
        while len(inner) == 3 and inner[0] == "{" and inner[-1] == "}":
            inner = inner[1:-1]
        output.append("{")
        output.extend(inner)
        output.append("}")
    return output


def _group_tokens(tokens: list[str], start: int) -> tuple[list[str], int]:
    """Tokens inside the braces that begin at ``start``. ``start`` points at ``{``."""
    if start >= len(tokens) or tokens[start] != "{":
        raise LatexError("a group must start with a brace")
    depth = 0
    index = start
    while index < len(tokens):
        if tokens[index] == "{":
            depth += 1
        elif tokens[index] == "}":
            depth -= 1
            if depth == 0:
                return tokens[start + 1 : index], index + 1
        index += 1
    raise LatexError("a group runs past the end of the formula")


def _script_group(tokens: list[str], start: int) -> tuple[str, int]:
    if start >= len(tokens):
        raise LatexError("a script is missing its argument")
    if tokens[start] == "{":
        inner, end = _group_tokens(tokens, start)
        return _join(inner), end
    return tokens[start], start + 1


def _join(tokens: list[str]) -> str:
    parts: list[str] = []
    for token in tokens:
        if (
            parts
            and parts[-1].startswith("\\")
            and parts[-1][1:].isalpha()
            and (token[:1].isalpha() or token.startswith("\\"))
        ):
            parts.append(" ")
        parts.append(token)
    return "".join(parts)


def _is_base(token: str) -> bool:
    return token[:1].isalnum() or token.startswith("\\") or ord(token[0]) > 127
