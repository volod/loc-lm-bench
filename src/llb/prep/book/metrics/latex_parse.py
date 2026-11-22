"""Recursive-descent check that a canonical formula is one complete expression.

The grammar covers the spellings this harness compares: numbers, letters, commands, groups,
parentheses, fractions, roots, juxtaposition, and the infix operators ``+``, ``-``, ``=``, ``/``,
``\\cdot``, and ``\\times``. It rejects a truncated command such as ``\\frac{a}{``.
"""

from llb.prep.book.metrics.latex_norm import LatexError, _tokens

_INFIX = frozenset({r"\cdot", r"\times"})


def consume(canonical: str) -> None:
    """Raise LatexError unless ``canonical`` is one complete expression."""
    tokens = _tokens(canonical)
    if not tokens:
        raise LatexError("a formula is empty")
    end = _parse_expr(tokens, 0)
    if end != len(tokens):
        raise LatexError("the formula has trailing tokens")


def _parse_expr(tokens: list[str], index: int) -> int:
    index = _parse_add(tokens, index)
    while index < len(tokens) and tokens[index] == "=":
        index = _parse_add(tokens, index + 1)
    return index


def _parse_add(tokens: list[str], index: int) -> int:
    index = _parse_mul(tokens, index)
    while index < len(tokens) and tokens[index] in "+-":
        index = _parse_mul(tokens, index + 1)
    return index


def _parse_mul(tokens: list[str], index: int) -> int:
    index = _parse_unary(tokens, index)
    while index < len(tokens) and (
        tokens[index] in _INFIX or tokens[index] == "/" or _starts(tokens, index)
    ):
        if tokens[index] in _INFIX or tokens[index] == "/":
            index += 1
        index = _parse_unary(tokens, index)
    return index


def _parse_unary(tokens: list[str], index: int) -> int:
    if index < len(tokens) and tokens[index] == "-":
        return _parse_unary(tokens, index + 1)
    return _parse_postfix(tokens, index)


def _parse_postfix(tokens: list[str], index: int) -> int:
    index = _parse_primary(tokens, index)
    while index < len(tokens) and tokens[index] in {"_", "^", "'"}:
        if tokens[index] == "'":
            index += 1
            continue
        index = _parse_primary(tokens, index + 1)
    return index


def _parse_primary(tokens: list[str], index: int) -> int:
    if index >= len(tokens):
        raise LatexError("the formula ended early")
    token = tokens[index]
    if token == "{":
        return _closing_brace(tokens, index)
    if token == "(":
        return _parse_paren(tokens, index)
    if token == r"\frac":
        return _closing_brace(tokens, _closing_brace(tokens, index + 1))
    if token == r"\sqrt":
        return _parse_sqrt(tokens, index)
    if _command(token):
        return _parse_command(tokens, index)
    if _atom(token):
        return index + 1
    raise LatexError(f"unexpected token {token}")


def _parse_paren(tokens: list[str], index: int) -> int:
    end = _parse_expr(tokens, index + 1)
    if end >= len(tokens) or tokens[end] != ")":
        raise LatexError("a parenthesis is unclosed")
    return end + 1


def _parse_sqrt(tokens: list[str], index: int) -> int:
    nxt = index + 1
    if nxt < len(tokens) and tokens[nxt] == "[":
        return _closing_brace(tokens, _closing(tokens, nxt, "[", "]"))
    if nxt < len(tokens) and tokens[nxt] == "{":
        return _closing_brace(tokens, nxt)
    return _parse_primary(tokens, nxt)


def _parse_command(tokens: list[str], index: int) -> int:
    index += 1
    if index < len(tokens) and tokens[index] == "{":
        return _closing_brace(tokens, index)
    return index


def _command(token: str) -> bool:
    return token.startswith("\\") and token not in _INFIX


def _atom(token: str) -> bool:
    first = token[0]
    return first.isdigit() or token.isalpha() or ord(first) > 127


def _starts(tokens: list[str], index: int) -> bool:
    token = tokens[index]
    if token in _INFIX or token in "+-=/)":
        return False
    return (
        token == "{"
        or token == "("
        or token.startswith("\\")
        or token[:1].isalnum()
        or ord(token[0]) > 127
    )


def _closing_brace(tokens: list[str], index: int) -> int:
    return _closing(tokens, index, "{", "}")


def _closing(tokens: list[str], index: int, opener: str, closer: str) -> int:
    if index >= len(tokens) or tokens[index] != opener:
        raise LatexError(f"expected {opener}")
    depth = 0
    for cursor in range(index, len(tokens)):
        if tokens[cursor] == opener:
            depth += 1
        elif tokens[cursor] == closer:
            depth -= 1
            if depth == 0:
                return cursor + 1
    raise LatexError(f"unclosed {opener}")
