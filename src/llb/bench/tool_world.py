"""agentic benchmark deterministic in-memory tool-world -- the agentic sandbox (pure).

A small DETERMINISTIC environment the agentic loop's tools EXECUTE against (no tau-bench /
AgentBench): a mock filesystem, a mock key-value DB, substring search over a small UA corpus, and
a safe calculator. Every tool is a pure function of (env, args) -> observation, mutating only the
in-memory `ToolWorld`, so a task's success is checkable from the final env-state and the run is
fully reproducible + unit-testable without any model or network.
"""

import ast
import operator
from dataclasses import dataclass, field
from typing import Any, Callable

from llb.core.contracts.benchmarks import ToolDef

# Tool names.
READ_FILE = "read_file"
WRITE_FILE = "write_file"
DB_GET = "db_get"
DB_SET = "db_set"
SEARCH = "search"
CALCULATOR = "calculator"
FINISH = "finish"  # ends the episode with a final answer (not executed against the world)

# Observation strings (UA, ASCII-safe markers where relevant).
OBS_FILE_NOT_FOUND = "(файл не знайдено)"
OBS_DB_MISSING = "(ключ відсутній)"
OBS_NO_RESULTS = "(нічого не знайдено)"
OBS_OK = "ok"
OBS_BAD_ARGS = "(некоректні аргументи)"
OBS_CALC_ERROR = "(помилка обчислення)"
OBS_UNKNOWN_TOOL = "(невідомий інструмент)"

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str) -> str:
    """Evaluate an arithmetic expression with a restricted AST (numbers + + - * / // % ** and
    parentheses only; no names, calls, or attribute access). Returns the result or an error
    marker -- never executes arbitrary code."""

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    try:
        value = _eval(ast.parse(expression, mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError, OverflowError):
        return OBS_CALC_ERROR
    # Render whole floats as ints ("84.0" -> "84") for stable env-state assertions.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


@dataclass
class ToolWorld:
    """The mutable in-memory environment the agent's tools act on."""

    files: dict[str, str] = field(default_factory=dict)
    db: dict[str, str] = field(default_factory=dict)
    corpus: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_setup(cls, setup: dict[str, Any]) -> "ToolWorld":
        return cls(
            files=dict(setup.get("files", {}) or {}),
            db=dict(setup.get("db", {}) or {}),
            corpus=dict(setup.get("corpus", {}) or {}),
        )

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one tool against the world and return its observation text."""
        handler = _HANDLERS.get(name)
        if handler is None:
            return OBS_UNKNOWN_TOOL
        return handler(self, arguments)


def _arg(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    return None if value is None else str(value)


def _read_file(world: ToolWorld, arguments: dict[str, Any]) -> str:
    path = _arg(arguments, "path")
    if path is None:
        return OBS_BAD_ARGS
    return world.files.get(path, OBS_FILE_NOT_FOUND)


def _write_file(world: ToolWorld, arguments: dict[str, Any]) -> str:
    path = _arg(arguments, "path")
    content = arguments.get("content")
    if path is None or content is None:
        return OBS_BAD_ARGS
    world.files[path] = str(content)
    return OBS_OK


def _db_get(world: ToolWorld, arguments: dict[str, Any]) -> str:
    key = _arg(arguments, "key")
    if key is None:
        return OBS_BAD_ARGS
    return world.db.get(key, OBS_DB_MISSING)


def _db_set(world: ToolWorld, arguments: dict[str, Any]) -> str:
    key = _arg(arguments, "key")
    value = arguments.get("value")
    if key is None or value is None:
        return OBS_BAD_ARGS
    world.db[key] = str(value)
    return OBS_OK


def _search(world: ToolWorld, arguments: dict[str, Any]) -> str:
    query = _arg(arguments, "query")
    if not query:
        return OBS_BAD_ARGS
    low = query.casefold()
    hits = [f"[{doc_id}] {text}" for doc_id, text in world.corpus.items() if low in text.casefold()]
    return "\n".join(hits) if hits else OBS_NO_RESULTS


def _calculator(world: ToolWorld, arguments: dict[str, Any]) -> str:
    expression = _arg(arguments, "expression")
    return safe_eval(expression) if expression else OBS_BAD_ARGS


_HANDLERS: dict[str, Callable[[ToolWorld, dict[str, Any]], str]] = {
    READ_FILE: _read_file,
    WRITE_FILE: _write_file,
    DB_GET: _db_get,
    DB_SET: _db_set,
    SEARCH: _search,
    CALCULATOR: _calculator,
}


def tool_catalog() -> dict[str, ToolDef]:
    """The OpenAI-style schema for every sandbox tool + `finish` (for the agent prompt)."""
    return {
        READ_FILE: _tool(READ_FILE, "Прочитати вміст файлу.", {"path": "string"}, ["path"]),
        WRITE_FILE: _tool(
            WRITE_FILE,
            "Записати вміст у файл.",
            {"path": "string", "content": "string"},
            ["path", "content"],
        ),
        DB_GET: _tool(DB_GET, "Отримати значення з бази за ключем.", {"key": "string"}, ["key"]),
        DB_SET: _tool(
            DB_SET,
            "Зберегти значення в базу за ключем.",
            {"key": "string", "value": "string"},
            ["key", "value"],
        ),
        SEARCH: _tool(SEARCH, "Пошук у корпусі за підрядком.", {"query": "string"}, ["query"]),
        CALCULATOR: _tool(
            CALCULATOR, "Обчислити арифметичний вираз.", {"expression": "string"}, ["expression"]
        ),
        FINISH: _tool(
            FINISH, "Завершити завдання й повернути відповідь.", {"answer": "string"}, ["answer"]
        ),
    }


def _tool(name: str, description: str, props: dict[str, str], required: list[str]) -> ToolDef:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {p: {"type": t} for p, t in props.items()},
            "required": required,
        },
    }
