"""Focused input implementation."""

import json
import re
from pathlib import Path
from typing import Any
from llb.prep.frontier import parse_json_block
from llb.prep.ontology.constants import NEAR_DUP_COSINE_THRESHOLD

DEFAULT_DEDUP_THRESHOLD = NEAR_DUP_COSINE_THRESHOLD

MIN_QUESTION_CHARS = 12

MIN_QUESTION_WORDS = 3

DEFAULT_MIN_CONTEXT_CHARS = 80

MAX_ANSWER_CHARS = 400

MAX_ANSWER_CONTEXT_FRACTION = 0.6

_STRUCTURE_REFERENCE_STEMS = (
    "у наведеному тексті",
    "в наведеному тексті",
    "у цьому документі",
    "в цьому документі",
    "у цьому тексті",
    "в цьому тексті",
    "у документі",
    "в документі",
    "в уривку",
    "в уривці",
    "у фрагменті",
    "згідно з текстом",
    "згідно з наведеним",
    "у параграфі",
    "в абзаці",
    "according to the text",
    "according to the passage",
    "according to the document",
    "in this document",
    "in this passage",
    "in this excerpt",
)


def normalize_text(text: str) -> str:
    """Casefold + collapse whitespace: the key used for exact-duplicate detection."""
    return " ".join(text.split()).casefold()


def references_document_structure(question: str) -> bool:
    """True when the question points at the document/passage instead of asking naturally."""
    q = normalize_text(question)
    return any(stem in q for stem in _STRUCTURE_REFERENCE_STEMS)


def question_too_vague(question: str) -> bool:
    """True for questions too short to identify a needle."""
    q = question.strip()
    return len(q) < MIN_QUESTION_CHARS or len(q.split()) < MIN_QUESTION_WORDS


def load_json_documents(path: Path) -> list[Any]:
    """Load every JSON value in a file: raw JSON, one or more ``` fenced blocks, or JSONL.

    Service replies are exported by hand; a file may hold one clean JSON document, several fenced
    code blocks (one per batch), or JSON Lines. Returns the parsed values in file order and raises
    on a file with no parseable JSON at all (silent emptiness would hide an export mistake).
    """
    text = path.read_text(encoding="utf-8")
    fenced = re.findall(r"```(?:json[l5]?|jsonl)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return [_parse_lenient(block, source=f"{path}#fence") for block in fenced]
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"{path}: empty artifact file")
    try:
        return [json.loads(stripped)]
    except json.JSONDecodeError:
        pass
    lines = [line for line in stripped.splitlines() if line.strip()]
    parsed: list[Any] = []
    for line_no, line in enumerate(lines, 1):
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            parsed = []
            break
    if parsed:
        return parsed
    return [_parse_lenient(stripped, source=str(path))]


def _parse_lenient(block: str, *, source: str) -> Any:
    try:
        return parse_json_block(block)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: unparseable JSON ({exc})") from exc


def load_jsonl_rows(values: list[Any]) -> list[Any]:
    """Flatten loaded JSON values into rows: arrays are splatted, objects pass through."""
    rows: list[Any] = []
    for value in values:
        if isinstance(value, list):
            rows.extend(value)
        else:
            rows.append(value)
    return rows


def load_corpus_texts(corpus_root: Path) -> dict[str, str]:
    """Read every .md/.txt under `corpus_root` keyed by its relative path (the doc id)."""
    texts: dict[str, str] = {}
    for path in sorted(corpus_root.rglob("*")):
        if path.suffix.lower() in (".md", ".txt") and path.is_file():
            texts[str(path.relative_to(corpus_root))] = path.read_text(encoding="utf-8")
    if not texts:
        raise SystemExit(f"[curate] no .md/.txt corpus documents under {corpus_root}")
    return texts
