"""Dispatch an engine's files or text onto proposed blocks."""

import json
from pathlib import Path

from llb.prep.book.engines.parse_docling import parse_docling_dir
from llb.prep.book.engines.parse_text import parse_dots, parse_grounding, parse_markdown
from llb.prep.book.engines.protocol import Proposed

RAW_BLOCKS = "raw_blocks.json"
_TEXT_PARSERS = {
    "dots": parse_dots,
    "deepseek": parse_grounding,
    "olmocr": parse_markdown,
    "markdown": parse_markdown,
}


def parse_text(parser: str, text: str, width: int, height: int) -> list[Proposed]:
    """A served or Ollama reading."""
    return _TEXT_PARSERS[parser](text, width, height)


def parse_directory(parser: str, directory: Path, width: int, height: int) -> list[Proposed]:
    """Files an in-process engine wrote under its work directory."""
    if parser == "raw-json":
        return _raw(directory / RAW_BLOCKS)
    if parser == "docling-json":
        return parse_docling_dir(directory, width, height)
    return _mineru_or_markdown(directory, width, height)


def output_present(parser: str, directory: Path) -> bool:
    """Whether the engine wrote the file its parser reads. An empty file still counts."""
    if parser == "raw-json":
        return (directory / RAW_BLOCKS).is_file()
    if parser == "docling-json":
        return any(path.is_file() for path in directory.glob("**/*.json"))
    return any(
        path.is_file() and path.suffix in {".md", ".json"} for path in directory.glob("**/*")
    )


def _raw(path: Path) -> list[Proposed]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("blocks") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    proposed = []
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        proposed.append(
            Proposed(
                str(item.get("kind") or "unknown"),
                int(item.get("order", index)),
                float(item["x0"]),
                float(item["y0"]),
                float(item["x1"]),
                float(item["y1"]),
                str(item.get("text") or ""),
                _confidence(item.get("confidence")),
            )
        )
    return proposed


def _confidence(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _mineru_or_markdown(directory: Path, width: int, height: int) -> list[Proposed]:
    for path in sorted(directory.glob("**/*content_list*.json")):
        proposed = _content_list(path)
        if proposed:
            return proposed
    texts = [
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(directory.glob("**/*.md"))
    ]
    if not texts:
        return []
    return parse_markdown("\n\n".join(texts), width, height)


def _content_list(path: Path) -> list[Proposed]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    proposed = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        box = item.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            continue
        proposed.append(
            Proposed(
                str(item.get("type") or "unknown"),
                index,
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
                str(item.get("text") or ""),
                None,
            )
        )
    return proposed
