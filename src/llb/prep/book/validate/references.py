"""The cross-reference counts render wrote, read back for the vault report."""

import json
from pathlib import Path
from typing import TypedDict

from llb.prep.book.constants import METADATA_DIRNAME
from llb.prep.book.render.constants import REFERENCE_REPORT_FILENAME


class ReferenceReport(TypedDict):
    """The counts render recorded. Keys are citation kinds, not sentences."""

    ambiguous: dict[str, int]
    ambiguous_keys: list[str]
    gaps: int
    grammar: str
    resolved: dict[str, int]
    unresolved: dict[str, int]


_MISSING = "reference report was not written"
_UNREADABLE = "reference report could not be read"


def reference_section(book_dir: Path) -> list[str]:
    """Markdown lines for the processing report. Empty of recovered sentences."""
    payload, problem = load_reference_report(book_dir)
    lines = ["## Cross-references", ""]
    if problem or payload is None:
        lines.append(_MISSING if problem == _MISSING else _UNREADABLE)
        lines.append("")
        return lines
    lines.append(f"Grammar: `{payload['grammar']}`.")
    lines.append(_count("Resolved", payload["resolved"]))
    lines.append(_count("Ambiguous", payload["ambiguous"]))
    lines.append(_count("Unresolved", payload["unresolved"]))
    gaps = payload["gaps"]
    lines.append(f"Mapping gaps left unlinked: {gaps}.")
    keys = payload["ambiguous_keys"]
    if keys:
        shown = ", ".join(f"`{key}`" for key in keys)
        lines.append(f"Ambiguous keys: {shown}.")
    lines.append("")
    return lines


def reference_warnings(book_dir: Path) -> list[str]:
    """A missing report, or gap citations, as warnings. Neither fails the check."""
    payload, problem = load_reference_report(book_dir)
    if problem:
        return [problem]
    if payload is None or not payload["gaps"]:
        return []
    return [f"{payload['gaps']} printed-page reference(s) fell in a mapping gap"]


def reference_payload(book_dir: Path) -> ReferenceReport | None:
    """The report object for ``validation.json``, or None when it cannot be read."""
    payload, problem = load_reference_report(book_dir)
    if problem or payload is None:
        return None
    return payload


def load_reference_report(book_dir: Path) -> tuple[ReferenceReport | None, str]:
    """The parsed report and a problem string. A clean read has an empty problem."""
    path = book_dir / METADATA_DIRNAME / REFERENCE_REPORT_FILENAME
    if not path.is_file():
        return None, _MISSING
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, _UNREADABLE
    if not isinstance(loaded, dict):
        return None, _UNREADABLE
    checked = _checked(loaded)
    if checked is None:
        return None, _UNREADABLE
    return checked, ""


def _checked(loaded: dict[str, object]) -> ReferenceReport | None:
    grammar = loaded.get("grammar")
    gaps = loaded.get("gaps")
    if not isinstance(grammar, str) or not isinstance(gaps, int):
        return None
    return {
        "ambiguous": _counts(loaded.get("ambiguous")),
        "ambiguous_keys": _keys(loaded.get("ambiguous_keys")),
        "gaps": gaps,
        "grammar": grammar,
        "resolved": _counts(loaded.get("resolved")),
        "unresolved": _counts(loaded.get("unresolved")),
    }


def _counts(raw: object) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    found: dict[str, int] = {}
    for key in sorted(raw):
        value = raw[key]
        if isinstance(key, str) and isinstance(value, int):
            found[key] = value
    return found


def _keys(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str)]


def _count(label: str, counts: object) -> str:
    if not isinstance(counts, dict) or not counts:
        return f"{label}: 0."
    shown = ", ".join(f"{name} {counts[name]}" for name in sorted(counts))
    return f"{label}: {shown}."
