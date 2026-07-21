"""Small helpers for turning ledger dictionaries into review sections."""

import json
from collections.abc import Iterable, Mapping

from llb.review.core import ReviewSection, SectionRole


def fields_section(
    title: str,
    row: Mapping[str, object],
    fields: Iterable[str],
    role: SectionRole,
) -> ReviewSection:
    lines = [f"{field}: {_text(row.get(field))}" for field in fields if _text(row.get(field))]
    return ReviewSection(title, "\n\n".join(lines) or "(none)", role)


def json_section(title: str, value: object, role: SectionRole) -> ReviewSection:
    return ReviewSection(title, json.dumps(value, ensure_ascii=False, indent=2), role)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)
