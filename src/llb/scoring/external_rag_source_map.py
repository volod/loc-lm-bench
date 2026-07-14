"""Focused external rag source map implementation."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KEY_ARTICLE_ID = "article_id"

KEY_URL = "url"

KEY_TITLE = "article_title"

SOURCE_RECALL_K = 3

SOURCE_AUDIT_COLUMNS = [
    "source_hit",
    "source_first_hit_rank",
    "source_hit_weak",
    "source_mapped_count",
    "source_unmapped_count",
]


@dataclass(frozen=True)
class SourceMapEntry:
    """One provider-source -> corpus location mapping."""

    doc_id: str
    char_start: int | None = None
    char_end: int | None = None

    @property
    def has_span(self) -> bool:
        return self.char_start is not None and self.char_end is not None


@dataclass(frozen=True)
class SourceMap:
    """Mapping indexes by provider key kind (see `KEY_*` precedence)."""

    by_article_id: dict[str, SourceMapEntry]
    by_url: dict[str, SourceMapEntry]
    by_title: dict[str, SourceMapEntry]

    def __len__(self) -> int:
        return len(self.by_article_id) + len(self.by_url) + len(self.by_title)


def load_source_map(path: Path | str) -> SourceMap:
    """Load a mapping sidecar (.json list, .jsonl, or .csv) into keyed indexes.

    Each record needs `doc_id` plus at least one provider key (`article_id`, `url`,
    `article_title`); `char_start`/`char_end` are optional and enable span-proof hits.
    """
    path = Path(path)
    rows = _read_mapping_rows(path)
    by_id: dict[str, SourceMapEntry] = {}
    by_url: dict[str, SourceMapEntry] = {}
    by_title: dict[str, SourceMapEntry] = {}
    for index, row in enumerate(rows, 1):
        doc_id = str(row.get("doc_id") or "").strip()
        if not doc_id:
            raise ValueError(f"{path}: mapping record {index} lacks doc_id")
        entry = SourceMapEntry(
            doc_id=doc_id,
            char_start=_int_or_none(row.get("char_start")),
            char_end=_int_or_none(row.get("char_end")),
        )
        keyed = False
        for key_field, index_map in (
            (KEY_ARTICLE_ID, by_id),
            (KEY_URL, by_url),
            (KEY_TITLE, by_title),
        ):
            key = str(row.get(key_field) or "").strip()
            if key:
                index_map[key] = entry
                keyed = True
        if not keyed:
            raise ValueError(
                f"{path}: mapping record {index} has no provider key "
                f"({KEY_ARTICLE_ID} / {KEY_URL} / {KEY_TITLE})"
            )
    return SourceMap(by_article_id=by_id, by_url=by_url, by_title=by_title)


def _read_mapping_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl_mapping_rows(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("mappings", [])
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list of mapping records")
    return [row for row in payload if isinstance(row, dict)]


def _read_jsonl_mapping_rows(path: Path) -> list[dict[str, Any]]:
    """One mapping dict per non-blank JSONL line (a non-object line is a hard error)."""
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected a JSON object")
        rows.append(row)
    return rows


def map_source(source: dict[str, Any], source_map: SourceMap) -> SourceMapEntry | None:
    """Resolve one returned source record through the map (id > url > title precedence)."""
    article_id = str(source.get("article_id") or source.get("id") or "").strip()
    if article_id and article_id in source_map.by_article_id:
        return source_map.by_article_id[article_id]
    url = str(source.get("url") or source.get("uri") or "").strip()
    if url and url in source_map.by_url:
        return source_map.by_url[url]
    title = str(source.get("article_title") or source.get("title") or source.get("name") or "")
    title = title.strip()
    if title and title in source_map.by_title:
        return source_map.by_title[title]
    return None


def _int_or_none(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(str(value))
