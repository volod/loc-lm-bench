"""Focused store io implementation."""

import json
from pathlib import Path
from typing import cast
from llb.core.contracts.rag import ChunkRecord


def _renumber(hits: list[ChunkRecord]) -> list[ChunkRecord]:
    """Reassign contiguous 1-based ranks after a filter removed candidates."""
    for rank, hit in enumerate(hits, 1):
        hit["rank"] = rank
    return hits


def _write_jsonl(rows: list[ChunkRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[ChunkRecord]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return cast(list[ChunkRecord], rows)
