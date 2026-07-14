"""Focused summarization io implementation."""

import json
from pathlib import Path
from llb.bench.summarization_scoring import (
    SummarizationCase,
)


def load_cases_file(path: Path | str) -> list[SummarizationCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array of summarization cases")
    return [SummarizationCase.from_record(r) for r in raw]
