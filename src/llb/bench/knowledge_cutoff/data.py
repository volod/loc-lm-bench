"""Event schema and reproducible local/Hugging Face dataset loading."""

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

DEFAULT_DATASET_ID = "apoorvumang/knowledge-cutoff-benchmark"
DEFAULT_DATASET_CONFIG = "events"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_DATASET_REVISION = "main"
DATASET_LICENSE = "CC BY 4.0"
UPSTREAM_PROJECT = "https://github.com/apoorvumang/knowledge-cutoff"

REAL_CATEGORIES = frozenset({"death", "office_change"})
CONTROL_CATEGORIES = frozenset({"control_alive", "fake_event"})
VALID_CATEGORIES = REAL_CATEGORIES | CONTROL_CATEGORIES
VALID_PREDICTABILITY = frozenset({"low", "medium", "high"})
LETTERS = ("A", "B", "C", "D")
COMMIT_REVISION = re.compile(r"^[0-9a-f]{40}$", re.I)


class CutoffEvent(BaseModel):
    """One dated knowledge probe from the public benchmark dataset."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    id: str
    date: str
    month: str
    category: str
    region: str
    predictability: str
    subject: str
    fact: str
    mcq_question: str
    mcq_choices: list[str]
    mcq_answer: str
    source: str = ""

    @field_validator("date", mode="before")
    @classmethod
    def _normalize_date(cls, value: object) -> object:
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return value

    @field_validator("category")
    @classmethod
    def _category_known(cls, value: str) -> str:
        if value not in VALID_CATEGORIES:
            raise ValueError(f"unsupported category {value!r}")
        return value

    @field_validator("predictability")
    @classmethod
    def _predictability_known(cls, value: str) -> str:
        if value not in VALID_PREDICTABILITY:
            raise ValueError(f"unsupported predictability {value!r}")
        return value

    @field_validator("mcq_answer")
    @classmethod
    def _answer_letter(cls, value: str) -> str:
        answer = value.upper()
        if answer not in LETTERS:
            raise ValueError("mcq_answer must be A, B, C, or D")
        return answer

    @model_validator(mode="after")
    def _dates_and_choices(self) -> "CutoffEvent":
        try:
            parsed = date.fromisoformat(self.date)
        except ValueError as exc:
            raise ValueError("date must use YYYY-MM-DD") from exc
        if self.month != parsed.strftime("%Y-%m"):
            raise ValueError("month must match date")
        if len(self.mcq_choices) != len(LETTERS):
            raise ValueError("mcq_choices must contain exactly four choices")
        if len(set(self.mcq_choices)) != len(self.mcq_choices):
            raise ValueError("mcq_choices must be unique")
        return self

    @property
    def counts_for_curve(self) -> bool:
        return self.category in REAL_CATEGORIES and self.predictability in {"low", "medium"}


@dataclass(frozen=True, slots=True)
class EventSource:
    kind: str
    identity: str
    requested_revision: str | None
    resolved_revision: str
    config: str | None
    split: str | None
    license: str


@dataclass(frozen=True, slots=True)
class LoadedEvents:
    events: list[CutoffEvent]
    source: EventSource


DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]
RevisionResolver = Callable[[str, str], str]


def _parse_records(records: Iterable[Mapping[str, Any]], identity: str) -> list[CutoffEvent]:
    events: list[CutoffEvent] = []
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        try:
            event = CutoffEvent.model_validate(dict(record))
        except ValueError as exc:
            raise ValueError(f"{identity}: invalid event {index}: {exc}") from exc
        if event.id in seen:
            raise ValueError(f"{identity}: duplicate event id {event.id!r}")
        seen.add(event.id)
        events.append(event)
    if not events:
        raise ValueError(f"{identity}: no events loaded")
    return events


def _local_events(path: Path) -> LoadedEvents:
    raw = path.read_bytes()
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}: line {line_number} must be a JSON object")
        records.append(record)
    digest = hashlib.sha256(raw).hexdigest()
    return LoadedEvents(
        events=_parse_records(records, str(path)),
        source=EventSource(
            "local", str(path), None, f"sha256:{digest}", None, None, "operator-provided"
        ),
    )


def _default_revision_resolver(dataset_id: str, revision: str) -> str:
    from huggingface_hub import HfApi

    return str(HfApi().dataset_info(dataset_id, revision=revision).sha)


def _default_dataset_loader(*args: Any, **kwargs: Any) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face loading needs the cutoff extra; run `make venv` or install `.[cutoff]`"
        ) from exc
    return cast(Iterable[Mapping[str, Any]], load_dataset(*args, **kwargs))


def load_events(
    *,
    path: Path | str | None = None,
    dataset_id: str = DEFAULT_DATASET_ID,
    revision: str = DEFAULT_DATASET_REVISION,
    cache_dir: Path | str | None = None,
    dataset_loader: DatasetLoader | None = None,
    revision_resolver: RevisionResolver | None = None,
) -> LoadedEvents:
    """Load local JSONL or pin a Hugging Face dataset revision before reading it."""
    if path is not None:
        return _local_events(Path(path))
    resolver = revision_resolver or _default_revision_resolver
    resolved = revision if COMMIT_REVISION.fullmatch(revision) else resolver(dataset_id, revision)
    loader = dataset_loader or _default_dataset_loader
    records = loader(
        dataset_id,
        DEFAULT_DATASET_CONFIG,
        split=DEFAULT_DATASET_SPLIT,
        revision=resolved,
        **({"cache_dir": str(cache_dir)} if cache_dir is not None else {}),
    )
    return LoadedEvents(
        events=_parse_records(records, dataset_id),
        source=EventSource(
            "huggingface",
            dataset_id,
            revision,
            resolved,
            DEFAULT_DATASET_CONFIG,
            DEFAULT_DATASET_SPLIT,
            DATASET_LICENSE,
        ),
    )


def select_events(events: list[CutoffEvent], limit: int | None) -> list[CutoffEvent]:
    """Deterministically spread a smoke limit across the complete time horizon."""
    if limit is None or limit >= len(events):
        return list(events)
    if limit < 1:
        raise ValueError("limit must be at least 1")
    ordered = sorted(events, key=lambda event: (event.month, event.id))
    return [
        ordered[min(int(index * len(ordered) / limit), len(ordered) - 1)] for index in range(limit)
    ]
