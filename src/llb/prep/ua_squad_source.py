"""Pinned source and deterministic selection for the reviewed UA-SQuAD fixture."""

from collections.abc import Iterable, Iterator
from itertools import islice

from llb.core.contracts.rag import SquadRecord

DATASET_ID = "FIdo-AI/ua-squad"
DATASET_REVISION = "943ef27daea65e400350ef1875d07c7e97288177"
DATASET_SPLIT = "validation"
SOURCE_FILE = "val.json"
SOURCE_SHA256 = "5ff2384d103eb7d4ccb317c8071bd7b2a5a23b221188d8cd0514a85d305745b4"
DEFAULT_ITEMS = 250


def has_grounded_answer(record: SquadRecord) -> bool:
    """Whether the first answer is a non-empty exact substring of the context."""
    answers = record.get("answers") or {}
    texts = answers.get("text") or []
    return bool(texts and texts[0] and texts[0] in record["context"])


def iter_context_diverse(records: Iterable[SquadRecord]) -> Iterator[SquadRecord]:
    """Yield grounded records in source order, with at most one QA per context."""
    contexts: set[str] = set()
    for record in records:
        context = record["context"]
        if context in contexts or not has_grounded_answer(record):
            continue
        contexts.add(context)
        yield record


def select_context_diverse(records: Iterable[SquadRecord], max_items: int) -> list[SquadRecord]:
    """Return exactly `max_items` records from the fixture's deterministic selection."""
    selected = list(islice(iter_context_diverse(records), max_items))
    if len(selected) < max_items:
        raise ValueError(f"source contains only {len(selected)} eligible distinct contexts")
    return selected
