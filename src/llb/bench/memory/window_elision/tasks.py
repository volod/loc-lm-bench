"""Deterministic head, middle, and tail evidence tasks for summary-input elision."""

from typing import Any, cast

from llb.bench.agentic.context import TranscriptEntry
from llb.bench.agentic.context_summary import format_summary_transcript
from llb.bench.agentic.model import (
    ASSERT_ANSWER_CONTAINS,
    ASSERT_WORLD_NOT_CONTAINS,
    ASSERT_WORKFLOW_COMPLETE,
)
from llb.bench.tool_world import ADVANCE, OBS_WORKFLOW_COMPLETE

STRATUM_HEAD = "head"
STRATUM_MIDDLE = "middle"
STRATUM_TAIL = "tail"
STRATA = (STRATUM_HEAD, STRATUM_MIDDLE, STRATUM_TAIL)

_STRATUM_CODE = {STRATUM_HEAD: "H", STRATUM_MIDDLE: "M", STRATUM_TAIL: "T"}


class FactNotOffered(ValueError):
    """The fold offered a transcript that stops before the stage planting this task's fact.

    Distinct from every other placement failure: nothing moved under the trim boundaries, the fold
    simply happened too early for the task's stage, so a caller scanning guards can name that as
    its own reason instead of calling it a misplacement.
    """


_PAD_UNIT = "operational checkpoint notes for the municipal archive "
_SLOT_LINE_CHARS = 64


def window_elision_stratum_task(
    index: int,
    *,
    stratum: str,
    fact_stage: int,
    depth: int,
    pad_chars: int,
) -> dict[str, Any]:
    """Build one workflow whose only answer fact occupies a declared transcript stratum."""
    if stratum not in STRATA:
        raise ValueError(f"unknown evidence stratum: {stratum!r}")
    if depth < 4 or not 0 <= fact_stage < depth - 1:
        raise ValueError("the answer fact must be in a non-final stage of a depth >= 4 workflow")
    code = f"ELI-{_STRATUM_CODE[stratum]}-{index:03d}-{(index * 37 + 11) % 997:03d}"
    token_prefix = _STRATUM_CODE[stratum].lower()
    workflow_tokens = [f"el-{token_prefix}-{index:03d}-{stage:02d}" for stage in range(depth)]
    workflow = [
        _padded_slot(_slot_line(stage, code if stage == fact_stage else None), pad_chars)
        for stage in range(depth - 1)
    ]
    workflow.append(_padded_slot(OBS_WORKFLOW_COMPLETE, pad_chars))
    return {
        "id": f"window-elision-{token_prefix}-{index:03d}-d{depth}",
        "family": f"window-elision-{stratum}",
        "prompt": (
            f"Case {token_prefix.upper()}{index:03d}. Почни з advance із токеном "
            f'"{workflow_tokens[0]}". Кожен результат містить токен для НАСТУПНОГО виклику '
            "advance; завжди використовуй новий токен, доки процес не повідомить про завершення. "
            "Рівно один результат містить answer_code; запам'ятай його. Після завершення виклич "
            "finish і передай рівно цей код. Ніколи не зберігай код у файлі чи базі."
        ),
        "setup": {"workflow": workflow, "workflow_tokens": workflow_tokens},
        "success": [
            {"kind": ASSERT_WORKFLOW_COMPLETE},
            {"kind": ASSERT_ANSWER_CONTAINS, "value": code},
            {"kind": ASSERT_WORLD_NOT_CONTAINS, "value": code},
        ],
        "elision_stratum": stratum,
        "fact_stage": fact_stage,
        "answer_code": code,
    }


def build_window_elision_stratum_tasks(
    *,
    n_tasks_per_stratum: int,
    fact_stages: dict[str, int],
    depth: int,
    pad_chars: int,
) -> list[dict[str, Any]]:
    """Build equal-size task groups for every declared evidence stratum."""
    if n_tasks_per_stratum < 1 or set(fact_stages) != set(STRATA):
        raise ValueError("stratum tasks need a positive count and one fact stage per stratum")
    return [
        window_elision_stratum_task(
            index,
            stratum=stratum,
            fact_stage=fact_stages[stratum],
            depth=depth,
            pad_chars=pad_chars,
        )
        for stratum in STRATA
        for index in range(n_tasks_per_stratum)
    ]


def answer_fact_placement(
    record: dict[str, Any], *, offered_chars: int, transcript_cap_chars: int
) -> dict[str, object]:
    """Locate the answer fact against the exact head/middle/tail trim boundaries."""
    offered, n_entries = _offered_with_length(record, offered_chars)
    code = str(record["answer_code"])
    start = offered.find(code)
    if start < 0:
        raise FactNotOffered(
            f"task {record['id']!r} plants its fact at stage {record['fact_stage']}, which the "
            f"{offered_chars}-char fold does not reach"
        )
    end = start + len(code)
    head_end = int(transcript_cap_chars * 0.6)
    tail_start = len(offered) - (transcript_cap_chars - head_end)
    stratum = _classify_span(start, end, head_end=head_end, tail_start=tail_start)
    return {
        "declared_stratum": record["elision_stratum"],
        "measured_stratum": stratum,
        "fact_stage": record["fact_stage"],
        "folded_entries": n_entries,
        "fact_start": start,
        "fact_end": end,
        "head_end": head_end,
        "tail_start": tail_start,
        "offered_chars": len(offered),
        "transcript_cap_chars": transcript_cap_chars,
    }


def _slot_line(stage: int, code: str | None) -> str:
    payload = f"answer_code={code}" if code is not None else f"checkpoint={stage:02d}-accepted"
    return f"[slot={stage:02d} {payload}]".ljust(_SLOT_LINE_CHARS, ".")


def _padded_slot(first_line: str, pad_chars: int) -> str:
    repeats = (pad_chars // len(_PAD_UNIT)) + 1
    return f"{first_line}\n{(_PAD_UNIT * repeats)[:pad_chars]}"


def _offered_with_length(record: dict[str, Any], offered_chars: int) -> tuple[str, int]:
    entries = _workflow_entries(record)
    matches = [
        (format_summary_transcript(entries[:count]), count)
        for count in range(1, len(entries) + 1)
        if len(format_summary_transcript(entries[:count])) == offered_chars
    ]
    if len(matches) != 1:
        raise ValueError(
            f"task {record['id']!r} has {len(matches)} transcript prefixes of {offered_chars} chars"
        )
    return matches[0]


def _workflow_entries(record: dict[str, Any]) -> list[TranscriptEntry]:
    setup = cast(dict[str, object], record["setup"])
    workflow = cast(list[str], setup["workflow"])
    tokens = cast(list[str], setup["workflow_tokens"])
    entries: list[TranscriptEntry] = []
    for index, observation in enumerate(workflow):
        if index + 1 < len(tokens):
            observation = f"{observation}\n[next token: {tokens[index + 1]}]"
        entries.append((ADVANCE, {"token": tokens[index]}, observation))
    return entries


def _classify_span(start: int, end: int, *, head_end: int, tail_start: int) -> str:
    if end <= head_end:
        return STRATUM_HEAD
    if start >= tail_start:
        return STRATUM_TAIL
    if start >= head_end and end <= tail_start:
        return STRATUM_MIDDLE
    return "boundary"
