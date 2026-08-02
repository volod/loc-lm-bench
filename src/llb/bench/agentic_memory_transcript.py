"""Memory-dependent long transcripts for the focused compact-versus-cap lane.

Each task exposes a final code in the first observation of a one-way workflow, followed by several
later observations. The workflow cursor makes progress externally observable and prevents a
repeated call from replaying the early fact, while a negative assertion prevents the model from
externalizing the final code. Completion therefore requires semantic state from the early
observation after the compact trigger.
"""

from typing import Any

from llb.bench.agentic.model import (
    ASSERT_ANSWER_CONTAINS,
    ASSERT_WORLD_NOT_CONTAINS,
    ASSERT_WORKFLOW_COMPLETE,
)
from llb.bench.tool_world import OBS_WORKFLOW_COMPLETE

DEFAULT_MEMORY_TASKS = 8
DEFAULT_MEMORY_DEPTH = 8
DEFAULT_MEMORY_PAD_CHARS = 1200
DEFAULT_MEMORY_MAX_STEPS = 14
DEFAULT_MEMORY_MIN_COMPACTION_RATE = 0.75
DEFAULT_CONTROL_TASKS = 4
DEFAULT_CONTROL_DEPTH = 10

_PAD_UNIT = "operational checkpoint notes for the municipal archive "


def _padded(first_line: str, pad_chars: int) -> str:
    if pad_chars <= 0:
        return first_line
    repeats = (pad_chars // len(_PAD_UNIT)) + 1
    return f"{first_line}\n{(_PAD_UNIT * repeats)[:pad_chars]}"


def memory_dependent_task(
    index: int,
    *,
    depth: int = DEFAULT_MEMORY_DEPTH,
    pad_chars: int = DEFAULT_MEMORY_PAD_CHARS,
) -> dict[str, Any]:
    """Build one task whose early one-way code must survive later workflow observations."""
    if depth < 3:
        raise ValueError(f"memory transcript depth must be >= 3, got {depth}")
    code = f"MEM-{index:03d}-{(index * 37 + 11) % 997:03d}"
    workflow_tokens = [f"wf-{index:03d}-{stage}" for stage in range(depth)]
    workflow = [_padded(f"[memory: final_code={code}]\nfinal code: {code}", pad_chars)]
    workflow.extend(
        _padded(f"checkpoint {stage + 1}/{depth}: accepted", pad_chars)
        for stage in range(depth - 2)
    )
    workflow.append(_padded(OBS_WORKFLOW_COMPLETE, pad_chars))
    return {
        "id": f"compact-memory-{index:03d}-d{depth}",
        "family": "compact-memory",
        "prompt": (
            f'Case {index:03d}. Почни з advance із токеном "{workflow_tokens[0]}". Кожен '
            "результат містить токен для НАСТУПНОГО виклику advance; завжди використовуй новий "
            "токен, доки процес не повідомить про завершення. Перший результат містить фінальний "
            "код і більше ніколи не повториться; запам'ятай його. Після завершення виклич finish "
            "і передай рівно цей код. Ніколи не зберігай фінальний код у файлі чи базі."
        ),
        "setup": {"workflow": workflow, "workflow_tokens": workflow_tokens},
        "success": [
            {"kind": ASSERT_WORKFLOW_COMPLETE},
            {"kind": ASSERT_ANSWER_CONTAINS, "value": code},
            {"kind": ASSERT_WORLD_NOT_CONTAINS, "value": code},
        ],
    }


def build_memory_dependent_tasks(
    *,
    n_tasks: int = DEFAULT_MEMORY_TASKS,
    depth: int = DEFAULT_MEMORY_DEPTH,
    pad_chars: int = DEFAULT_MEMORY_PAD_CHARS,
) -> list[dict[str, Any]]:
    """Build a deterministic task set with a stable shared digest."""
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    return [memory_dependent_task(i, depth=depth, pad_chars=pad_chars) for i in range(n_tasks)]


def token_chain_control_task(index: int, *, depth: int = DEFAULT_CONTROL_DEPTH) -> dict[str, Any]:
    """Build a memory-free control whose final observation supplies the answer.

    The task isolates basic token-chain following: unlike the transfer tasks, no fact must survive
    from an old observation. A model that cannot walk this chain is ineligible for a context-policy
    comparison because its failures cannot be attributed to compact versus observation-cap.
    """
    if depth < 3:
        raise ValueError(f"token-chain control depth must be >= 3, got {depth}")
    code = f"CTRL-{index:03d}-{(index * 53 + 17) % 997:03d}"
    workflow_tokens = [f"ctrl-{index:03d}-{stage}" for stage in range(depth)]
    workflow = [f"control checkpoint {stage + 1}/{depth}: accepted" for stage in range(depth - 1)]
    workflow.append(f"{OBS_WORKFLOW_COMPLETE}\nfinal control code: {code}")
    return {
        "id": f"token-chain-control-{index:03d}-d{depth}",
        "family": "token-chain-control",
        "prompt": (
            f'Case {index:03d}. Почни з advance із токеном "{workflow_tokens[0]}". Кожен '
            "результат містить токен для НАСТУПНОГО виклику advance; завжди використовуй новий "
            "токен, доки процес не повідомить про завершення. Лише фінальний результат містить "
            "контрольний код. Після завершення виклич finish і передай рівно цей код."
        ),
        "setup": {"workflow": workflow, "workflow_tokens": workflow_tokens},
        "success": [
            {"kind": ASSERT_WORKFLOW_COMPLETE},
            {"kind": ASSERT_ANSWER_CONTAINS, "value": code},
        ],
    }


def build_token_chain_control_tasks(
    *, n_tasks: int = DEFAULT_CONTROL_TASKS, depth: int = DEFAULT_CONTROL_DEPTH
) -> list[dict[str, Any]]:
    """Build the deterministic eligibility-pilot task set."""
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    return [token_chain_control_task(i, depth=depth) for i in range(n_tasks)]
