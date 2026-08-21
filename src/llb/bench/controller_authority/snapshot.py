"""Snapshot-isolation proof for controller-channel authority comparisons."""

import hashlib
import json
from typing import cast

from llb.bench.agentic.controller_channel import (
    CHANNEL_CONTROLLER,
    CHANNEL_OBSERVATION,
    CHANNEL_PREAMBLE,
)
from llb.bench.controller_authority.design import PREAMBLE_STUDY_KIND
from llb.bench.controller_authority.model import ChannelCell
from llb.core.contracts.common import ChatMessage


def _serialization_name(backend: str) -> str:
    return "ollama" if backend == "ollama" else "openai_compatible"


def _preamble_normalized(
    baseline: list[ChatMessage],
    candidate: list[ChatMessage],
    task_id: str,
    design: dict[str, object],
    transforms: dict[str, list[dict[str, str]]],
) -> object:
    if len(baseline) != 2 or len(candidate) != 2:
        raise ValueError(f"preamble snapshot cardinality is invalid for task {task_id}")
    if baseline[1]["content"] != design["authority_text"]:
        raise ValueError(f"authority snapshot text is invalid for task {task_id}")
    if candidate[0]["content"] != baseline[1]["content"]:
        raise ValueError(f"authority snapshot content changed for task {task_id}")
    if candidate[1]["content"] != baseline[0]["content"]:
        raise ValueError(f"task snapshot content changed for task {task_id}")
    expected_baseline = [
        {"role": step["role"], "content": baseline[index]["content"]}
        for index, step in enumerate(transforms[CHANNEL_OBSERVATION])
    ]
    expected_candidate = [
        {"role": step["role"], "content": candidate[index]["content"]}
        for index, step in enumerate(transforms[CHANNEL_PREAMBLE])
    ]
    if baseline != expected_baseline or candidate != expected_candidate:
        raise ValueError(f"preamble snapshot structure is invalid for task {task_id}")
    return {"prompt": baseline[0]["content"], "authority": baseline[1]["content"]}


def _role_normalized(
    baseline: list[ChatMessage],
    candidate: list[ChatMessage],
    task_id: str,
    design: dict[str, object],
    roles: dict[str, str],
) -> object:
    if [item["content"] for item in baseline] != [item["content"] for item in candidate]:
        raise ValueError(f"authority snapshot content changed for task {task_id}")
    if baseline[-1]["content"] != design["authority_text"]:
        raise ValueError(f"authority snapshot text is invalid for task {task_id}")
    if baseline[:-1] != candidate[:-1] or baseline[-1]["role"] != roles[CHANNEL_OBSERVATION]:
        raise ValueError(f"observation snapshot structure is invalid for task {task_id}")
    if candidate[-1]["role"] != roles[CHANNEL_CONTROLLER]:
        raise ValueError(f"controller snapshot structure is invalid for task {task_id}")
    return [{**item, "role": "authority"} for item in baseline]


def _content_digest(normalized: object) -> str:
    return hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def snapshot_proof(
    observation: ChannelCell,
    controller: ChannelCell,
    design: dict[str, object],
    backend: str,
) -> dict[str, object]:
    """Prove that placement is the only difference in every paired snapshot."""
    task_ids = sorted(observation.snapshots)
    if task_ids != sorted(controller.snapshots):
        raise ValueError("authority activation snapshots differ between placements")
    serialization = _serialization_name(backend)
    is_preamble = design["study_kind"] == PREAMBLE_STUDY_KIND
    pairs = []
    for task_id in task_ids:
        baseline = observation.snapshots[task_id]
        candidate = controller.snapshots[task_id]
        if is_preamble:
            transforms = cast(
                dict[str, dict[str, list[dict[str, str]]]], design["serializer_transforms"]
            )[serialization]
            normalized = _preamble_normalized(baseline, candidate, task_id, design, transforms)
        else:
            roles = cast(dict[str, dict[str, str]], design["role_serialization"])[serialization]
            normalized = _role_normalized(baseline, candidate, task_id, design, roles)
        pairs.append({"task_id": task_id, "content_digest": _content_digest(normalized)})
    return {"passed": True, "paired_tasks": len(pairs), "pairs": pairs}
