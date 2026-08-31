"""Render one benchmark task into the backend-neutral typed decision protocol."""

import json

from llb.core.contracts.common import ChatMessage
from llb.core.contracts.robotics import DeviceSnapshot
from llb.prompts.registry import render_chat
from llb.robotics.benchmark.models import BenchmarkTask
from llb.robotics.emulator_models import ActionPolicy


def benchmark_messages(
    task: BenchmarkTask,
    snapshot: DeviceSnapshot,
    policy: ActionPolicy,
    contexts: list[dict[str, object]],
) -> list[ChatMessage]:
    evidence = (
        json.dumps(contexts, ensure_ascii=True, indent=2)
        if contexts
        else "[] (retrieval withheld for this paired lane)"
    )
    values = {
        "goal": task.goal,
        "snapshot": snapshot.model_dump_json(indent=2),
        "policy": policy.model_dump_json(indent=2),
        "runtime_notice": task.setup.runtime_notice or "none",
        "evidence": evidence,
    }
    return render_chat("robotics.benchmark.decision", values)
