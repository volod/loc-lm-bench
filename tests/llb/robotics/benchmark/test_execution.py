import json

from llb.backends.base import ChatResult
from llb.core.contracts.common import ChatMessage
from llb.core.paths import PROJECT_ROOT
from llb.robotics.benchmark.constants import LANE_REFERENCE, LANE_RETRIEVAL
from llb.robotics.benchmark.design import load_design
from llb.robotics.benchmark.execution import evaluate_task
from llb.robotics.benchmark.metrics import aggregate, paired_verdict
from llb.robotics.emulator_fixture import load_emulator_fixture

DESIGN = PROJECT_ROOT / "samples" / "robotics" / "benchmark" / "design.json"
EMULATOR = PROJECT_ROOT / "samples" / "robotics" / "emulator" / "scenarios.json"


def _contexts(task):
    return [
        {"evidence_id": evidence_id, "doc_id": "fixture", "rank": 1, "text": "trusted note"}
        for evidence_id in task.expected_evidence_ids
    ]


def _proposal_payload(task, *, injection=False):
    arguments = [item.model_dump(mode="json") for item in task.arguments]
    evidence_ids = list(task.expected_evidence_ids)
    if injection:
        arguments = [{"name": "position_mm", "value": 999.0}]
        evidence_ids = ["evidence:retrieved-injection"]
    return {
        "decision": "propose",
        "reason": "fixture proposal",
        "proposal": {
            "device_id": task.device_id,
            "operation": task.operation,
            "arguments": arguments,
            "evidence_ids": evidence_ids,
            "preconditions": list(task.preconditions),
            "postconditions": list(task.postconditions),
            "risk_class": task.risk_class,
            "idempotency": task.idempotency,
        },
    }


def test_reference_controller_completes_or_refuses_every_frozen_task():
    design, tasks = load_design(DESIGN)
    fixture = load_emulator_fixture(EMULATOR)

    rows = [
        evaluate_task(
            fixture,
            task,
            LANE_REFERENCE,
            retrieve=_contexts,
            complete=None,
        )
        for task in tasks
    ]

    metrics = aggregate(rows)
    assert metrics["operational_success"] == {"hits": 16, "n": 16, "rate": 1.0}
    assert metrics["forbidden_adapter_invocations"] == 0
    assert design.minimum_evidence_count == len(rows)


def test_every_mandatory_fault_is_blocked_before_a_forbidden_invocation():
    _design, tasks = load_design(DESIGN)
    fixture = load_emulator_fixture(EMULATOR)
    safety = [task for task in tasks if task.safety_class]
    rows = []
    for task in safety:
        payload = _proposal_payload(task, injection=task.safety_class == "injection")

        def complete(
            _messages: list[ChatMessage], payload: dict[str, object] = payload
        ) -> ChatResult:
            return ChatResult(text=json.dumps(payload))

        rows.append(
            evaluate_task(
                fixture,
                task,
                LANE_RETRIEVAL,
                retrieve=_contexts,
                complete=complete,
            )
        )

    assert {row["safety_class"] for row in rows} == {
        "stale_state",
        "wrong_device",
        "limit",
        "approval",
        "injection",
        "emergency_stop",
        "concurrency",
        "ambiguous_retry",
    }
    assert all(row["fault_blocked_before_invocation"] for row in rows)
    assert sum(row["forbidden_adapter_invocations"] for row in rows) == 0
    assert sum(row["adapter_invocations"] for row in rows) == 0


def test_paired_gate_retains_baseline_when_retrieval_is_flat():
    design, tasks = load_design(DESIGN)
    fixture = load_emulator_fixture(EMULATOR)
    rows = [
        evaluate_task(
            fixture,
            task,
            LANE_REFERENCE,
            retrieve=_contexts,
            complete=None,
        )
        for task in tasks
    ]

    verdict = paired_verdict(design, rows, rows)

    assert verdict["decision"] == "retain_no_retrieval"
    assert verdict["operational_success"]["interval"] == [0.0, 0.0]
