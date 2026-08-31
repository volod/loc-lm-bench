"""Strict JSON decision parsing and trusted construction of proposal identity fields."""

import json

from pydantic import ValidationError

from llb.core.contracts.robotics import ActionProposal
from llb.robotics.benchmark.models import BenchmarkTask, ModelDecision
from llb.robotics.digests import value_digest


def _json_object(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    start = stripped.find("{")
    if start < 0:
        raise ValueError("model output contains no JSON object")
    try:
        payload, end = json.JSONDecoder().raw_decode(stripped[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"model output is not valid JSON: {exc.msg}") from None
    if stripped[start + end :].strip():
        raise ValueError("model output contains trailing text after its JSON object")
    return payload


def parse_model_decision(text: str) -> ModelDecision:
    try:
        return ModelDecision.model_validate_json(json.dumps(_json_object(text)))
    except ValidationError as exc:
        raise ValueError(f"model decision violates the typed schema: {exc}") from None


def action_proposal(
    decision: ModelDecision, task: BenchmarkTask, lane: str, state_revision: int
) -> ActionProposal:
    if decision.proposal is None:
        raise ValueError("cannot construct an action proposal from a refusal")
    proposal = ActionProposal(
        schema_version=1,
        proposal_id=f"{task.task_id}:{lane}",
        proposal_digest=f"sha256:{'0' * 64}",
        expected_state_revision=state_revision,
        **decision.proposal.model_dump(mode="python"),
    )
    digest = value_digest(proposal.model_dump(mode="json", exclude={"proposal_digest"}))
    return proposal.model_copy(update={"proposal_digest": digest})
