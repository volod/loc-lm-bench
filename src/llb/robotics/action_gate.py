"""Deterministic robotics action policy gate with no adapter side effects."""

from typing import Literal

from llb.core.contracts.robotics import (
    ActionProposal,
    DeviceReference,
    DeviceSnapshot,
    GateDecision,
    NamedValue,
)
from llb.robotics.digests import value_digest
from llb.robotics.emulator_models import ActionPolicy, OperatorApproval, OperationPolicy
from llb.robotics.fake_driver import DriverContractError, validate_operation_arguments

DECISION_SCHEMA_VERSION: Literal[1] = 1
GateOutcome = tuple[Literal["deny", "escalate"], str]


def _rule(policy: ActionPolicy, proposal: ActionProposal) -> OperationPolicy | None:
    matches = [
        rule
        for rule in policy.operations
        if rule.device_id == proposal.device_id and rule.operation == proposal.operation
    ]
    return matches[0] if len(matches) == 1 else None


def required_devices(policy: ActionPolicy, proposal: ActionProposal) -> tuple[str, ...]:
    rule = _rule(policy, proposal)
    dependencies = rule.dependencies if rule is not None else ()
    return tuple(dict.fromkeys((proposal.device_id, *dependencies)))


def _state(snapshot: DeviceSnapshot) -> dict[str, object]:
    return {item.name: item.value for item in snapshot.state}


def _policy_argument_reason(rule: OperationPolicy, arguments: tuple[NamedValue, ...]) -> str | None:
    values = {argument.name: argument.value for argument in arguments}
    for limit in rule.argument_limits:
        value = values.get(limit.name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return "deployment_argument_invalid"
        if limit.minimum is not None and value < limit.minimum:
            return "deployment_limit_exceeded"
        if limit.maximum is not None and value > limit.maximum:
            return "deployment_limit_exceeded"
    return None


def _precondition_reason(
    rule: OperationPolicy, proposal: ActionProposal, snapshot: DeviceSnapshot
) -> str | None:
    expected = {condition.statement for condition in rule.preconditions}
    if len(set(proposal.preconditions)) != len(proposal.preconditions):
        return "precondition_contract_invalid"
    if set(proposal.preconditions) != expected:
        return "precondition_contract_invalid"
    current = _state(snapshot)
    if any(
        current.get(condition.state_name) != condition.value for condition in rule.preconditions
    ):
        return "precondition_failed"
    return None


def _approval_reason(
    policy: ActionPolicy,
    proposal: ActionProposal,
    approval: OperatorApproval | None,
) -> str | None:
    if approval is None:
        return "approval_required" if proposal.risk_class in policy.approval_required else None
    observed = value_digest(approval.model_dump(mode="json", exclude={"approval_digest"}))
    if approval.approval_digest != observed:
        return "approval_digest_invalid"
    if approval.proposal_digest != proposal.proposal_digest:
        return "approval_binding_invalid"
    if approval.risk_class != proposal.risk_class:
        return "approval_risk_invalid"
    return None


def _decision(
    policy: ActionPolicy,
    proposal: ActionProposal,
    snapshot_id: str,
    decision: Literal["approve", "deny", "escalate"],
    reason: str,
    approval: OperatorApproval | None = None,
) -> GateDecision:
    return GateDecision(
        schema_version=DECISION_SCHEMA_VERSION,
        decision_id=f"decision:{proposal.proposal_id}:{snapshot_id}",
        proposal_id=proposal.proposal_id,
        proposal_digest=proposal.proposal_digest,
        snapshot_id=snapshot_id,
        policy_digest=policy.policy_digest,
        decision=decision,
        reasons=(reason,),
        approval_id=approval.approval_id if approval else None,
        approval_digest=approval.approval_digest if approval else None,
    )


def _integrity_outcome(
    proposal: ActionProposal,
    policy: ActionPolicy,
    attempted_non_idempotent: frozenset[str],
    unresolved_devices: frozenset[str],
) -> tuple[OperationPolicy | None, GateOutcome | None]:
    observed_policy = value_digest(policy.model_dump(mode="json", exclude={"policy_digest"}))
    if policy.policy_digest != observed_policy:
        return None, ("deny", "policy_digest_invalid")
    observed = value_digest(proposal.model_dump(mode="json", exclude={"proposal_digest"}))
    if proposal.proposal_digest != observed:
        return None, ("deny", "proposal_digest_invalid")
    if proposal.proposal_digest in attempted_non_idempotent:
        return None, ("escalate", "non_idempotent_retry_forbidden")
    if proposal.device_id in unresolved_devices:
        return None, ("escalate", "unresolved_outcome_requires_reconciliation")
    rule = _rule(policy, proposal)
    if rule is None:
        return None, ("deny", "operation_not_allowed")
    return rule, None


def _live_outcome(
    proposal: ActionProposal,
    policy: ActionPolicy,
    rule: OperationPolicy,
    snapshot: DeviceSnapshot | None,
    reference: DeviceReference | None,
    unavailable_devices: frozenset[str],
    emergency_stop_devices: frozenset[str],
) -> GateOutcome | None:
    if snapshot is None or reference is None:
        return "escalate", "device_unreachable"
    if snapshot.device_id != proposal.device_id or reference.device_id != proposal.device_id:
        return "deny", "device_identity_mismatch"
    if proposal.risk_class != rule.risk_class:
        return "deny", "risk_class_mismatch"
    if proposal.idempotency != rule.idempotency:
        return "deny", "idempotency_mismatch"
    if set(proposal.evidence_ids) & set(policy.tainted_evidence_ids):
        return "deny", "tainted_evidence"
    if set(required_devices(policy, proposal)) & set(emergency_stop_devices):
        return "deny", "emergency_stop_active"
    if unavailable_devices:
        return "escalate", "device_lock_conflict"
    if proposal.expected_state_revision != snapshot.state_revision:
        return "escalate", "state_revision_stale"
    return None


def _contract_outcome(
    proposal: ActionProposal,
    policy: ActionPolicy,
    rule: OperationPolicy,
    snapshot: DeviceSnapshot,
    reference: DeviceReference,
    approval: OperatorApproval | None,
) -> GateOutcome | None:
    try:
        validate_operation_arguments(
            reference, proposal.operation, proposal.arguments, access="write"
        )
    except DriverContractError:
        return "deny", "driver_contract_invalid"
    argument_reason = _policy_argument_reason(rule, proposal.arguments)
    if argument_reason:
        return "deny", argument_reason
    precondition_reason = _precondition_reason(rule, proposal, snapshot)
    if precondition_reason:
        return "deny", precondition_reason
    approval_reason = _approval_reason(policy, proposal, approval)
    if approval_reason:
        return "escalate", approval_reason
    return None


def decide_action(
    proposal: ActionProposal,
    policy: ActionPolicy,
    snapshot: DeviceSnapshot | None,
    reference: DeviceReference | None,
    approval: OperatorApproval | None = None,
    *,
    unavailable_devices: frozenset[str] = frozenset(),
    emergency_stop_devices: frozenset[str] = frozenset(),
    unresolved_devices: frozenset[str] = frozenset(),
    attempted_non_idempotent: frozenset[str] = frozenset(),
) -> GateDecision:
    """Return a side-effect-free decision from explicit, trusted inputs."""
    snapshot_id = snapshot.snapshot_id if snapshot else f"unavailable:{proposal.device_id}"
    rule, outcome = _integrity_outcome(
        proposal, policy, attempted_non_idempotent, unresolved_devices
    )
    if outcome is None and rule is not None:
        outcome = _live_outcome(
            proposal,
            policy,
            rule,
            snapshot,
            reference,
            unavailable_devices,
            emergency_stop_devices,
        )
    if outcome is None and rule is not None and snapshot is not None and reference is not None:
        outcome = _contract_outcome(proposal, policy, rule, snapshot, reference, approval)
    if outcome is not None:
        return _decision(policy, proposal, snapshot_id, *outcome)
    return _decision(policy, proposal, snapshot_id, "approve", "policy_checks_passed", approval)
