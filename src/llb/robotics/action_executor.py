"""Single-invocation executor that composes the pure gate with the emulator."""

from llb.core.contracts.robotics import (
    ActionProposal,
    ActionReceipt,
    DeviceReference,
    DeviceSnapshot,
)
from llb.robotics.action_gate import decide_action, required_devices
from llb.robotics.device_emulator import DeviceEmulator, DeviceUnavailableError
from llb.robotics.emulator_models import (
    ActionExecution,
    ActionPolicy,
    OperatorApproval,
)
from llb.robotics.fake_driver import DriverContractError


class ActionExecutor:
    """Own outcome reconciliation and the no-retry rule above one adapter."""

    def __init__(self, emulator: DeviceEmulator, policy: ActionPolicy) -> None:
        self._emulator = emulator
        self._policy = policy
        self._unresolved_devices: set[str] = set()
        self._attempted_non_idempotent: set[str] = set()

    def process(
        self, proposal: ActionProposal, approval: OperatorApproval | None = None
    ) -> ActionExecution:
        devices = required_devices(self._policy, proposal)
        with self._emulator.reserve(devices) as unavailable:
            snapshot, reference = self._fresh_inputs(proposal.device_id)
            decision = decide_action(
                proposal,
                self._policy,
                snapshot,
                reference,
                approval,
                unavailable_devices=unavailable,
                emergency_stop_devices=self._emulator.emergency_stop_devices,
                unresolved_devices=frozenset(self._unresolved_devices),
                attempted_non_idempotent=frozenset(self._attempted_non_idempotent),
            )
            if decision.decision != "approve":
                return ActionExecution(
                    decision=decision,
                    receipt=self._not_invoked(proposal, snapshot, decision.reasons[0]),
                )
            try:
                receipt = self._emulator.invoke(proposal)
            except DriverContractError as exc:
                receipt = self._failed(proposal, snapshot, str(exc))
            if proposal.idempotency == "non_idempotent":
                self._attempted_non_idempotent.add(proposal.proposal_digest)
            if receipt.status == "outcome_unknown":
                self._unresolved_devices.add(proposal.device_id)
            return ActionExecution(decision=decision, receipt=receipt)

    def reconcile(self, device_id: str) -> DeviceSnapshot:
        """Require an explicit successful read before another proposal can proceed."""
        snapshot = self._emulator.snapshot(device_id)
        self._unresolved_devices.discard(device_id)
        return snapshot

    def _fresh_inputs(self, device_id: str) -> tuple[DeviceSnapshot | None, DeviceReference | None]:
        try:
            return self._emulator.snapshot(device_id), self._emulator.reference(device_id)
        except DeviceUnavailableError:
            return None, None

    @staticmethod
    def _not_invoked(
        proposal: ActionProposal, snapshot: DeviceSnapshot | None, reason: str
    ) -> ActionReceipt:
        revision = snapshot.state_revision if snapshot else proposal.expected_state_revision
        return ActionReceipt(
            schema_version=1,
            receipt_id=f"receipt:{proposal.proposal_id}:not-invoked",
            proposal_id=proposal.proposal_id,
            proposal_digest=proposal.proposal_digest,
            device_id=proposal.device_id,
            operation=proposal.operation,
            status="not_invoked",
            state_revision_before=revision,
            state_revision_after=None,
            error=reason,
        )

    @staticmethod
    def _failed(
        proposal: ActionProposal, snapshot: DeviceSnapshot | None, error: str
    ) -> ActionReceipt:
        revision = snapshot.state_revision if snapshot else proposal.expected_state_revision
        return ActionReceipt(
            schema_version=1,
            receipt_id=f"receipt:{proposal.proposal_id}:failed",
            proposal_id=proposal.proposal_id,
            proposal_digest=proposal.proposal_digest,
            device_id=proposal.device_id,
            operation=proposal.operation,
            status="failed",
            state_revision_before=revision,
            state_revision_after=None,
            error=error,
        )
