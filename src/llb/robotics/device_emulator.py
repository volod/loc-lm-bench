"""Multi-device emulator with locks, emergency stops, and deterministic faults."""

from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Literal

from llb.core.contracts.robotics import (
    ActionProposal,
    ActionReceipt,
    DeviceReference,
    DeviceSnapshot,
    NamedValue,
)
from llb.robotics.emulator_models import EmulatorDevice, InjectedFault
from llb.robotics.fake_driver import DriverContractError, ProtocolNeutralFakeDriver


class DeviceUnavailableError(RuntimeError):
    """The emulator could not provide a fresh device observation."""


class DeviceEmulator:
    """Protocol-neutral workcell with no model, transport, or hardware dependency."""

    def __init__(self, devices: tuple[EmulatorDevice, ...]) -> None:
        self._drivers = {
            device.reference.device_id: ProtocolNeutralFakeDriver(
                device.reference,
                device.snapshot,
                self._effect_map(device),
            )
            for device in devices
        }
        if len(self._drivers) != len(devices):
            raise DriverContractError("duplicate emulator device id")
        self._faults: dict[tuple[str, str | None], deque[str]] = defaultdict(deque)
        self._emergency_stops: set[str] = set()
        self._external_locks: set[str] = set()
        self._reservations: set[str] = set()
        self._lock = Lock()
        self._invocations: list[str] = []

    @staticmethod
    def _effect_map(device: EmulatorDevice) -> dict[str, dict[str, tuple[str, str]]]:
        effects: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
        for effect in device.effects:
            if effect.argument_name in effects[effect.operation]:
                raise DriverContractError("duplicate emulator operation effect")
            effects[effect.operation][effect.argument_name] = (
                effect.state_name,
                effect.mode,
            )
        return dict(effects)

    @property
    def invocation_count(self) -> int:
        return len(self._invocations)

    @property
    def emergency_stop_devices(self) -> frozenset[str]:
        return frozenset(self._emergency_stops)

    def reference(self, device_id: str) -> DeviceReference:
        try:
            return self._drivers[device_id].reference()
        except KeyError as exc:
            raise DeviceUnavailableError(f"unknown device: {device_id}") from exc

    def snapshot(self, device_id: str) -> DeviceSnapshot:
        fault = self._take_fault(device_id, None, "unreachable_read")
        if fault is not None:
            raise DeviceUnavailableError(f"device read failed: {device_id}")
        try:
            return self._drivers[device_id].discover()
        except KeyError as exc:
            raise DeviceUnavailableError(f"unknown device: {device_id}") from exc

    def discover(self, device_id: str) -> DeviceSnapshot:
        """Discover one device and its current revision."""
        return self.snapshot(device_id)

    def read(
        self, device_id: str, operation: str, arguments: tuple[NamedValue, ...] = ()
    ) -> tuple[NamedValue, ...]:
        """Read through the discovered driver contract without write authority."""
        if self._take_fault(device_id, operation, "unreachable_read") is not None:
            raise DeviceUnavailableError(f"device read failed: {device_id}")
        try:
            return self._drivers[device_id].read(operation, arguments)
        except KeyError as exc:
            raise DeviceUnavailableError(f"unknown device: {device_id}") from exc

    def inject_fault(self, fault: InjectedFault) -> None:
        if fault.device_id not in self._drivers:
            raise DeviceUnavailableError(f"unknown fault device: {fault.device_id}")
        self._faults[(fault.device_id, fault.operation)].append(fault.mode)

    def set_emergency_stop(self, device_id: str, active: bool = True) -> None:
        self.reference(device_id)
        if active:
            self._emergency_stops.add(device_id)
        else:
            self._emergency_stops.discard(device_id)

    def set_external_lock(self, device_id: str, active: bool = True) -> None:
        self.reference(device_id)
        if active:
            self._external_locks.add(device_id)
        else:
            self._external_locks.discard(device_id)

    @contextmanager
    def reserve(self, device_ids: tuple[str, ...]) -> Iterator[frozenset[str]]:
        requested = set(device_ids)
        with self._lock:
            unavailable = requested & (self._external_locks | self._reservations)
            if not unavailable:
                self._reservations.update(requested)
        try:
            yield frozenset(unavailable)
        finally:
            if not unavailable:
                with self._lock:
                    self._reservations.difference_update(requested)

    def invoke(self, proposal: ActionProposal) -> ActionReceipt:
        """Invoke exactly once; driver checks remain authoritative below the gate."""
        if proposal.device_id in self._emergency_stops:
            raise DriverContractError("emergency stop is active")
        try:
            driver = self._drivers[proposal.device_id]
        except KeyError as exc:
            raise DriverContractError("proposal targets an unknown device") from exc
        self._invocations.append(proposal.proposal_digest)
        fault = self._take_fault(proposal.device_id, proposal.operation)
        if fault == "write_failed":
            return self._fault_receipt(proposal, "failed", "injected write failure")
        if fault == "outcome_unknown_before_apply":
            return self._fault_receipt(proposal, "outcome_unknown", "acknowledgement lost")
        if fault == "outcome_unknown_after_apply":
            driver.write(proposal)
            return self._fault_receipt(proposal, "outcome_unknown", "acknowledgement lost")
        return driver.write(proposal)

    def write(self, proposal: ActionProposal) -> ActionReceipt:
        """Explicit adapter write alias used by conformance callers."""
        return self.invoke(proposal)

    def _take_fault(
        self, device_id: str, operation: str | None, expected: str | None = None
    ) -> str | None:
        queue = self._faults[(device_id, operation)]
        if not queue:
            return None
        if expected is not None and queue[0] != expected:
            return None
        return queue.popleft()

    def _fault_receipt(
        self,
        proposal: ActionProposal,
        status: Literal["failed", "outcome_unknown"],
        error: str,
    ) -> ActionReceipt:
        return ActionReceipt(
            schema_version=1,
            receipt_id=f"receipt:{proposal.proposal_id}:fault:{self.invocation_count}",
            proposal_id=proposal.proposal_id,
            proposal_digest=proposal.proposal_digest,
            device_id=proposal.device_id,
            operation=proposal.operation,
            status=status,
            state_revision_before=proposal.expected_state_revision,
            state_revision_after=None,
            error=error,
        )
