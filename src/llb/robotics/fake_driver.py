"""Protocol-neutral fake for public discover/read/write/reference/limit semantics."""

from llb.core.contracts.robotics import (
    ActionProposal,
    ActionReceipt,
    DeviceOperation,
    DeviceParameter,
    DeviceReference,
    DeviceSnapshot,
    NamedValue,
    ScalarValue,
)
from llb.robotics.digests import value_digest


class DriverContractError(ValueError):
    """The caller requested an operation outside the discovered device contract."""


def _operation(reference: DeviceReference, name: str) -> DeviceOperation:
    matches = [operation for operation in reference.operations if operation.name == name]
    if len(matches) != 1:
        raise DriverContractError(f"unknown operation: {name}")
    return matches[0]


def _matches_type(value: ScalarValue, value_type: str) -> bool:
    if value_type == "boolean":
        return isinstance(value, bool)
    if value_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if value_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if value_type == "string":
        return isinstance(value, str)
    return False


def _validate_value(parameter: DeviceParameter, value: ScalarValue) -> None:
    if not _matches_type(value, parameter.value_type):
        raise DriverContractError(
            f"argument {parameter.name} must have type {parameter.value_type}"
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if parameter.minimum is not None and value < parameter.minimum:
            raise DriverContractError(
                f"argument {parameter.name} is below hard minimum {parameter.minimum}"
            )
        if parameter.maximum is not None and value > parameter.maximum:
            raise DriverContractError(
                f"argument {parameter.name} exceeds hard maximum {parameter.maximum}"
            )


def _validate_arguments(
    operation: DeviceOperation, arguments: tuple[NamedValue, ...]
) -> dict[str, ScalarValue]:
    values = {argument.name: argument.value for argument in arguments}
    if len(values) != len(arguments):
        raise DriverContractError("duplicate operation arguments")
    parameters = {parameter.name: parameter for parameter in operation.parameters}
    unknown = set(values) - set(parameters)
    if unknown:
        raise DriverContractError(f"unknown operation arguments: {sorted(unknown)}")
    missing = {name for name, parameter in parameters.items() if parameter.required} - set(values)
    if missing:
        raise DriverContractError(f"missing operation arguments: {sorted(missing)}")
    for name, value in values.items():
        _validate_value(parameters[name], value)
    return values


class ProtocolNeutralFakeDriver:
    """In-memory driver with no HFlow, MHS, transport, or credential dependency."""

    def __init__(self, reference: DeviceReference, snapshot: DeviceSnapshot) -> None:
        if reference.device_id != snapshot.device_id:
            raise DriverContractError("reference and snapshot device ids differ")
        if reference.reference_digest != snapshot.reference_digest:
            raise DriverContractError("reference and snapshot digests differ")
        if reference.operations != snapshot.operations:
            raise DriverContractError("reference and snapshot operations differ")
        observed_digest = value_digest(
            reference.model_dump(mode="json", exclude={"reference_digest"})
        )
        if reference.reference_digest != observed_digest:
            raise DriverContractError("device reference digest does not match its contract")
        self._reference = reference
        self._snapshot = snapshot

    def discover(self) -> DeviceSnapshot:
        return self._snapshot

    def reference(self) -> DeviceReference:
        return self._reference

    def read(
        self, operation_name: str, arguments: tuple[NamedValue, ...] = ()
    ) -> tuple[NamedValue, ...]:
        operation = _operation(self._reference, operation_name)
        if operation.access != "read":
            raise DriverContractError(f"operation is not readable: {operation_name}")
        _validate_arguments(operation, arguments)
        return self._snapshot.state

    def write(self, proposal: ActionProposal) -> ActionReceipt:
        observed_digest = value_digest(
            proposal.model_dump(mode="json", exclude={"proposal_digest"})
        )
        if proposal.proposal_digest != observed_digest:
            raise DriverContractError("proposal digest does not match its contract")
        if proposal.device_id != self._snapshot.device_id:
            raise DriverContractError("proposal targets a different device")
        if proposal.expected_state_revision != self._snapshot.state_revision:
            raise DriverContractError("proposal state revision is stale")
        operation = _operation(self._reference, proposal.operation)
        if operation.access != "write":
            raise DriverContractError(f"operation is not writable: {proposal.operation}")
        values = _validate_arguments(operation, proposal.arguments)
        current = {item.name: item.value for item in self._snapshot.state}
        current.update(values)
        next_revision = self._snapshot.state_revision + 1
        state = tuple(NamedValue(name=name, value=value) for name, value in current.items())
        self._snapshot = self._snapshot.model_copy(
            update={
                "snapshot_id": f"{self._snapshot.device_id}:state:{next_revision}",
                "state_revision": next_revision,
                "observed_at": "fixture-replay",
                "state": state,
            }
        )
        return ActionReceipt(
            schema_version=1,
            receipt_id=f"receipt:{proposal.proposal_id}:{next_revision}",
            proposal_id=proposal.proposal_id,
            proposal_digest=proposal.proposal_digest,
            device_id=proposal.device_id,
            operation=proposal.operation,
            status="succeeded",
            state_revision_before=proposal.expected_state_revision,
            state_revision_after=next_revision,
            result=proposal.arguments,
            error=None,
        )

    def validate_limit_probe(self, operation_name: str, arguments: tuple[NamedValue, ...]) -> None:
        operation = _operation(self._reference, operation_name)
        _validate_arguments(operation, arguments)
