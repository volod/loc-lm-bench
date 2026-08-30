"""Load and cross-check the committed device-emulator scenario fixture."""

from pathlib import Path

from llb.core.contracts.robotics import DeviceReference
from llb.robotics.digests import value_digest
from llb.robotics.emulator_models import EmulatorDevice, EmulatorFixture, OperationPolicy
from llb.robotics.fake_driver import ProtocolNeutralFakeDriver, operation_for


def _check_digest(record: object, field: str) -> None:
    model_dump = getattr(record, "model_dump")
    declared = getattr(record, field)
    observed = value_digest(model_dump(mode="json", exclude={field}))
    if declared != observed:
        raise ValueError(f"{field} does not match its record")


def _validate_device(device: EmulatorDevice) -> None:
    ProtocolNeutralFakeDriver(device.reference, device.snapshot)
    operations = {operation.name: operation for operation in device.reference.operations}
    state_names = {item.name for item in device.snapshot.state}
    effect_keys: set[tuple[str, str]] = set()
    for effect in device.effects:
        if effect.operation not in operations:
            raise ValueError(f"effect names unknown operation: {effect.operation}")
        if effect.state_name not in state_names:
            raise ValueError(f"effect names unknown state: {effect.state_name}")
        parameters = {parameter.name for parameter in operations[effect.operation].parameters}
        if effect.argument_name not in parameters:
            raise ValueError(f"effect names unknown argument: {effect.argument_name}")
        key = (effect.operation, effect.argument_name)
        if key in effect_keys:
            raise ValueError(f"duplicate emulator effect: {key}")
        effect_keys.add(key)


def _validate_devices(devices: tuple[EmulatorDevice, ...]) -> dict[str, DeviceReference]:
    references: dict[str, DeviceReference] = {}
    for device in devices:
        _validate_device(device)
        device_id = device.reference.device_id
        if device_id in references:
            raise ValueError(f"duplicate emulator device: {device_id}")
        references[device_id] = device.reference
    return references


def _validate_policy_rule(rule: OperationPolicy, references: dict[str, DeviceReference]) -> None:
    key = (rule.device_id, rule.operation)
    if rule.device_id not in references:
        raise ValueError(f"policy names unknown device: {rule.device_id}")
    operation = operation_for(references[rule.device_id], rule.operation)
    if operation.access != "write":
        raise ValueError(f"policy cannot authorize a non-write operation: {key}")
    if set(rule.dependencies) - set(references):
        raise ValueError(f"policy names unknown device dependency: {key}")
    parameters = {parameter.name for parameter in operation.parameters}
    unknown_limits = {limit.name for limit in rule.argument_limits} - parameters
    if unknown_limits:
        raise ValueError(f"policy limits unknown arguments: {sorted(unknown_limits)}")


def _validate_policy(fixture: EmulatorFixture, references: dict[str, DeviceReference]) -> None:
    _check_digest(fixture.policy, "policy_digest")
    keys = [(rule.device_id, rule.operation) for rule in fixture.policy.operations]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate operation policy")
    for rule in fixture.policy.operations:
        _validate_policy_rule(rule, references)


def _approval_ids(fixture: EmulatorFixture) -> set[str]:
    approval_ids: set[str] = set()
    for approval in fixture.approvals:
        _check_digest(approval, "approval_digest")
        if approval.approval_id in approval_ids:
            raise ValueError(f"duplicate approval: {approval.approval_id}")
        approval_ids.add(approval.approval_id)
    return approval_ids


def _validate_scenarios(fixture: EmulatorFixture, approval_ids: set[str]) -> None:
    for scenario in fixture.scenarios:
        for step in scenario.steps:
            if step.proposal is not None:
                _check_digest(step.proposal, "proposal_digest")
            if step.approval_id is not None and step.approval_id not in approval_ids:
                raise ValueError(f"unknown approval: {step.approval_id}")


def load_emulator_fixture(path: Path) -> EmulatorFixture:
    fixture = EmulatorFixture.model_validate_json(path.read_text(encoding="utf-8"))
    references = _validate_devices(fixture.devices)
    _validate_policy(fixture, references)
    _validate_scenarios(fixture, _approval_ids(fixture))
    return fixture
