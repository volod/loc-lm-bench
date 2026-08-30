import pytest

from llb.core.paths import PROJECT_ROOT
from llb.robotics.action_executor import ActionExecutor
from llb.robotics.device_emulator import DeviceEmulator
from llb.robotics.emulator_fixture import load_emulator_fixture
from llb.robotics.emulator_models import InjectedFault
from llb.robotics.fake_driver import DriverContractError

FIXTURE_PATH = PROJECT_ROOT / "samples" / "robotics" / "emulator" / "scenarios.json"


def _fixture():
    return load_emulator_fixture(FIXTURE_PATH)


def _scenario(fixture, scenario_id: str):
    return next(item for item in fixture.scenarios if item.scenario_id == scenario_id)


def _approval(fixture, approval_id: str | None):
    return next((item for item in fixture.approvals if item.approval_id == approval_id), None)


def test_adapter_enforces_hard_limit_when_called_without_the_gate() -> None:
    fixture = _fixture()
    proposal = _scenario(fixture, "driver-hard-limit").steps[0].proposal
    assert proposal is not None
    emulator = DeviceEmulator(fixture.devices)
    with pytest.raises(DriverContractError, match="hard maximum"):
        emulator.invoke(proposal)


def test_emulator_exposes_discover_read_and_write_surfaces() -> None:
    fixture = _fixture()
    proposal = _scenario(fixture, "approved-idempotent-write").steps[0].proposal
    assert proposal is not None
    emulator = DeviceEmulator(fixture.devices)
    assert emulator.discover("arm-cell").state_revision == 1
    state = {item.name: item.value for item in emulator.read("arm-cell", "read_position")}
    assert state["position_mm"] == 10.0
    assert emulator.write(proposal).status == "succeeded"
    assert emulator.discover("arm-cell").state_revision == 2


def test_dependency_reservation_reports_external_lock_without_taking_it() -> None:
    fixture = _fixture()
    emulator = DeviceEmulator(fixture.devices)
    emulator.set_external_lock("arm-cell")
    with emulator.reserve(("clamp-cell", "arm-cell")) as unavailable:
        assert unavailable == frozenset({"arm-cell"})
    assert emulator.invocation_count == 0


def test_ambiguous_non_idempotent_write_is_never_retried() -> None:
    fixture = _fixture()
    scenario = _scenario(fixture, "ambiguous-non-idempotent-no-retry")
    first = scenario.steps[0]
    assert first.proposal is not None
    emulator = DeviceEmulator(fixture.devices)
    emulator.inject_fault(
        InjectedFault(
            device_id="arm-cell",
            operation="jog_relative",
            mode="outcome_unknown_after_apply",
        )
    )
    executor = ActionExecutor(emulator, fixture.policy)
    approval = _approval(fixture, first.approval_id)
    unknown = executor.process(first.proposal, approval)
    retry = executor.process(first.proposal, approval)
    assert unknown.receipt.status == "outcome_unknown"
    assert retry.decision.decision == "escalate"
    assert retry.decision.reasons == ("non_idempotent_retry_forbidden",)
    assert emulator.invocation_count == 1
    reconciled = executor.reconcile("arm-cell")
    state = {item.name: item.value for item in reconciled.state}
    assert reconciled.state_revision == 2
    assert state["position_mm"] == 12.0
    assert executor.process(first.proposal, approval).receipt.status == "not_invoked"
    assert emulator.invocation_count == 1


def test_emergency_stop_is_enforced_below_the_gate_too() -> None:
    fixture = _fixture()
    proposal = _scenario(fixture, "approved-idempotent-write").steps[0].proposal
    assert proposal is not None
    emulator = DeviceEmulator(fixture.devices)
    emulator.set_emergency_stop("arm-cell")
    with pytest.raises(DriverContractError, match="emergency stop"):
        emulator.invoke(proposal)
