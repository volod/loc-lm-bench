import pytest

from llb.core.contracts.robotics import NamedValue
from llb.core.paths import PROJECT_ROOT
from llb.robotics.digests import value_digest
from llb.robotics.fake_driver import DriverContractError, ProtocolNeutralFakeDriver
from llb.robotics.fixtures import load_fixture

FIXTURE_ROOT = PROJECT_ROOT / "samples" / "robotics" / "contracts"


def _driver() -> tuple[ProtocolNeutralFakeDriver, object]:
    records = load_fixture(FIXTURE_ROOT).records
    return ProtocolNeutralFakeDriver(records.device_reference, records.device_snapshot), records


def test_fake_exposes_only_discovered_reference_and_state() -> None:
    driver, records = _driver()
    assert driver.discover() == records.device_snapshot
    assert driver.reference() == records.device_reference
    assert driver.read("read_position") == records.device_snapshot.state


def test_fake_refuses_unknown_arguments_and_hard_limit() -> None:
    driver, _records = _driver()
    with pytest.raises(DriverContractError, match="unknown operation arguments"):
        driver.validate_limit_probe("move_absolute", (NamedValue(name="speed", value=1.0),))
    with pytest.raises(DriverContractError, match="hard maximum"):
        driver.validate_limit_probe("move_absolute", (NamedValue(name="position_mm", value=101.0),))


def test_fake_refuses_stale_write_without_mutating_state() -> None:
    driver, records = _driver()
    stale = records.action_proposal.model_copy(update={"expected_state_revision": 6})
    stale = stale.model_copy(
        update={
            "proposal_digest": value_digest(
                stale.model_dump(mode="json", exclude={"proposal_digest"})
            )
        }
    )
    before = driver.discover()
    with pytest.raises(DriverContractError, match="state revision is stale"):
        driver.write(stale)
    assert driver.discover() == before


def test_fake_write_returns_the_pinned_receipt() -> None:
    driver, records = _driver()
    assert driver.write(records.action_proposal) == records.action_receipt
    assert driver.discover().state_revision == 8
