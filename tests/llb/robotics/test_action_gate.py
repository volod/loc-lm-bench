from llb.robotics.action_gate import decide_action
from llb.robotics.digests import value_digest
from llb.robotics.emulator_fixture import load_emulator_fixture
from llb.robotics.emulator_models import OperatorApproval
from llb.core.paths import PROJECT_ROOT

FIXTURE_PATH = PROJECT_ROOT / "samples" / "robotics" / "emulator" / "scenarios.json"


def _fixture():
    return load_emulator_fixture(FIXTURE_PATH)


def _proposal(fixture, scenario_id: str):
    scenario = next(item for item in fixture.scenarios if item.scenario_id == scenario_id)
    proposal = scenario.steps[0].proposal
    assert proposal is not None
    return proposal


def _approval(fixture, proposal_id: str):
    return next(item for item in fixture.approvals if item.approval_id == f"approval:{proposal_id}")


def test_pure_gate_approves_without_mutating_the_fresh_snapshot() -> None:
    fixture = _fixture()
    proposal = _proposal(fixture, "approved-idempotent-write")
    device = fixture.devices[0]
    before = device.snapshot.model_dump(mode="json")
    decision = decide_action(
        proposal,
        fixture.policy,
        device.snapshot,
        device.reference,
        _approval(fixture, proposal.proposal_id),
    )
    assert decision.decision == "approve"
    assert decision.proposal_digest == proposal.proposal_digest
    assert device.snapshot.model_dump(mode="json") == before


def test_gate_refuses_tampered_policy_before_using_its_limits() -> None:
    fixture = _fixture()
    proposal = _proposal(fixture, "approved-idempotent-write")
    device = fixture.devices[0]
    widened_rule = fixture.policy.operations[0].model_copy(update={"argument_limits": ()})
    tampered = fixture.policy.model_copy(
        update={"operations": (widened_rule, *fixture.policy.operations[1:])}
    )
    decision = decide_action(
        proposal,
        tampered,
        device.snapshot,
        device.reference,
        _approval(fixture, proposal.proposal_id),
    )
    assert decision.decision == "deny"
    assert decision.reasons == ("policy_digest_invalid",)


def test_gate_refuses_approval_bound_to_another_proposal() -> None:
    fixture = _fixture()
    proposal = _proposal(fixture, "approved-idempotent-write")
    device = fixture.devices[0]
    approval = OperatorApproval(
        schema_version=1,
        approval_id="approval:wrong-binding",
        approval_digest=f"sha256:{'0' * 64}",
        proposal_digest=f"sha256:{'a' * 64}",
        risk_class="low",
    )
    approval = approval.model_copy(
        update={
            "approval_digest": value_digest(
                approval.model_dump(mode="json", exclude={"approval_digest"})
            )
        }
    )
    decision = decide_action(proposal, fixture.policy, device.snapshot, device.reference, approval)
    assert decision.decision == "escalate"
    assert decision.reasons == ("approval_binding_invalid",)


def test_driver_hard_limit_wins_even_if_policy_is_re_signed_wider() -> None:
    fixture = _fixture()
    proposal = _proposal(fixture, "driver-hard-limit")
    device = fixture.devices[0]
    widened_rule = fixture.policy.operations[0].model_copy(update={"argument_limits": ()})
    widened = fixture.policy.model_copy(
        update={"operations": (widened_rule, *fixture.policy.operations[1:])}
    )
    widened = widened.model_copy(
        update={
            "policy_digest": value_digest(
                widened.model_dump(mode="json", exclude={"policy_digest"})
            )
        }
    )
    decision = decide_action(
        proposal,
        widened,
        device.snapshot,
        device.reference,
        _approval(fixture, proposal.proposal_id),
    )
    assert decision.decision == "deny"
    assert decision.reasons == ("driver_contract_invalid",)
