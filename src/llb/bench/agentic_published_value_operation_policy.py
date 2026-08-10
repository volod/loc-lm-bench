"""Check an operation's shipped-policy declaration by perturbing every policy field.

Input recording can observe values reached through a source, a design row, or a measurement.  A
shipped policy is different: production supplies the whole `ContextPolicy`, so the operation could
read any field whether it declared that dependency or not.  The check here asks the observable
question instead.  It runs every declared probe under the baseline policy and again with one field
changed; a declared field must change the answer at some point, and an undeclared field must change
it at none.

This is intentionally perturbation rather than general-purpose tracing.  It checks the policy
surface the pin gate can act on, while globals unrelated to that policy remain outside the claim.
"""

from dataclasses import dataclass, replace

from llb.bench.agentic.context_policy import ContextPolicy
from llb.bench.agentic_policy_change_audit import AUDITABLE_FIELDS
from llb.bench.agentic_policy_change_interaction_couplings import FIELD_MOVES
from llb.bench.agentic_published_value_operation_probe import probe_inputs
from llb.bench.agentic_published_value_operations import (
    DerivationInputs,
    DerivationOperation,
    DerivedValue,
)


@dataclass(frozen=True, slots=True)
class _ProbeOutcome:
    answer: DerivedValue | None = None
    error: tuple[str, str] | None = None


def _perturbed(policy: ContextPolicy, field: str) -> ContextPolicy:
    """Use the existing plausible field move, reversed when the probe already sits at its target."""
    baseline, candidate = FIELD_MOVES[field]
    current = getattr(policy, field)
    changed = candidate if current != candidate else baseline
    return replace(policy, **{field: changed})


def _outcome(
    operation: DerivationOperation, probe: DerivationInputs, policy: ContextPolicy
) -> _ProbeOutcome:
    """Call one point through its recording inputs and retain either answer or typed failure."""
    inputs = probe_inputs(operation, probe, set())
    try:
        answer = operation.apply(
            inputs.sources,
            inputs.stated,
            measured=inputs.measured,
            policy=policy,
            where=f"the `{operation.name}` operation's policy probe",
        )
    except Exception as exc:  # a changed failure is a changed answer for dependency purposes
        return _ProbeOutcome(error=(type(exc).__name__, str(exc)))
    return _ProbeOutcome(answer=answer)


def policy_declaration_refusals(operation: DerivationOperation) -> tuple[str, ...]:
    """Refuse missing and unused shipped-policy dependencies across the operation's probe set."""
    refusals: list[str] = []
    declared = set(operation.policy_fields)
    for field in AUDITABLE_FIELDS:
        moved = any(
            _outcome(operation, probe, probe.policy)
            != _outcome(operation, probe, _perturbed(probe.policy, field))
            for probe in operation.probes
        )
        if moved and field not in declared:
            refusals.append(
                f"the `{operation.name}` operation changes when the shipped `{field}` policy "
                "field is perturbed and does not declare it, so a policy-pin change cannot name "
                "the published values whose arithmetic it moves"
            )
        elif not moved and field in declared:
            refusals.append(
                f"the `{operation.name}` operation declares the shipped `{field}` policy field "
                f"and its answer does not change at any of its {len(operation.probes)} probe "
                "point(s) when that field is perturbed, so the dependency is not exercised"
            )
    return tuple(refusals)
