# Robotics Action Gate and Device Emulator

The robotics side-effect boundary is deterministic, model-independent, and protocol-neutral. It
turns a typed `ActionProposal` into approve, deny, or escalate from trusted device, policy,
approval, and execution-state inputs. Only an approved decision can reach the emulator adapter;
retrieved text and model output cannot add an operation, widen a limit, grant approval, or bypass a
fresh device read.

## Implementation

[`action_gate.py`](../../../../src/llb/robotics/action_gate.py) is the pure decision surface. It
validates the policy and proposal digests, deployment allowlist, discovered driver contract,
driver and deployment argument limits, fresh state revision, typed policy-bound preconditions,
risk and idempotency declarations, tainted evidence, emergency stops, dependency locks, and an
operator approval bound to the exact proposal digest. Deny and escalate decisions carry a
`not_invoked` receipt and do not call the adapter.

[`action_executor.py`](../../../../src/llb/robotics/action_executor.py) owns the time-of-use
sequence: reserve the target and its declared dependencies, re-read the device, call the pure gate,
and make at most one adapter invocation. It records an unacknowledged non-idempotent write as
`outcome_unknown`, requires an explicit successful reconciliation read before another proposal,
and permanently refuses to retry the same non-idempotent proposal digest.

[`device_emulator.py`](../../../../src/llb/robotics/device_emulator.py) composes one
`ProtocolNeutralFakeDriver` per device into a workcell. It provides independent discovery and
state revisions, atomic multi-device reservations, external lock contention, dependency locking,
emergency-stop enforcement above and below the gate, and queued read/write faults. Assignment
effects model idempotent operations; additive effects model non-idempotent operations. Driver hard
limits remain enforced in [`fake_driver.py`](../../../../src/llb/robotics/fake_driver.py) even if a
deployment policy is re-signed with a wider range.

Strict policy, approval, fault, effect, and scenario records live in
[`emulator_models.py`](../../../../src/llb/robotics/emulator_models.py). The loader in
[`emulator_fixture.py`](../../../../src/llb/robotics/emulator_fixture.py) checks every policy,
proposal, approval, device-reference, effect, dependency, and cross-record digest before replay.
The committed two-device workcell and scenario ledger are in
[`samples/robotics/emulator/`](../../../../samples/robotics/emulator/).

## Run and artifacts

The operator-facing workflow runs the focused tests and then writes `report.json` and `report.md`
under `$DATA_DIR/robotics-emulator/<run>/`:

```bash
make test-robotics-emulator
```

The CLI-only replay is `llb robotics-emulator-check --fixture <scenario-json>`. Its implementation
is in [`emulator_run.py`](../../../../src/llb/robotics/emulator_run.py) and the Typer adapter is in
[`robotics_emulator.py`](../../../../src/llb/cli/robotics_emulator.py).

Focused coverage in [`tests/llb/robotics/`](../../../../tests/llb/robotics/) proves that the pure
gate does not mutate fresh state, stale policy and mismatched approvals fail closed, driver limits
win over a wider re-signed policy, dependency reservations serialize the workcell, emergency stops
also hold below the gate, and an ambiguous additive write cannot be retried before or after
reconciliation. The committed matrix remains ordinary CPU CI; no local model or CUDA runtime is
needed to exercise the physical side-effect semantics.

## Verification result

On 2026-08-30, `make test-robotics-emulator` replayed the committed two-device workcell on the RTX
4060 Ti 16 GB CUDA host with driver 595.84. This was deliberately a CPU-only, network-free,
model-free run because the side-effect boundary must work independently of inference. All 11
focused tests passed. The ledger contained 13 scenarios and 15 process attempts: 12 deny or
escalate attempts invoked the adapter zero times, while the three approved attempts made exactly
one invocation each. Those approved attempts covered one successful idempotent assignment, one
recoverable driver failure, and one additive non-idempotent write whose lost acknowledgement became
`outcome_unknown`.

The planted wrong-device, stale-state, driver-limit, deployment-limit, missing-approval,
injection-derived, emergency-stop, dependency-lock, failed-precondition, unreachable-device, and
ambiguous-retry cases all produced their declared reason. The ambiguous write moved the emulated
axis from 10.0 mm to 12.0 mm before the acknowledgement was lost; both retry attempts remained
non-invoking, including the one after an explicit read observed state revision 2. The finite-suite
reading is therefore zero executed out-of-policy actions across 12 blocked attempts, not a physical
safety proof.

This result licenses the emulator as the side-effect boundary for the held-out robotics benchmark.
It does not authorize hardware, claim MHS compatibility, validate a physical workcell, or replace
driver limits, interlocks, or an operator stop. A changed policy or device contract, a scenario
that reaches the adapter after deny/escalate, more than one invocation for a proposal, or a retry
after an ambiguous non-idempotent result would overturn the result.
