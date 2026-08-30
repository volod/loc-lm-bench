# Robotics Boundary Contracts and Upstream Pins

The first robotics seam is strict, replayable, and independent of both upstream packages. It names
the exact HFlow file contract used by future episode projection, records the smaller public surface
currently available from the Model Hardware Standard (MHS) preview, and refuses to turn preview
prose into a compatibility claim. HFlow and MHS are not base-install dependencies.

## Exchange records

[`src/llb/core/contracts/robotics.py`](../../../../src/llb/core/contracts/robotics.py) defines frozen
Pydantic records with unknown fields forbidden:

- `RoboticsEvidence` binds an accepted, quarantined, or unsettled text projection to a canonical
  MCAP digest,
  episode, channels, half-open nanosecond interval, producer versions, and character offsets.
- `DeviceSnapshot` binds live state and its revision to the independently digested device reference
  and discovered operation set.
- `ActionProposal` carries typed arguments, the expected state revision, evidence ids,
  preconditions, postconditions, risk class, and idempotency without invoking an operation.
- `GateDecision` binds approve, deny, or escalate to the exact proposal digest, snapshot, policy,
  and optional approval.
- `ActionReceipt` records whether an invocation succeeded, failed, was not invoked, or has an
  ambiguous outcome, together with the before/after state revisions.

`DeviceReference` is the sixth supporting record. It carries the discoverable operations,
parameter types, access modes, hard ranges, driver identity, and its own digest. The record is kept
separate from model-retrieved prose so retrieval cannot add an operation or widen a limit.

The strict validators and cross-record checks live in
[`src/llb/robotics/fixtures.py`](../../../../src/llb/robotics/fixtures.py). They verify both embedded
record digests and the links from evidence -> proposal -> decision -> receipt. A changed model schema
invalidates the fixture's contract-schema digest before replay begins.

## Pinned upstream boundary

The committed [`upstreams.json`](../../../../samples/robotics/contracts/upstreams.json) records the
following boundary. Discovery happened outside CI; tests and normal runs use only these pinned
bytes.

| Source | Exact pin | Inspectable contract | License | Surfaces consumed |
| --- | --- | --- | --- | --- |
| HFlow | tag `v0.2.3`, commit `d2e0f3700f2267cfeb0db1957743bb9f5f41256b` | `docs/FORMAT.md`, with architecture and package metadata retained as references | Apache-2.0 | MCAP, Parquet, DuckDB manifest |
| MHS | public research-preview publication `research-preview-2026-08-27`; no public schema revision | none | none published for an inspectable package or schema | only the announced discover, read, write, generated-reference, hard-limit semantics over MCP, CLI, or code APIs |

HFlow's pinned reference hashes are `45222345f72b720f56b3dd36677602fa9712c80dc43c515769902ad4e5e92ec1`
for `docs/FORMAT.md`, `7530d880950e6e5bf11af2821c3b8b09a0eaa17fb5ad1baec7d7a5b054efe1c9`
for `docs/ARCHITECTURE.md`, `8b1af457f86957a9669c63691a414954ec580826bb6aca784f4055e103dd5330`
for `pyproject.toml`, and `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`
for `LICENSE`. The fixture therefore goes stale if the declared release, revision, or reference
content moves without an explicit repin.

The MHS announcement and [limited-preview page](https://www.modelhardwarestandard.com/) expose no
public normative schema, SDK revision, or applicable source/package license. The local
[`mhs-public-semantics.json`](../../../../samples/robotics/contracts/mhs-public-semantics.json)
preserves only a paraphrased, inspectable snapshot of the five public semantics. It contains no
preview bytes or inferred private schema. Classification has three deliberate levels:

1. `protocol-neutral` when the normative contract, revision, and license are not jointly
   inspectable;
2. `contract-inspectable` when those inputs exist but no named passing conformance input does; and
3. `MHS-compatible` only when a named, digest-bound conformance input for that schema revision
   passes.

The stopped public-schema question and its reopening condition are in
[future research](../../future-research.md#robotics-boundary-contract-and-upstream-pins----whether-mhs-has-a-public-conformance-contract).

## Protocol-neutral fake and report

[`src/llb/robotics/fake_driver.py`](../../../../src/llb/robotics/fake_driver.py) implements an
in-memory fake over exactly discovery, device reference, read, write, and driver-side limit
validation. It rejects unknown devices, operations, arguments, types, stale revisions, and values
outside the discovered hard range. Its direct `write` method is an adapter-conformance surface, not
an action gate; policy, approval, fresh-read, concurrency, and ambiguous-write handling remain
separate forward work and no production caller is authorized by this fake.

Committed inputs live under [`samples/robotics/contracts/`](../../../../samples/robotics/contracts/).
`fixture-manifest.json` binds every fixture byte, the generated Pydantic schema set, and independent
digests of both upstream pin records. No network lookup is part of the replay. The runner in
[`src/llb/robotics/check.py`](../../../../src/llb/robotics/check.py) writes `report.json` and
`report.md` under `$DATA_DIR/robotics-contract/<run>/`:

```bash
make robotics-contract-check
```

The CLI entry point is in [`src/llb/cli/robotics.py`](../../../../src/llb/cli/robotics.py), while
[`make/robotics.mk`](../../../../make/robotics.mk) is the operator-facing workflow.

## Verification result

On 2026-08-30, `make robotics-contract-check` replayed the committed fixture on the RTX 4060 Ti 16
GB CUDA host. This was deliberately a CPU-only, network-free contract run: it loaded no model,
device, HFlow package, or MHS package. All six records round-tripped under strict validation, and
discovery, reference, read, write, and the planted hard-limit refusal all passed. The result was
`protocol-neutral`, because the MHS source had no normative schema revision, public license, or
named conformance input. The contract-schema digest is
`477171dd8a41ca04bb0a13134e4b000755a3b16383f9632b0112c383a67dff0f`; the fixture-manifest digest
is `b325df09902e95d2de725fd1803e4533eab30566a090a5bbb714ee80f3b719cd`.

The reading is overturned by any schema or pinned-byte drift, or by an inspectable MHS contract and
license that can be exercised through a named conformance input. It says only that this boundary is
reproducible and honestly labelled; it is not hardware evidence, an action-gate safety verdict, or
MHS conformance.

The focused tests in [`tests/llb/robotics/`](../../../../tests/llb/robotics/) cover unknown-field
refusal for every record, schema/release/reference staleness, fake-driver constraints, both
protocol-neutral and contract-inspectable classification, and the named-input requirement for an
MHS-compatible label. The same pinned replay runs inside quick CI through these tests.
