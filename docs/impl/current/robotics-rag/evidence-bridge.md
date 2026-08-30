# HFlow Robotics Evidence Bridge

The offline bridge turns a strict, version-pinned HFlow Parquet projection manifest into the
existing text-corpus format without importing HFlow at replay time. It preserves the link from
each projection to its canonical MCAP bytes, channel/time window, quality state, and complete
producer generation. The bridge does not operate a robot, run Airflow, choose quality thresholds,
or put raw media in a model prompt.

## Boundary and admission rules

The strict models in
[`src/llb/robotics/evidence_models.py`](../../../../src/llb/robotics/evidence_models.py) and the
Parquet codec in
[`src/llb/robotics/hflow_manifest.py`](../../../../src/llb/robotics/hflow_manifest.py) require every
row to carry:

- the exact HFlow release and revision, format schema, pipeline version, curation-query digest,
  and non-empty check and enrichment version sets;
- the content-addressed episode id, canonical MCAP URI and digest, channel set, and half-open
  nanosecond interval;
- projection kind, author, language, verification state, source URI and digest, and exact
  character offsets; and
- quality state plus quarantine tags when quarantined.

[`src/llb/robotics/mcap_validation.py`](../../../../src/llb/robotics/mcap_validation.py) opens the
canonical bytes with the stock `mcap` reader. It checks the summary, content-addressed episode id,
digest, channels, bounds, and the presence of at least one referenced message in the interval. A
complete fixture manifest pins every portable input byte. Mixed HFlow generations, stale files,
missing producer versions, unknown fields, invalid intervals, or mismatched episode identities are
refused before corpus ingestion.

Admission is fail-closed:

- accepted HFlow quality plus a verified human or pipeline projection enters the corpus;
- accepted HFlow quality plus a model-authored projection enters only when the existing accepted
  ledger binds that projection id to the exact source text and offsets;
- an unverified projection remains a draft; and
- quarantined or unsettled-quality evidence remains in the evidence ledger but receives no corpus
  source span.

[`src/llb/robotics/evidence_bridge.py`](../../../../src/llb/robotics/evidence_bridge.py) stages only
admitted text, calls the shared mixed-corpus ingest, validates the resulting source spans through
the existing retrieval overlap predicate, and records the normal corpus fingerprint. Its
`evidence-ledger.jsonl` retains all admissions and exclusions with a `RoboticsEvidence` record;
`report.json` and `report.md` summarize the run under `$DATA_DIR/robotics-evidence/<run>/`.

## Pinned upstream exercise and offline replay

The committed fixture in
[`samples/robotics/hflow/`](../../../../samples/robotics/hflow/) contains two canonical MCAP files,
a standard Parquet projection manifest, five text projections, and an accepted-ledger fixture for
the verified model projection. It pins HFlow `v0.2.3` at commit
`d2e0f3700f2267cfeb0db1957743bb9f5f41256b`. HFlow is not a base dependency: the explicit upstream
exercise installs that exact VCS revision in an isolated `uv` environment.

[`src/llb/robotics/hflow_integration.py`](../../../../src/llb/robotics/hflow_integration.py)
synthesizes one smooth and one discontinuous joint-state recording. The same HFlow `App` runs
`app.test(record=True)` on both, quarantines the planted discontinuity, runs one enrichment on the
clean episode, and uses `hflow.curate` to produce a two-row Parquet selection. It verifies those
curated rows against the cataloged `app.test` identities and versions, builds the portable fixture,
then invokes the offline bridge. Airflow internals are never imported.

Run the network-free committed replay or the exact upstream exercise with:

```bash
make robotics-evidence-replay
make robotics-evidence-fixture
```

The CLI commands are `robotics-evidence-bridge` and `robotics-hflow-integration`; their registration
lives in
[`src/llb/cli/robotics_evidence.py`](../../../../src/llb/cli/robotics_evidence.py). Operator details
are in the [HFlow evidence guide](../../../guides/data-prep/robotics-hflow-evidence.md).

## Verification result

On 2026-08-30, `make robotics-evidence-fixture` passed on the RTX 4060 Ti 16 GB CUDA host. The run
was CPU-only because this boundary transforms and validates data rather than invoking a model.
HFlow `0.2.3` at the exact pinned commit produced two cataloged episodes under schema `1` and
pipeline `9cd5d34f513d`: the smooth episode was accepted and the planted joint discontinuity was
quarantined. `hflow.curate` returned both rows, and both canonical files opened with the stock MCAP
reader.

The portable bridge evaluated five projections: two entered a two-document corpus, while one
unverified model draft, one quarantined projection, and one unsettled-quality projection remained
ledger-only. All five retained the curation-query digest, schema and pipeline versions, check
version `llb_bridge_quality/1`, enrichment version `llb_projection/1`, and exact MCAP/source
coordinates. The committed projection-manifest digest is
`edccf56cdc1a81cb1951307562c19349bf13d9d00a5d2572ade86eee08519f24`; the complete fixture-manifest
digest is `03d3d6c27fc2814e92ed6c567c33b10d67e2a60f30f13970d5a43332357c182b`.

This result is overturned by any pinned fixture byte, upstream revision, schema, pipeline,
curation query, check, or enrichment version changing; by a referenced MCAP window no longer
opening with the standard reader; or by an admission outcome changing under the accepted-ledger
gate. It demonstrates an evidence bridge, not retrieval benefit, physical-device behavior, or a
robotics safety claim.

Focused coverage lives in
[`tests/llb/robotics/test_evidence_bridge.py`](../../../../tests/llb/robotics/test_evidence_bridge.py)
and
[`tests/llb/robotics/test_hflow_manifest.py`](../../../../tests/llb/robotics/test_hflow_manifest.py).
It includes the offline replay, stock-reader check, strict schema, stale-byte and mixed-generation
refusals, exact accepted-ledger requirement, and empty temporal-window refusal.
