# Prepare HFlow Robotics Evidence

Use this workflow to project approved text from landed robot episodes into the normal loc-lm-bench
corpus. It is an offline data-preparation boundary: it does not operate hardware, ingest raw MCAP
bytes into a prompt, run HFlow's scheduler, or decide the quality thresholds owned by the HFlow
pipeline.

## Replay the committed fixture

Install the development environment normally, then run:

```bash
make robotics-evidence-replay
```

This command is network-free. It validates every file pinned by
`samples/robotics/hflow/fixture-manifest.json`, opens each referenced episode with the stock MCAP
reader, applies the admission gates, and writes a run under
`$DATA_DIR/robotics-evidence/<run>/`.

Read these outputs:

- `report.json` for the verdict, admission counts, generation, manifest digest, and corpus
  fingerprint;
- `evidence-ledger.jsonl` for every admitted or excluded projection and its exact temporal/source
  evidence; and
- `corpus/` for only the admitted documents in the same format as ordinary text ingestion.

A projection with `admission` set to `draft`, `quarantined`, or `unverified` must have no
`source_span` and must not occur under `corpus/`. Do not copy ledger-only text into the corpus to
work around the gate.

## Exercise the pinned HFlow producer

When network access is available, run:

```bash
make robotics-evidence-fixture
```

The target creates an isolated environment with HFlow `v0.2.3` at commit
`d2e0f3700f2267cfeb0db1957743bb9f5f41256b`. It runs `app.test(record=True)` for clean and planted
defect episodes, runs `hflow.curate`, opens the resulting canonicals with the standard MCAP reader,
and replays a portable projection fixture. It writes its integration report alongside the bridge
artifacts under `$DATA_DIR/robotics-evidence/<run>/`; it does not modify the committed fixture.

Use this command after intentionally changing the HFlow pin, pipeline declaration, projection
schema, or fixture generator. An ordinary bridge consumer should use the network-free replay.

## Supply a projection manifest

The bridge accepts a portable fixture directory containing `fixture-manifest.json` and its pinned
`manifest.parquet`. Every projection row must name one generation consistently: HFlow release and
revision, HFlow schema, pipeline, curation-query digest, and the complete check and enrichment
producer-version sets. It must also provide:

- content-addressed episode id, MCAP URI and digest;
- channels and a non-empty half-open nanosecond interval containing a message;
- quality state and any quarantine tags; and
- projection author, verification state, language, file digest, and exact character offsets.

Model-authored text marked verified must reference an accepted ledger whose matching item and
corpus copy bind the exact projection id, file, offsets, and text. Otherwise the bridge refuses the
manifest. Human-authored text also requires an explicit verified state, but its review workflow is
owned by the producer. The bridge preserves producer decisions; it does not infer verification
from prose.

For the implementation contract, failure modes, and recorded evidence, read the
[HFlow robotics evidence bridge](../../impl/current/robotics-rag/evidence-bridge.md).
