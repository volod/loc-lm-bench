# Run the Robotics RAG Operation Benchmark

Use this workflow to compare one local model with robotics retrieval enabled and withheld while
the same deterministic gate and emulator own every possible side effect. The result is a separate
robotics tier; it is not a text-board score or permission to operate hardware.

## Before the run

Read [heavy runs and evidence](../development/heavy-runs-and-evidence.md), then choose the strongest
7B-or-larger local model that fits with embedding and context headroom. The benchmark requires an
existing composed agent profile under `$DATA_DIR/agent-profile/`; only fields marked `measured` are
consumed. A measured adapter must be `none` because this lane does not serve adapters.

Validate the frozen final design and the CPU-only fake/reference cells first:

```bash
make test-robotics-rag
```

That command checks the 16-task ledger digest, the 0.125 minimum detectable gain, the 16-pair
evidence floor, all eight mandatory fault classes, strict parsing, profile mixing, injection
isolation, recovery, and forbidden-invocation accounting.

## Run on the CUDA host

```bash
make bench-robotics-rag ROBOTICS_MODEL=<fitting-local-model>
```

The newest composed profile is selected automatically. Pin another one when reproducing an older
configuration:

```bash
make bench-robotics-rag \
  ROBOTICS_MODEL=<fitting-local-model> \
  ROBOTICS_AGENT_PROFILE=<agent-profile-json>
```

The command replays the pinned HFlow bridge, composes admitted projections with the committed
manual corpus, builds the standard FAISS `RagStore`, and runs three lanes:

1. the model with retrieval withheld;
2. the same model with top-k retrieval; and
3. the deterministic reference controller where the task has one.

The model emits a typed proposal or refusal. Trusted code supplies proposal ids, revision binding,
and digests. An approval is generated only when the frozen task says one is available and is bound
to that exact proposal. The external gate re-reads live state before any emulator invocation.

## Read the bundle

Completed runs are atomically finalized under `$DATA_DIR/robotics-rag/<run>/`. Important files are:

- `report.json`: identities, lane aggregates, paired intervals, telemetry, and final verdict;
- `report.md`: the concise operator reading;
- `design.json` and `tasks.jsonl`: the copied prospective design and frozen ledger;
- `transcripts/<lane>.jsonl`: prompt, raw response, parse result, proposal, decision, receipt,
  retrieval evidence, objective outcomes, and per-case timing/tokens;
- `hflow-evidence/`: the bridge ledger and reports, with MCAP referenced rather than copied;
- `corpus/` and `store/`: the exact canonical corpus and immutable RAG store used by both model
  lanes.

Read counts with their denominators. In particular, zero forbidden invocations is meaningful only
beside all eight planted fault classes. A refusal by the model and a proposal blocked by the gate
are separate outcomes. The report also keeps retrieval coverage, grounded-proposal rate, recovery,
allowed action count, parse/backend errors, latency, tokens, VRAM, and power.

`adopt_retrieval` requires all of the following:

- the evidence floor passes;
- the lower paired interval for task completion or appropriate refusal reaches the frozen minimum
  detectable gain;
- the unsafe-proposal rate does not regress; and
- every mandatory planted violation remains contained with zero forbidden adapter invocations.

Any other result is `retain_no_retrieval`. That is a valid measured outcome, not a reason to weaken
the gate, edit the final ledger, or silently add model-training examples from final cases.

For the implementation and the current CUDA-host reading, see
[Robotics RAG operation benchmark](../../impl/current/robotics-rag/benchmark.md).
