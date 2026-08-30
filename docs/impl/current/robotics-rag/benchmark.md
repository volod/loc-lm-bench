# Robotics RAG Operation Benchmark

The robotics benchmark is a separate, protocol-neutral tier that asks one operational question:
does retrieval improve safe task completion or appropriate refusal for a local model when every
proposed side effect still passes through the external action gate? It never contributes a row to
the text-answer board and never grants hardware authority.

## Implementation

The prospective design in
[`samples/robotics/benchmark/`](../../../../samples/robotics/benchmark/) freezes 16 final-split
tasks, a 0.125 minimum detectable gain, a 16-pair evidence floor, 95% paired bootstrap intervals,
10,000 resamples, and seed 20260830. Eight tasks expect completion or idempotent recovery, including
the retrieved-injection case. Eight expect refusal: seven other mandatory fault classes plus an
unreachable device case. The task-ledger digest is part of the design, so editing a final case
fails before a model call.

The implementation is split along the benchmark phases:

- [`design.py`](../../../../src/llb/robotics/benchmark/design.py) validates the frozen ledger,
  evidence floor, unique task identities, and complete mandatory-fault vocabulary.
- [`profile.py`](../../../../src/llb/robotics/benchmark/profile.py) imports only fields whose
  composed agent-profile state is `measured`. A measured model or backend disagreement and any
  measured non-null adapter fail closed; demoted, refused, and unmeasured values are recorded but
  never used as operating recommendations.
- [`retrieval.py`](../../../../src/llb/robotics/benchmark/retrieval.py) replays the pinned HFlow
  bridge, composes its accepted projections with the committed operation manuals, ingests one
  canonical corpus, and builds the existing `RagStore`. The adversarial document is admitted only
  to the declared injection case rather than contaminating unrelated tasks.
- [`parser.py`](../../../../src/llb/robotics/benchmark/parser.py) accepts one strict JSON decision.
  Trusted code supplies proposal identity, expected revision, and digest; model text cannot mint
  those fields. Proposal preconditions must copy the signed policy statements exactly.
- [`task_runtime.py`](../../../../src/llb/robotics/benchmark/task_runtime.py) prepares a fresh
  emulator and runs proposals through `ActionExecutor`, binds approvals to the exact proposal
  digest, replays the recoverable idempotent failure, and seeds the ambiguous non-idempotent
  no-retry state. [`execution.py`](../../../../src/llb/robotics/benchmark/execution.py) owns the
  model/reference decision and objective case scoring above that runtime.
- [`metrics.py`](../../../../src/llb/robotics/benchmark/metrics.py) reports objective counts and
  denominators, paired intervals, unsafe proposals, block reasons, recovery, actions, reliability,
  tokens, latency, and the predeclared adopt-or-retain verdict.
- [`run.py`](../../../../src/llb/robotics/benchmark/run.py) checks all fingerprints, runs the same
  model first without retrieval and then with retrieval, evaluates the deterministic reference
  controller, samples VRAM and power, and atomically finalizes the operation bundle.

Every bundle pins the design, task ledger, measured profile, corpus, store, model/backend, adapter,
driver references, and action policy. It retains JSONL request/response, parsed proposal,
gate-decision, receipt, retrieved-context, and recovery transcripts beside machine- and
human-readable reports. HFlow MCAP media remains referenced by digest and is not copied.

The CLI adapters are in
[`robotics_benchmark.py`](../../../../src/llb/cli/robotics_benchmark.py). The standard workflows
are:

```bash
make test-robotics-rag
make bench-robotics-rag ROBOTICS_MODEL=<fitting-local-model>
```

Focused fake, reference-controller, parser, profile-mixing, retrieval-isolation, digest, and
mandatory-fault coverage lives in
[`tests/llb/robotics/benchmark/`](../../../../tests/llb/robotics/benchmark/).

## CUDA-host result

On 2026-08-30, `make bench-robotics-rag` ran the 16 frozen final tasks on the RTX 4060 Ti 16 GB
CUDA host with driver 595.84. The local backend was Ollama and the model was
`hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M`, selected as the strongest
UA-capable 12B-class model that left RAG headroom. Ollama reported all 8.22 GB of model weights GPU
resident. The measured profile supplied adapter `none`, an 8192-token context budget,
`observation_cap`, and the loop policy; its demoted model/backend/store fields were not consumed.

The corpus combined two admitted text projections from a five-projection, two-episode pinned HFlow
manifest with four committed manuals. The store used `intfloat/multilingual-e5-base`, recursive
400-character chunks with 80-character overlap, flat FAISS retrieval, and top k = 3. Each model
lane used temperature zero, seed 20260830, and at most 512 completion tokens.

Both MamayLM lanes completed 8/8 completion tasks, made 8/8 appropriate refusals, and therefore
reached 16/16 operational successes. The deterministic controller matched the same 16/16 result.
The retrieval-minus-no-retrieval task-completion delta was 0.000 [0.000, 0.000] with 0 wins, 0
losses, and 8 ties; appropriate refusal was also 0.000 [0.000, 0.000] with 0/0/8; combined
operational success was 0.000 [0.000, 0.000] with 0/0/16. The 16-pair evidence floor passed, but
neither interval cleared the predeclared 0.125 gain.

Retrieval did change grounding: expected evidence was retrieved on 16/16 tasks and all 8/8 action
proposals cited only retrieved evidence, versus 0/8 grounded proposals when retrieval was withheld.
That improvement did not change an operational decision because the no-retrieval lane was already
at the ceiling. Both lanes made 0/8 unsafe proposals, contained all 8/8 mandatory fault classes,
recorded zero forbidden adapter invocations, and recovered the one planted idempotent write failure.
Each lane made nine allowed adapter invocations: eight successful workflows plus the failed first
attempt in the recovery case.

Retrieval increased the mean per-case generation latency from 5.794 s to 6.248 s and total lane
time from 92.713 s to 100.391 s. Prompt tokens increased from 22,019 to 29,111 and completion tokens
from 1,839 to 1,976. Mean sampled total GPU power was 120.15 W without retrieval and 125.44 W with
retrieval; sampled peaks were 149.23 W and 160.89 W, over 177 and 192 readings respectively. Peak
observed total VRAM was 10,615 MB and 10,643 MB. These cost readings are descriptive because the
fixed lane order ran no-retrieval first and retrieval second; they are not an independently
randomized performance comparison.

The verdict is therefore `retain_no_retrieval`: retrieval improved evidence citation but bought no
completion or refusal gain and added context cost. This negative result does not remove the RAG
lane; it prevents its adoption as the operating default for this model and ledger.

An initial local shakeout failed closed because otherwise correct model proposals abbreviated the
signed policy preconditions. The final implementation makes the prompt require verbatim policy
statements and confines the planted injection document to its declared task. The focused suite
locks both corrections before the result above was measured.

The result would be overturned by a frozen task set on which retrieval creates completion or
appropriate-refusal wins whose paired lower interval reaches 0.125 without an unsafe-proposal
regression and while all mandatory gate cases remain contained. A harder ledger that removes the
current no-retrieval ceiling, another model family, changed grounding labels, changed driver or
policy contracts, or a physical device can all change the reading and require a new bundle. This
emulator result does not claim MHS compatibility, physical transfer, or safety certification.
