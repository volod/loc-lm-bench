# Compact versus cap with active compaction

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

`make bench-agentic-context-compact-long` reuses the medium-observation long-transcript task
builder, raises the step ceiling to 12, and pairs `compact` directly against `observation_cap`.
Unlike the broad four-policy lane, its baseline is the cap policy. The command fails with an
`inactive` verdict when no compact episode records `n_compactions > 0`; an operator can then
lengthen the transcript or tighten `AGENT_CONTEXT_COMPACT_LONG_MAX_PROMPT_CHARS`. The default
16,000-character guard was selected because the deterministic worst-case probe keeps the cap lane
within its window while crossing the shipped `compact_share=0.5` trigger.

The cost comparison uses `total_model_input_tokens`, so compact receives no free summarizer: each
summary prompt is included, locally refused controller prompts are excluded, and `n_model_calls`
adds controller and summarizer calls. Repeated compactions also feed the prior running summary into
the next summary prompt and preserve its machine aggregate headers instead of overwriting earlier
memory. After the first summary, trigger hysteresis lets live work grow to the full prompt guard
before summarizing again; each summary input remains capped at the initial trigger size. Core
locations are `src/llb/bench/context_policy/compact_vs_cap.py`,
`src/llb/bench/context_policy/compact_vs_cap_report.py`,
`src/llb/cli/bench/context/compact_vs_cap.py`, and
`tests/llb/bench/context_policy/test_agentic_compact_vs_cap.py`.

```bash
make bench-agentic-context-compact-long MODEL=<model> BACKEND=<backend> \
  AGENT_CONTEXT_COMPACT_LONG_MAX_MODEL_LEN=8192
```

CUDA host evidence (2026-07-30, RTX 4060 Ti 16 GB):
`MamayLM-Gemma-3-12B-IT-v2.0` on Ollama with `num_ctx=8192`, 14 medium-search tasks,
`max_steps=12`, a 16,000-character prompt guard, 206 calls at 4.3 tok/s. Bundles are under
`.data/agentic-compact-vs-cap/20260730T19462*`.

| policy | completion | mean steps | model calls | total input tok | compactions | overflow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `observation_cap` | 0.929 | 7.29 | 7.29 | 10461 | 0 | 0 |
| `compact` | 0.929 | 7.29 | 7.43 | 10553 | 2 | 0 |

Compaction was active in 2 of 14 episodes. Both policies completed the same 13 tasks and failed the
same `medium-search-count-008` task at the step ceiling. The two compacted cases completed under
both policies in the same 10 controller steps; compact added one summary call to each. Paired
`compact - observation_cap` deltas were completion +0.000 [+0.000, +0.000], total model-input
tokens +92 [+0, +232.5], and model calls +0.143 [+0, +0.357].

Verdict: **still tied**. Once the active summarizer cost is counted, compact does not improve
completion and its extra input/call cost does not separate clear of zero at this task count.
Use `observation_cap` for this medium-search shape because it reaches the same observed outcome
without the extra mechanism; retain compact as an unproven option for transcripts whose old state,
rather than repeated search, must survive the trigger.

## Memory-dependent transcript

`make bench-agentic-context-compact-memory` builds an eight-task deterministic tool world in which
the first one-way workflow observation carries a typed `[memory: final_code=...]` fact. Seven later
observations expose unique next-step tokens, while progress lives in the world cursor rather than
the transcript. A stale token cannot advance or replay the code, and objective checks require the
cursor to finish, the answer to contain the early code, and the code not to have been copied into
the file or DB world. Both policies see the same tasks and shared digest.

Compact folds typed memory markers into the running summary independently of free-text summary
quality, as aggregate-safe search headers already do for counts. Only when the recent live
observation says the workflow is complete does the prompt expose a finish cue with the remembered
code. The focused runner records a predeclared minimum activation rate and returns `inactive` when
too few compact episodes cross the trigger.

```bash
make bench-agentic-context-compact-memory MODEL=<model> BACKEND=<backend>
```

Core locations are `src/llb/bench/memory/transcript.py` (task builder),
`src/llb/bench/tool_world.py` (one-way token workflow),
`src/llb/bench/agentic/context.py` (typed memory folding and finish cue), the focused compact-vs-cap
runner and report modules, `make/eval/categories-platform.mk`, and focused tests in
`tests/llb/bench/memory/test_agentic_memory_transcript.py` and
`tests/llb/bench/context_policy/test_agentic_compact_vs_cap.py`.

CUDA host evidence (2026-08-02, RTX 4060 Ti 16 GB): `qwen3:14b` on Ollama, eight depth-8 tasks,
`max_steps=14`, 8,000-character prompt guard, 136 model calls at 19.9 tok/s. The predeclared
activation floor was 75% (6/8); compact activated in 8/8 episodes with 16 compactions. Bundles are
under `.data/agentic-compact-vs-cap/20260802T11390*`; the summary manifest is
`.data/agentic-compact-vs-cap/20260802T113901.203273Z-2bf71a093172/manifest.json`.

| policy | completion | mean steps | mean model calls | mean input tok | compactions | overflow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `observation_cap` | 0.000 | 6.00 | 6.00 | 10466.0 | 0 | 8 |
| `compact` | **1.000** | 9.00 | 11.00 | 17570.6 | 16 | 0 |

Paired `compact - observation_cap` completion was +1.000 [+1.000, +1.000] with 8/0/0
wins/losses/ties (exact randomization p=0.00390625). Total model-input cost, including summary
prompts, increased by +7104.625 [+7099.000, +7112.375] tokens per task, and model calls increased
by +5.000 [+5.000, +5.000]. Verdict: **prefer compact for this memory-dependent shape** because
completion separates after the activation gate, accepting the measured summary cost. This result
does not change the shipped default: the medium-search lane still prefers cap, so policy choice is
task-shape dependent.

The model-control pilot first tried the UA-specialized 12B MamayLM used by the earlier context
lanes. It repeatedly issued a stale file or workflow call and could not pass the task-control path,
so it was not used for the policy verdict. `qwen3:14b` passed that control and fits the 16 GB host;
the installed 27B/31B options do not leave safe context headroom.

## Non-Qwen depth/trigger transfer

`make bench-agentic-context-compact-memory-transfer` evaluates a committed sequential non-Qwen
candidate roster against a memory-free token-chain control before it exposes any model to the
policy matrix. The control puts its answer only in the final observation, so it checks basic token
progression without requiring old state. A candidate must complete at least 3/4 depth-10 controls;
failed pilots retain per-episode statuses, answers, and call sequences in the aggregate artifact.
The first eligible candidate alone advances to the six-pair cells at `depth=6/10` and
`compact_share=0.4/0.6`, bracketing the Qwen reference geometry while holding the 8,000-character
guard, observation cap, padding, tasks, and success contract fixed.

The prospective design is
`samples/benchmarks/agentic_compact_memory_transfer_design.json`. Orchestration and analysis live
in `src/llb/bench/memory/transfer/run.py`, aggregate rendering/persistence in
`src/llb/bench/memory/transfer/report.py`, and the CLI in
`src/llb/cli/bench/memory/transfer.py`. Focused contracts are in
`tests/llb/bench/memory/test_agentic_memory_transfer.py`; token-chain task construction shares
`src/llb/bench/memory/transcript.py` with the memory-dependent lane.

```bash
make bench-agentic-context-compact-memory-transfer
```

CUDA host evidence (2026-08-02, RTX 4060 Ti 16 GB): the 11.8B Lapa/Gemma candidate failed the
control at 0/4 after exhausting the step budget; the installed Q4 `gemma4:e4b` candidate passed
4/4 at 38.27 tok/s and became the selected non-Qwen family. In every matrix cell, cap completed
0/6 and compact completed 6/6. All compact episodes crossed the predeclared 75% activation floor;
depth 6 produced six compactions per cell and depth 10 produced twelve. The audit-complete
aggregate is
`$DATA_DIR/agentic-compact-memory-transfer/20260802T120539.133308Z-146807806b61/manifest.json`.

| depth | compact share | cap completion | compact completion | paired d(completion) | cap input tok | compact input tok | paired d(input tok) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.40 | 0.000 | **1.000** | +1.000 [+1.000, +1.000] | 10466.0 | 12763.0 | +2297.0 [+2289.0, +2305.0] |
| 6 | 0.60 | 0.000 | **1.000** | +1.000 [+1.000, +1.000] | 10466.0 | 12482.3 | +2016.3 [+1988.0, +2041.5] |
| 10 | 0.40 | 0.000 | **1.000** | +1.000 [+1.000, +1.000] | 10466.0 | 21570.2 | +11104.2 [+11056.8, +11179.3] |
| 10 | 0.60 | 0.000 | **1.000** | +1.000 [+1.000, +1.000] | 10466.0 | 22072.8 | +11606.8 [+11572.2, +11640.7] |

Every completion comparison has 6/0/0 wins/losses/ties and exact randomization p=0.015625.
Verdict: **portable across the tested matrix**. The memory-dependent compact completion gain is
not Qwen-only and survives both tested depths and trigger shares on an eligible Gemma 4 family,
while the total model-input premium grows sharply with workflow depth. This evidence does not
change the shipped default: medium-search tasks still favor observation cap, and the six-pair
rows become borderline one convention tighter at 97.5% confidence.

## Second-family replication and cap-fitting boundary

`make bench-agentic-context-compact-memory-replication` extends the transfer contract without
changing its task, control threshold, activation floor, paired metrics, or policy defaults. Its
committed design, `samples/benchmarks/agentic_compact_memory_replication_design.json`, excludes
the Qwen and Gemma 4 reference families, evaluates candidates in host-fit order, raises every
transfer cell from six to seven tasks, and requires all four completion rows to remain separated
at the 97.5% stability reading. A fifth predeclared cell raises the prompt guard from 8,000 to
12,000 characters at depth 6; the replication is invalid unless cap has zero context overflows and
compact still activates in at least 6/7 episodes.

```bash
make bench-agentic-context-compact-memory-replication
```

Shared cell execution and its self-contained row schema live in
`src/llb/bench/memory/transfer/cells.py`; the original transfer runner now reuses that
module. Replication validation, execution, and analysis live in
`src/llb/bench/memory/replication/run.py`, rendering and persistence in
`src/llb/bench/memory/replication/report.py`, the CLI in
`src/llb/cli/bench/memory/replication.py`, and focused contracts in
`tests/llb/bench/memory/test_agentic_memory_replication.py`. The boundary analyzer records a
direction-aware lower-is-better cost gate: it uses the original compact-minus-cap delta and the
two-sided exact sign test rather than misreading the shared positive-tail randomization field.

CUDA host evidence (2026-08-02, RTX 4060 Ti 16 GB): `aya-expanse:8b` failed the unchanged control
at 0/4. `mistral-small3.1:24b` passed 4/4 and became the second eligible non-Qwen family. It ran at
12.49 tok/s with a served 4,096-token window; live host telemetry showed 89% GPU / 11% CPU layer
placement, 14,730 MiB VRAM resident, and high GPU utilization rather than swap-thrashing. The
audit-complete, direction-corrected aggregate is
`$DATA_DIR/agentic-compact-memory-transfer-replication/20260802T131252.023379Z-488fd4867be7/manifest.json`.

| cell | guard | cap completion | compact completion | paired d(completion) | cap input tok | compact input tok | paired d(input tok) | compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| depth 6, share 0.4 | 8000 | 0.000 | **1.000** | +1.000 [+1.000, +1.000] | 10466.0 | 12771.3 | +2305.3 [+2106.1, +2662.7] | 7 |
| depth 6, share 0.6 | 8000 | 0.000 | **1.000** | +1.000 [+1.000, +1.000] | 10466.0 | 12375.4 | +1909.4 [+1872.9, +1942.4] | 7 |
| depth 10, share 0.4 | 8000 | 0.000 | **1.000** | +1.000 [+1.000, +1.000] | 10466.0 | 21455.1 | +10989.1 [+10761.6, +11413.7] | 14 |
| depth 10, share 0.6 | 8000 | 0.000 | **1.000** | +1.000 [+1.000, +1.000] | 10466.0 | 21868.1 | +11402.1 [+11332.1, +11466.1] | 14 |
| cap fits: depth 6, share 0.5 | 12000 | **1.000** | **1.000** | +0.000 [+0.000, +0.000] | 13258.0 | **12441.6** | **-816.4 [-827.9, -807.6]** | 7 |

Each overflow cell has 7/0/0 completion wins/losses/ties, exact randomization p=0.0078125, and a
non-borderline `separated` reading at 97.5%. Compact activated in all 7 episodes, while cap
overflowed in all 7. In the boundary, both policies completed 7/7 with zero overflow; compact
activated 7/7, added one model call per task, but saved 816.4 total input tokens because the
summary reduced repeated controller-prompt input. All seven cost pairs favor compact and the
two-sided exact sign-test p is 0.015625.

Verdict: **replicated including the cap-fitting boundary**. The memory-dependent completion gain
survives a second non-Qwen family at the tighter convention, and it is not solely an overflow
artifact: at the tested usable boundary, compact ties completion and more than repays its summary
input through smaller later prompts. The shipped default remains task-shape dependent and is not
changed by this focused memory transcript.
