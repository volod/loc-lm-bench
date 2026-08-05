# Extended Workflows

Extended workflows cover comparison axes that sit beside the main RAG leaderboard: agentic
harnesses, agent context-management policies, judge diagnostics, and prompt-system packages.

## Agentic Harness Comparison

The agentic benchmark can run the same task set through multiple harnesses while keeping the model,
tools, world state, objective checks, optional judge, and context-management policy fixed.

Core locations:

- `src/llb/bench/agentic/model.py`: `Harness` protocol (now carries `policy` + `budget`) and harness
  names;
- `src/llb/bench/agentic/run.py`: runner integration;
- `src/llb/bench/harness/base.py`: pure loop harness;
- `src/llb/bench/harness/langgraph.py`: LangGraph agent/tool graph (same `step_prompt` /
  `ContextState` seam as the loop);
- `src/llb/bench/harness/crewai.py`: CrewAI adapter (accepts the kwargs, does not apply them);
- `src/llb/board/harnesses.py`: one-model harness comparison board rows + prompt-size appendix.

```bash
llb bench-agentic --harness loop --model <model> --backend <backend> \
  --context-policy full
llb bench-agentic --harness langgraph --model <model> --backend <backend> \
  --context-policy observation_cap
llb bench-agentic --harness crewai --model <model> --backend <backend>
llb bench-agentic-compare --model <model> --context-policy observation_cap
make agentic-harness-compare MODEL=<model> BACKEND=<backend> \
  AGENTIC_CONTEXT_POLICY=full AGENTIC_HARNESSES='loop langgraph'
```

### Protocol decision: policies transfer on the harness seam

The `Harness` protocol takes optional `policy` and `budget` alongside `(task, complete, catalog,
max_steps)`. Product rule:

- `loop` and `langgraph` APPLY the requested policy to every step prompt (shared
  `step_prompt` / `ContextState` / overflow guard), so a measured win from
  `bench-agentic-context` is one the operator's LangGraph cell can actually run;
- `crewai` ACCEPTS the kwargs for protocol parity but does NOT apply them -- CrewAI owns its
  ReAct transcript. Episodes stamp `context_policy_supported=false`, and the comparison labels
  the policy cell as `full*` / `observation_cap*` rather than silently reporting our `full`
  assembly. Prompt sizes the framework actually sent still ride on episode telemetry.

Every harness records per-step prompt sizes on `Episode.telemetry`. Bundles persist
`context_policy`, `context_policy_supported`, and `mean_max_prompt_tokens`. The harness comparison
keeps the best run per `(model, harness, context_policy)` and ranks harnesses under ONE fixed
policy (explicit `--context-policy`, else the policy with the most harness coverage), so the axis
never silently mixes framework and context management. The ranked board stays completion-only; an
appendix table reports `prompt-tok`, the requested policy, and whether it was applied.

### CUDA host evidence (2026-07-29)

`MamayLM-Gemma-3-12B-IT-v2.0` on Ollama (`--max-model-len 8192`), 4-task UA seed,
`AGENTIC_HARNESSES='loop langgraph'` (CrewAI extra not installed on this host; fake-crew CI covers
its unsupported path):

| policy | harness | completion | mean steps | mean max prompt tok | applied |
| --- | --- | --- | --- | --- | --- |
| `full` | loop | 0.250 | 6.00 | 906.2 | yes |
| `full` | langgraph | 0.250 | 6.00 | 906.2 | yes |
| `observation_cap` | loop | 0.250 | 6.00 | 906.2 | yes |
| `observation_cap` | langgraph | 0.250 | 6.00 | 906.2 | yes |

Loop and LangGraph matched item-for-item on completion and prompt tokens under both policies
(seed-task observations sit under the 800-char cap, so `observation_cap` is a no-op on this set --
the transfer seam is what the run proves). Bundles under `.data/agentic/20260729T12*`.

### Agent Loop-Policy Recommendation

`bench-agentic-loop` measures the controller policy separately from the framework axis. It runs a
Cartesian grid over the step budget, malformed-call handling, and repeated-identical-call handling
while holding the model, task order, tool world, context policy, and objective checks fixed. The
exact legacy cell (`max_steps=6`, `answer`, `allow`) is mandatory and every cell carries paired
deltas against it.

The policy vocabulary is:

- `answer`: preserve legacy behavior by treating an unreadable structured response as the final
  answer;
- `repair_once`: send one additional bounded prompt containing the tool schemas, rejected output,
  and parse error, then continue strictly if that repair is still malformed;
- `strict`: count the malformed model step, execute nothing, place a parse-error note in the live
  transcript, and continue;
- `allow`: execute consecutive identical calls as before;
- `noop`: record the repeated call and return an explicit controller note without executing its
  world mutation again.

`parse_tool_call_detailed` in `src/llb/scoring/tool_calls.py` preserves whether a response was a
structured-call attempt and its parse error while the existing `parse_tool_call` API remains
unchanged. `src/llb/bench/agentic/loop_policy.py` owns the policy constants and repair prompt;
`src/llb/bench/agentic/episode.py` applies them; generic batch execution/scoring is split into
`src/llb/bench/agentic/batch.py`. The grid, persistence, paired report, and CLI live in
`src/llb/bench/agentic_loop_policy.py`, `agentic_loop_policy_persist.py`,
`agentic_loop_policy_report.py`, and `src/llb/cli/bench/category_agentic_loop_policy.py`.

Every case row reports objective completion, malformed-call count and rate, logical steps, tool and
model calls, repair count, repeated no-ops, total model-input tokens, and wall-clock seconds.
Completion, malformed rate, steps, tool calls, input tokens, and wall clock are all paired against
the baseline over shared bootstrap index sets. A default changes only for a positive completion
delta under the repository's standard paired verdict; an underpowered point gain is recorded but
does not become a shipped recommendation.

```bash
make bench-agentic-loop MODEL=<model> BACKEND=<backend> \
  AGENT_MAX_STEPS=4,6,10 \
  AGENT_MALFORMED_POLICY=answer,repair_once,strict \
  AGENT_REPEATED_CALL_POLICY=allow,noop
```

Each cell persists under `$DATA_DIR/agentic-loop-policy/<run>/`. Its manifest contains the complete
grid, task digest, per-cell means, all paired blocks, context-budget provenance, and the per-model
recommendation. `comparison.md` and `recommendation.json` sit beside every manifest so an operator
can inspect or consume the decision without reconstructing the grid.

CUDA-host evidence (2026-07-31): MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M ran through Ollama on the RTX
4060 Ti 16 GB with `num_ctx=8192` (about 8.3 GiB model VRAM). The committed four-task UA seed
covered all 18 cells in 480 model calls / 15,144 generated tokens at 5.4 tok/s:

| max steps | repeat | completion | malformed rate | mean input tokens | mean wall seconds |
| --- | --- | --- | --- | --- | --- |
| 4 | `allow` | 0.250 | 0.000 | 3338.5 | 24.86-28.71 |
| 4 | `noop` | 0.500 | 0.000 | 3392.2 | 25.61-29.74 |
| 6 | `allow` | 0.250 | 0.000 | 5131.5 | 32.39-34.70 |
| 6 | `noop` | 0.500 | 0.000 | 5299.5 | 33.24-35.00 |
| 10 | `allow` | 0.250 | 0.000 | 8960.0 | 54.49-55.22 |
| 10 | `noop` | 0.500 | 0.000 | 9592.2 | 56.68-57.53 |

The malformed policy was inactive on this model/task set: all three settings reproduced the same
row inside each `(max_steps, repeat)` block. `noop` improved the point estimate by `+0.250`, but its
paired 95% interval was `[0.000, 0.750]` with only one discordant task, so the standard reading is
flat. Raising the budget to 10 added about 3828.5 input tokens per episode under `allow` without a
completion gain. The evidence-backed recommendation therefore retains `6/answer/allow` at 0.250
completion, 5131.5 mean input tokens, and 34.70 mean wall seconds; shipped defaults did not change.
The 18 bundles are under `.data/agentic-loop-policy/20260731T061934*` through
`.data/agentic-loop-policy/20260731T061938*`.

CI coverage in `tests/llb/bench/test_agentic_loop_policy.py` drives all malformed branches over the
fake completion seam, including an unreadable JSON call repaired into a successful tool call,
strict continuation, repeated no-op behavior, mandatory baseline validation, paired metrics,
recommendation gating, and per-cell comparison artifacts. The explicit default policy is also
checked against the implicit legacy loop behavior.

#### Powered Repeat-Noop Comparison

The focused power lane adds a prospective study contract without changing the general loop-policy
grid. `samples/benchmarks/agentic_loop_repeat_power_design.json` declares the model and task-family
coverage, `+0.25` minimum detectable completion gain, 80% target power at alpha 0.05, 32 planned
independent tasks, the six-discordant-pair evidence floor, 50% minimum activation, and maximum
relative cost increases of 10% for model-input tokens and 20% for wall time. The prior four-task
seed supplies the paired reference SD of 0.5; the normal approximation requires all 32 planned
tasks, while a `+0.25` gain can supply eight discordant pairs.

`samples/benchmarks/agentic_loop_repeat_power_uk.json` contains eight non-duplicate tasks in each of
four deterministic families: repeated file reads, corpus searches, calculator calls, and state
mutations. Every task carries its family in the task digest and per-case row. The episode telemetry
now counts `n_repeated_calls` under both `allow` and `noop`; `n_repeated_noops` remains the separate
suppression count. Reports show activation rate and mean repeats beside completion, while every
cell manifest carries family counts, the prospective contract, completion and paired cost gates,
and `power-analysis.json`.

`src/llb/bench/agentic_loop_policy_power.py` validates the declaration before model inference and
resolves its gates afterward. Its per-family support flag requires coverage, repeat activation, a
positive separated completion delta at least as large as the declared gain, and upper paired cost
bounds within both ceilings. A single family cannot set `changes_shipped_defaults`; that remains
false until the full predeclared roster supports the change. The CLI accepts
`--repeat-power-design` and `--model-family`. `make bench-agentic-loop-repeat-power` runs the fixed
`6/answer/allow,noop` comparison over the predeclared local Gemma and Qwen families:

```bash
make bench-agentic-loop-repeat-power
```

CUDA-host evidence (2026-07-31), RTX 4060 Ti 16 GB, identical task digest
`eb05651bf9b50f8cefd85c1dfcf2ba60f0d1a51ab031230f0fd861fceb5227cc`:

| model family | policy | completion | activation | mean repeats | prompt tokens | wall seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MamayLM-Gemma 12B | `allow` | 0.156 | 1.000 | 4.94 | 5180.7 | 33.46 |
| MamayLM-Gemma 12B | `noop` | 0.156 | 1.000 | 4.94 | 5404.0 | 34.92 |
| Qwen3 14B | `allow` | 0.562 | 1.000 | 2.75 | 3657.8 | 4.63 |
| Qwen3 14B | `noop` | 0.562 | 1.000 | 2.75 | 3780.1 | 4.39 |

MamayLM tied on all 32 completion pairs (`+0.000 [0.000, 0.000]`); its paired prompt increase was
223.3 tokens with interval `[205.3, 238.4]`, inside the declared ceiling. Qwen had two wins, two
losses, and 28 ties, for a net completion delta of `+0.000 [-0.125, 0.125]`. Its prompt delta was
122.3 tokens with interval `[-216.4, 459.5]`, whose upper bound exceeds the 365.8-token ceiling;
wall time stayed inside its ceiling for both families. Thus activation and coverage pass, but the
material completion gate fails for both families and Qwen also fails the prompt-cost gate.
`allow` remains the repeated-call default.

The four cell bundles are:

- MamayLM `allow`:
  `.data/agentic-loop-policy/20260731T094604.022763Z-bd8879dc435f/`
- MamayLM `noop`:
  `.data/agentic-loop-policy/20260731T094605.300042Z-7654dc4ad227/`
- Qwen `allow`: `.data/agentic-loop-policy/20260731T095054.804103Z-86b26eada0a8/`
- Qwen `noop`: `.data/agentic-loop-policy/20260731T095055.716092Z-06242ba52f0a/`

`tests/llb/bench/test_agentic_loop_policy_power.py` checks declaration validation, duplicate refusal,
family coverage, activation/completion separation, recommendation gating, persistence, and the
committed 32-task fixture. The original loop-policy suite also checks that repeated calls are
counted without suppression under `allow`. Validation on 2026-07-31: `make ci` passed 2,460 tests
with 45 opt-in/slow tests deselected, and `make lint-md` passed.

##### Localized Repeat Feedback Comparison

The repeat-feedback lane keeps that powered ledger, model roster, and fixed
`6/answer/allow,noop` policy while adding only the controller notice as an experimental axis.
`samples/benchmarks/agentic_loop_feedback_localization_design.json` predeclares `current`, `uk`,
and `bilingual` variants with the same `+0.25` completion target, activation requirements, and
paired token/wall cost ceilings. `make bench-agentic-loop-repeat-feedback` runs all four cells: one
`allow/current` reference plus three `noop` feedback cells.

`src/llb/bench/agentic/loop_policy.py` owns the validated feedback variants, while
`src/llb/bench/agentic/episode.py` records whether a suppressed repeat is followed by a changed
tool call or final answer. `src/llb/bench/agentic_loop_feedback.py` reports that response rate
overall and per task family, pairs each localized cell directly against `noop/current`, and admits
a family-level recommendation only when activation, material separated completion, prompt-token,
and wall-time gates all pass. Each cell id and manifest includes the feedback variant;
`feedback-study-design.json` and `feedback-analysis.json` make the prospective contract and
decision independently inspectable. The general recommendation remains isolated from these
experimental `noop` cells, so a one-family result cannot alter shipped defaults.

CUDA-host evidence (2026-07-31), RTX 4060 Ti 16 GB, identical 32-task digest:

| model family | feedback | response | completion | prompt tokens | wall seconds | support |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| MamayLM-Gemma 12B | `current` | 0.000 | 0.156 | 5404.0 | 35.13 | reference |
| MamayLM-Gemma 12B | `uk` | 0.000 | 0.156 | 5371.3 | 33.92 | no |
| MamayLM-Gemma 12B | `bilingual` | 0.062 | 0.219 | 5322.3 | 35.20 | no |
| Qwen3 14B | `current` | 0.562 | 0.562 | 3780.1 | 4.43 | reference |
| Qwen3 14B | `uk` | 0.281 | 0.312 | 4831.1 | 5.47 | no |
| Qwen3 14B | `bilingual` | 0.875 | 0.875 | 2906.6 | 3.78 | yes |

For MamayLM, bilingual feedback redirected only two search-family episodes and produced a
`+0.062 [0.000, 0.156]` completion delta, below the target and statistically flat. For Qwen,
bilingual feedback redirected 28 of 32 activated episodes, including five of eight calculator
tasks, and produced a separated `+0.312 [0.156, 0.469]` completion delta with ten wins and no
losses. Its paired prompt delta was `-873.5 [-1324.4, -424.4]` tokens and wall delta was
`-0.64 [-1.13, -0.16]` seconds, so both cost gates passed. Ukrainian-only feedback regressed Qwen
completion and failed both cost gates. The evidence therefore recommends bilingual feedback for
the Qwen family only; `current` remains the cross-family and shipped default.

The eight cell bundles are:

- MamayLM `allow/current`:
  `.data/agentic-loop-policy/20260731T123111.702646Z-99c26f12d1c7/`
- MamayLM `noop/current`:
  `.data/agentic-loop-policy/20260731T123112.592149Z-ad0261bc51cf/`
- MamayLM `noop/uk`:
  `.data/agentic-loop-policy/20260731T123112.777345Z-8e1b53d48e28/`
- MamayLM `noop/bilingual`:
  `.data/agentic-loop-policy/20260731T123112.965330Z-ea4c46af639f/`
- Qwen `allow/current`:
  `.data/agentic-loop-policy/20260731T124056.430200Z-95e2e6b53e49/`
- Qwen `noop/current`:
  `.data/agentic-loop-policy/20260731T124057.322698Z-c5209e02c44a/`
- Qwen `noop/uk`:
  `.data/agentic-loop-policy/20260731T124057.491186Z-5df4be92d12a/`
- Qwen `noop/bilingual`:
  `.data/agentic-loop-policy/20260731T124057.667031Z-a3be207406a8/`

`tests/llb/bench/test_agentic_loop_feedback.py` checks exact grid construction, design validation,
redirect telemetry, prospective decision gates, rendered reporting, persistence, and the committed
design/task contract. The run path was also exercised end to end on both predeclared local models.
Validation on 2026-07-31: `make ci` passed 2,463 tests with 45 opt-in/slow tests deselected, and
`make lint-md` passed.

###### Seeded Cross-Family Generalization

`samples/benchmarks/agentic_loop_feedback_generalization_design.json` extends the same immutable
32-task ledger and fixed `6/answer/allow,noop` policy to four independent model families, seeds 13
and 29, and temperature 0.2. The roster retains Qwen3 14B and MamayLM-Gemma 12B and adds Aya
Expanse 8B and Mistral Small 3.1 24B. The predeclared adoption rule requires support on both seeds
for a family, at least three of four supported families, and support from an added family before a
global change is possible.

`make bench-agentic-loop-repeat-feedback-generalization` validates the complete roster, task
digest, sampling contract, and installed Ollama models before inference. Seed and temperature now
flow through both native and OpenAI-compatible Ollama completion paths. The runner persists every
three-cell family/seed bundle before aggregation; the aggregate records the exact coordinate grid,
coverage, current and bilingual activation, response rate, completion and paired cost deltas, cell
manifests, stable family routing, and the global decision. The core analysis and reporting live in
`src/llb/bench/agentic_loop_feedback_generalization.py` and
`src/llb/bench/agentic_loop_feedback_generalization_report.py`.

CUDA-host evidence (2026-07-31), RTX 4060 Ti 16 GB:

| family | seed | response | completion | completion delta | support |
| --- | ---: | ---: | ---: | ---: | --- |
| Aya Expanse 8B | 13 | 0.107 | 0.312 | -0.094 | no |
| Aya Expanse 8B | 29 | 0.037 | 0.344 | -0.219 | no |
| Mistral Small 3.1 24B | 13 | 1.000 | 0.875 | 0.000 | no |
| Mistral Small 3.1 24B | 29 | 1.000 | 0.906 | 0.000 | no |
| Qwen3 14B | 13 | 0.844 | 0.844 | +0.344 | yes |
| Qwen3 14B | 29 | 0.812 | 0.812 | +0.312 | yes |
| MamayLM-Gemma 12B | 13 | 0.062 | 0.219 | +0.062 | no |
| MamayLM-Gemma 12B | 29 | 0.062 | 0.219 | +0.062 | no |

All eight family/seed coordinates passed task coverage and both activation gates. Qwen alone
cleared the completion and paired cost gates on both seeds, so it routes to `bilingual`; Aya,
Mistral, and Gemma remain on `current`. Stable support is one of four families, below the declared
three-family threshold, and neither added family supports the variant. The global feedback default
therefore remains `current`.

The audit-complete aggregate is
`.data/agentic-loop-policy/20260731T203955.027095Z-138f43552ed1/manifest.json`; its analysis indexes
all 24 additive cell manifests. `tests/llb/bench/test_agentic_loop_feedback_generalization.py`
checks the prospective design, exact family/seed grid, coordinate metadata, activation telemetry,
stable routing, global adoption rule, reporting, and persistence.
Validation on 2026-07-31: `make ci` passed 2,469 tests with 45 opt-in/slow tests deselected.

###### Family-Adapted Repeat Feedback

The family-adaptation lane tests one concise controller notice per non-Qwen family without letting
wording leak across families. Its prospective design is
`samples/benchmarks/agentic_loop_feedback_family_adaptation_design.json`; it fixes the powered
32-task digest, seeds 13 and 29, temperature 0.2, the `6/answer/allow,noop` policy, a `+0.25`
completion target, 10% prompt-token and 20% wall-time ceilings, and a two-of-three-family adoption
threshold. The registered ASCII notices are:

- `aya_direct`: `[loop] Repeated tool call skipped. Choose a different action or give the final
  answer now.`
- `mistral_use`: `[loop] Repeated call skipped. Use the existing result: answer now, or change the
  tool arguments.`
- `gemma_choice`: `[loop] Repeated call skipped. Output one different JSON tool call or the final
  answer; do not repeat.`

`make bench-agentic-loop-repeat-feedback-family-adaptation` validates the exact Aya, Mistral, and
Gemma Ollama roster, notice text and hypotheses, sampling contract, task digest, seed grid, and
candidate isolation before inference. The implementation lives in
`src/llb/bench/agentic_loop_feedback_adaptation.py`, its report and persistence layer in
`src/llb/bench/agentic_loop_feedback_adaptation_report.py`, and its CLI orchestration in
`src/llb/cli/bench/category_agentic_loop_feedback_adaptation.py`. Each aggregate seed row exposes
coverage, baseline and candidate activation, completion, prompt-cost, wall-cost, and combined-cost
gate decisions in addition to response and effect values. `src/llb/bench/agentic/episode.py` also
counts a final answer after a suppressed repeated call as a redirect, including when the
malformed-call policy accepts that answer as the episode result.

CUDA-host evidence (2026-08-01), RTX 4060 Ti 16 GB, identical 32-task digest. In the gates column,
`C`, `P`, and `W` mean completion, prompt-token cost, and wall-time cost respectively; `-` is the
only failed gate. Coverage and both activation checks passed in every row.

| family | seed | candidate | response | completion | completion delta | prompt delta | wall delta | gates | support |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Aya Expanse 8B | 13 | `aya_direct` | 0.643 | 0.562 | +0.156 | -227.8 | -4.91 | `-PW` | no |
| Aya Expanse 8B | 29 | `aya_direct` | 0.667 | 0.594 | +0.031 | -412.1 | -4.56 | `-PW` | no |
| Mistral Small 3.1 24B | 13 | `mistral_use` | 1.000 | 0.875 | 0.000 | +4.0 | -0.02 | `-PW` | no |
| Mistral Small 3.1 24B | 29 | `mistral_use` | 1.000 | 0.906 | 0.000 | +4.4 | -0.20 | `-PW` | no |
| MamayLM-Gemma 12B | 13 | `gemma_choice` | 0.281 | 0.375 | +0.219 | -585.3 | -1.15 | `-PW` | no |
| MamayLM-Gemma 12B | 29 | `gemma_choice` | 0.250 | 0.375 | +0.219 | -579.4 | -1.65 | `-PW` | no |

Aya's completion gains were seed-sensitive and below the material target. Mistral already
redirected every activated candidate episode but gained no completion, so its adapted wording tied
`current`. Gemma produced seven wins and no losses on each seed, a separated
`+0.219 [0.094, 0.375]` completion delta, but missed the predeclared `+0.25` mean target by one of
32 tasks. Its notice redirected all eight search tasks on both seeds, zero read and calculator
tasks, and only one mutation task on seed 13. This stable but narrow effect does not justify a
family-wide route.

No candidate cleared the completion gate on either seed. Each family therefore has zero of two
supporting seeds, every family remains routed to `current`, and the supported-family fraction is
zero of three, below the declared cross-family threshold. The audit-complete aggregate is
`.data/agentic-loop-policy/20260801T122638.618382Z-450efe8e5e90/manifest.json`; it indexes all 18
source cell manifests and carries the explicit per-gate decisions.

`tests/llb/bench/test_agentic_loop_feedback_adaptation.py` checks the immutable design, exact
wording, family/seed grid, candidate isolation, stable routing, aggregate gate reporting,
persistence, and an end-to-end fake run. The redirect regression is in
`tests/llb/bench/test_agentic_loop_policy.py`. Validation on 2026-08-01: `make ci` passed 2,475
tests with 45 opt-in/slow tests deselected.

###### Task-Family-Neutral Gemma Transfer

The transfer lane tests one task-family-neutral Gemma notice on a fresh holdout instead of tuning
against the family-adaptation ledger. Its prospective design is
`samples/benchmarks/agentic_loop_feedback_task_family_transfer_design.json`; the balanced
32-task ledger is `samples/benchmarks/agentic_loop_feedback_task_family_transfer.json`, with eight
new ASCII cases in each of read, calculator, search, and mutation. Mutation success requires both
the state change and a confirming final answer, so a successful first write cannot hide failure to
advance after suppression. The holdout digest is
`10fef23bc2b2d855f6b7395d7e94ac42013005b4967d29d1a968ada99a215465`, distinct from the powered
ledger digest recorded in the design.

The immutable `gemma_progress` notice is `[loop] The previous action already succeeded. Continue
from its result instead of repeating it.` It contains no tool family, task name, or expected value.
The validator fixes that exact registered text, its completed-state hypothesis, seeds 41 and 73,
temperature 0.2, 8192-token served context, a 25% response floor in at least three of four task
families on both seeds, a `+0.125` material paired completion target, and maximum relative cost
increases of 10% for prompt tokens and 20% for wall time. It also refuses the prior ledger digest
and any task-specific word in the controller notice before inference.

`make bench-agentic-loop-repeat-feedback-task-family-transfer` runs the fixed
`6/answer/allow,noop` comparison on the local MamayLM-Gemma 3 12B model. The core validation and
two-seed decision live in `src/llb/bench/agentic_loop_feedback_transfer.py`; report and aggregate
persistence live in `src/llb/bench/agentic_loop_feedback_transfer_report.py`; orchestration lives
in `src/llb/cli/bench/category_agentic_loop_feedback_transfer.py`. Aggregate rows retain baseline
and candidate response, per-family response deltas, the full paired completion comparison, both
full cost-gate objects, and links to every source cell manifest.

CUDA-host evidence (2026-08-01), RTX 4060 Ti 16 GB, MamayLM-Gemma 3 12B Q4_K_M. In the gates
column, `R`, `C`, `P`, and `W` mean task-family response, completion, prompt-token cost, and
wall-time cost; `-` is a failed gate.

| seed | current response | candidate response | responsive families | completion delta | prompt delta | wall delta | gates |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 41 | 0.000 | 0.094 | mutation 0.375 | +0.000 | +27.4 [6.2, 43.3] | -0.19 [-0.84, 0.35] | `--PW` |
| 73 | 0.031 | 0.062 | mutation 0.250 | +0.000 | +37.3 [26.0, 43.4] | +2.52 [0.38, 4.87] | `--PW` |

Coverage and baseline/candidate activation passed on both seeds. Read, calculator, and search had
zero candidate redirects on both seeds; mutation alone reached its response floor, with three
redirects on seed 41 and two on seed 73. None of those redirects completed a task. Candidate and
current completion were both `0.000`, all 32 completion pairs tied on each seed, and the paired
completion interval was `[0.000, 0.000]`. Both paired cost gates passed. Thus the completed-state
hypothesis does not transfer a useful redirect across task families, and `current` remains the
recommended Gemma feedback variant.

The audit-complete aggregate is
`.data/agentic-loop-policy/20260801T145240.263763Z-778f62e7297b/manifest.json`. It indexes the six
source cell manifests under `.data/agentic-loop-policy/20260801T134214*` through
`.data/agentic-loop-policy/20260801T145005*` and preserves the exact prospective design. The first
aggregate remains at `.data/agentic-loop-policy/20260801T145005.639616Z-1bdfdb79eb25/`; the later
aggregate adds baseline response and the complete paired gate objects from those same source
cells, with no additional inference.

`tests/llb/bench/test_agentic_loop_feedback_transfer.py` checks the immutable neutral notice and
hypothesis, fresh digest, balanced ledger, exact seed grid, candidate isolation, three-family and
two-seed response rule, completion and cost decisions, report persistence, and an end-to-end fake
run. Validation on 2026-08-01: `make ci` passed 2,479 tests with 45 opt-in/slow tests deselected.

###### Controller-Authority Gemma Transfer

The controller-authority lane tests whether an explicit controller ruling can overcome Gemma's
literal repetition on a second fresh balanced holdout. The registered `gemma_authority` notice is
`[loop] Controller ruling: suppression satisfies the requested repetition. You must now take the
next distinct action.` It is ASCII, contains no task name, expected value, tool family, or
family-specific action choice, and remains a controller observation rather than a task-specific
hint.

The prospective design is
`samples/benchmarks/agentic_loop_feedback_controller_authority_design.json`; its 32-case ledger is
`samples/benchmarks/agentic_loop_feedback_controller_authority.json`, with eight new read,
calculator, search, and mutation cases. The ledger digest is
`a2e8e0bf49c04ca27cebb9d06072e7008026f874a93a99ae5098d3b938b98f82`, distinct from both prior
ledgers fixed in the design. The contract fixes MamayLM-Gemma 3 12B, seeds 107 and 149,
temperature 0.2, an 8192-token context, the `6/answer/allow,noop` grid, a 25% response floor in at
least three families on both seeds, a `+0.125` paired completion target, minimum four discordant
pairs, and maximum relative increases of 10% for prompt tokens and 20% for wall time.

`make bench-agentic-loop-repeat-feedback-controller-authority-transfer` validates the full
contract before inference and writes a two-seed aggregate. The immutable notice lives in
`src/llb/bench/agentic/loop_policy.py`; authority validation and decision wrapping live in
`src/llb/bench/agentic_loop_feedback_authority.py`; response-versus-completion summaries live in
`src/llb/bench/agentic_loop_feedback_outcomes.py`; reporting lives in
`src/llb/bench/agentic_loop_feedback_authority_report.py`; and CLI orchestration shares
`src/llb/cli/bench/category_agentic_loop_feedback_neutral.py` with the earlier neutral-transfer
lane. Aggregate persistence uses the design's study kind, so authority artifacts cannot be
mislabelled as the earlier task-family-transfer study.

CUDA-host evidence (2026-08-01), RTX 4060 Ti 16 GB, MamayLM-Gemma 3 12B Q4_K_M. Each family cell
below is `response rate / redirected completion rate`; the latter counts completions after a
changed post-suppression action over activated tasks.

| seed | current response | candidate response | calculator | mutation | read | search | completion delta | prompt delta | wall delta | gates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 107 | 0.031 | 0.344 | 0.125 / 0.000 | 0.625 / 0.000 | 0.625 / 0.625 | 0.000 / 0.000 | +0.156 | -374.1 | -1.23 | `--PW` |
| 149 | 0.094 | 0.312 | 0.000 / 0.000 | 0.625 / 0.000 | 0.500 / 0.500 | 0.125 / 0.125 | +0.156 | -365.8 | -7.44 | `--PW` |

Coverage and baseline/candidate activation passed on both seeds. Read cleared the response floor
and every read redirect completed; mutation also cleared the response floor but none of its ten
redirects produced the required confirming final answer. Calculator stayed below the floor on
both seeds, and search reached only one successful redirect on seed 149. Thus only two families
were responsive per seed, below the required three.

The candidate produced five wins, no losses, and 27 ties on each seed for a paired `+0.156`
completion delta with interval `[+0.031, +0.281]`. The standard stability reading remained
borderline and `insufficient_evidence` (`randomization p=0.03125`, sign-test `p=0.0625`), so the
completion gate also failed. Prompt-token deltas were `-374.1 [-748.3, -16.2]` and
`-365.8 [-741.4, -7.3]`; wall-time deltas were `-1.23 [-4.07, +1.40]` seconds and
`-7.44 [-9.80, -5.33]` seconds. Both paired cost gates passed on both seeds.

The authority wording therefore shows stable read completion and mutation response, but it does
not establish task-family transfer; `current` remains the recommended Gemma feedback variant. The
audit-complete aggregate is
`$DATA_DIR/agentic-loop-policy/20260801T180932.422116Z-a925598313d9/manifest.json`; it links all six
source cell manifests and preserves response-versus-completion outcomes per family.

`tests/llb/bench/test_agentic_loop_feedback_authority.py` checks the immutable wording,
hypothesis, fresh balanced ledger, seeds, candidate isolation, breadth and paired gates,
authority-specific study identity, persistence, and an end-to-end fake run. The shared feedback
tests check per-family redirected completion accounting. Validation on 2026-08-01: `make ci`
passed 2,483 tests with 45 opt-in/slow tests deselected, and `make lint-md` passed.

###### Controller-Channel Authority

The controller-channel lane isolates transcript authority from authority wording. After an
identical repeated call is suppressed, both cells send the same task message and the same immutable
authority text in the same message position. Only the authority message role changes:
`observation` serializes to `user`, while `controller` serializes to `system`. The exact mapping is
declared for native Ollama and OpenAI-compatible chat in
`samples/benchmarks/agentic_controller_channel_authority_design.json`; typed serialization lives in
`src/llb/bench/agentic/controller_channel.py`. This keeps the task prompt and all ordinary tool
observations fixed and gives the agent loop a backend-neutral controller-message seam.

`make bench-agentic-loop-controller-channel-authority` runs the predeclared two-seed comparison.
The fresh 32-case ledger is
`samples/benchmarks/agentic_controller_channel_authority.json`, balanced over eight read,
calculator, search, and mutation cases. Its digest is
`5d6148e0851ca749a65fd75768b388931419ae3dccf2c03be20c095c97e9ead1`. The contract fixes
MamayLM-Gemma 3 12B Q4_K_M, seeds 211 and 257, temperature 0.2, 512 completion tokens, an
8192-token context, six controller steps, and the same 25% response floor in at least three
families on both seeds. Adoption also requires a paired completion gain of at least 0.125, at
least four discordant completion pairs, and maximum relative increases of 10% for prompt tokens
and 20% for wall time.

Every source cell persists `prompt-snapshots.json`. Analysis pairs the first authority-bearing
snapshot by task and refuses the run unless the full message content is byte-identical while only
the final role changes. Runner, analysis, and persistence live in
`src/llb/bench/agentic_controller_authority_run.py`,
`src/llb/bench/agentic_controller_authority.py`, and
`src/llb/bench/agentic_controller_authority_report.py`. The general backend adapter now exposes
typed-message `local_chat` and `launcher_chat` callables alongside the legacy string-prompt
adapters.

CUDA-host evidence (2026-08-01), RTX 4060 Ti 16 GB, MamayLM-Gemma 3 12B Q4_K_M:

| seed | gates | observation response | controller response | observation completion | controller completion | completion delta | prompt delta | wall delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 211 | `SA--PW` | 0.094 | 0.031 | 0.000 | 0.000 | +0.000 | +7.4 | -0.32 s |
| 257 | `SA--PW` | 0.094 | 0.062 | 0.000 | 0.000 | +0.000 | +5.8 | -2.05 s |

`S`, `A`, `R`, `C`, `P`, and `W` denote snapshot, activation, family response, completion,
prompt-cost, and wall-cost gates. Snapshot and activation coverage passed for all 32 tasks on both
seeds, and both paired cost gates passed. The controller role responded only in mutation: 0.125
on seed 211 and 0.250 on seed 257, with zero redirected completions. Calculator, read, and search
response were zero. Both placements completed 0/32 tasks on both seeds, leaving zero discordant
completion pairs and a flat paired completion reading.

Structural controller authority is therefore not supported for this Gemma model and transcript
shape; `observation` remains the recommended placement, and no shipped default changes. The
audit-complete aggregate is
`$DATA_DIR/agentic-loop-policy/20260801T201419.190139Z-62e3df17e112/manifest.json`; its analysis
links the four source manifests and all 64 paired snapshot proofs. The negative result is scoped to
this model and serialization, not a claim that role never matters across model families.

`tests/llb/bench/test_agentic_controller_authority.py` checks exact role-only serialization,
fresh-ledger and two-seed validation, balanced family coverage, snapshot refusal, every adoption
gate, persistence, the committed contract, and an end-to-end fake run.
Validation on 2026-08-01: `make ci` passed 2,487 tests with 45 opt-in/slow tests deselected, and
`make lint-md` passed.

**Qwen cross-model transfer.** `make bench-agentic-loop-controller-channel-cross-model` applies
the same authority text, role mapping, message order, task shape, sampling policy, and six adoption
gates to the non-Gemma `qwen3:14b` family. Its prospective design is
`samples/benchmarks/agentic_controller_channel_cross_model_design.json`; its fresh 32-case ledger
is `samples/benchmarks/agentic_controller_channel_cross_model.json`, balanced over eight cases in
each family with digest
`177adb511124b972f748a1ef8beb21365f1bcee315c3039c11fb43e4413bcc70`. The contract fixes seeds
307 and 353, temperature 0.2, 512 completion tokens, and an 8192-token context. A distinct
`agent_loop_policy_controller_channel_authority_cross_model` study kind and immutable reference to
the Gemma study prevent the transfer row from being mislabelled or pointed back at Gemma.

CUDA-host evidence (2026-08-02), RTX 4060 Ti 16 GB, Qwen3 14B:

| seed | gates | observation response | controller response | observation completion | controller completion | completion delta | prompt delta | wall delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 307 | `SA--P-` | 0.031 | 0.000 | 0.031 | 0.000 | -0.031 | +89.3 | +2.25 s |
| 353 | `SA--P-` | 0.031 | 0.000 | 0.031 | 0.000 | -0.031 | +89.3 | +1.92 s |

Snapshot and activation gates passed for all 32 tasks on both seeds, and the prompt-cost gate
passed. The controller role redirected zero tasks in every family on both seeds, so the response
gate failed. The observation role redirected and completed one search task per seed; the controller
role completed none, producing one loss and 31 ties per seed with interval `[-0.094, 0.000]`.
Controller wall-time increases were separated above the 20% ceiling on both seeds, so the wall-cost
gate also failed. Observation remains the recommendation: structural controller authority does not
transfer to this Qwen chat template and transcript shape, and no shipped default changes.

The audit-complete aggregate is
`$DATA_DIR/agentic-loop-policy/20260802T052500.927860Z-2dd4f2c196c8/manifest.json`; it links four
source cell manifests, 64 role-only snapshot proofs, and per-cell throughput from 16.21 to 22.26
tokens/s. Cross-model validation lives beside the base contract in
`src/llb/bench/agentic_controller_authority_design.py`; the CLI model preflight now queries the
configured Ollama host, and the dedicated Make target pins the committed cross-model design and
ledger.
Validation on 2026-08-02: `make ci` passed 2,489 tests with 45 opt-in/slow tests deselected, and
`make lint-md` passed.

**Template-native preamble placement.**
`make bench-agentic-loop-controller-preamble-placement` separates canonical template placement
from the earlier role-only comparison. The observation baseline serializes
`[task prompt:user, authority:user]`; the candidate serializes
`[authority:system, task prompt:user]`. The immutable authority bytes and task-prompt bytes are
identical across the pair. Both Ollama and OpenAI-compatible transforms are declared exactly in
`samples/benchmarks/agentic_controller_preamble_placement_design.json`; inference is refused if a
transform, authority byte, model-seed cell, or first authority-bearing prompt pair differs.

The prospective design reuses the 32-case balanced controller-channel ledger and its
`5d6148e0851ca749a65fd75768b388931419ae3dccf2c03be20c095c97e9ead1` digest so only placement and
model family vary. It fixes MamayLM-Gemma 3 12B Q4_K_M and `qwen3:14b`, seeds 401 and 443,
temperature 0.2, 512 completion tokens, an 8192-token context, six steps, and the existing
snapshot, activation, three-family response, paired completion, prompt-cost, and wall-cost gates.
Adoption requires all four model-seed cells to pass.

Typed source/role transforms live in `src/llb/bench/agentic/controller_channel.py`; the episode
seam is in `src/llb/bench/agentic/episode.py`. Multi-model grid validation, execution, analysis,
reporting, and CLI orchestration reuse the controller-authority modules. The result schema exposes
generic candidate-placement support and a preamble-specific decision while retaining the earlier
controller-channel fields for artifact compatibility. Every source cell persists its exact first
authority-bearing prompt snapshots.

CUDA-host evidence (2026-08-02), RTX 4060 Ti 16 GB:

| family | seed | gates | observation response | preamble response | observation completion | preamble completion | completion delta | prompt delta | wall delta |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma | 401 | `SA--PW` | 0.062 | 0.125 | 0.000 | 0.000 | +0.000 | -5.7 | -1.91 s |
| Gemma | 443 | `SA--P-` | 0.062 | 0.062 | 0.000 | 0.000 | +0.000 | +3.8 | +7.36 s |
| Qwen | 401 | `SA--P-` | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | +0.0 | +2.09 s |
| Qwen | 443 | `SA--P-` | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | +0.0 | +2.10 s |

All 128 prompt pairs passed the snapshot proof and every cell passed activation and prompt-cost
gates. Gemma preamble response was confined to mutation: 4/8 on seed 401 and 2/8 on seed 443,
with no redirected completion. Read, calculator, and search response was zero. Both placements
completed 0/32 tasks on both seeds, so every completion comparison was flat with zero discordant
pairs. Gemma wall cost passed on seed 401 and failed on seed 443.

Qwen produced no redirect or completion under either placement on either seed. Its preamble added
`+2.092 [+2.038, +2.124]` and `+2.095 [+2.042, +2.128]` seconds per paired task, above the 20%
wall-cost ceiling on both seeds; prompt-token deltas were exactly zero. The template-native
preamble therefore does not improve repeated-call recovery for either tested family and makes
Qwen materially slower. `observation` remains the recommendation, with no shipped-default change.

The audit-complete aggregate is
`$DATA_DIR/agentic-loop-policy/20260802T081304.031917Z-fabb673e7134/manifest.json`; it links eight
source manifests, four gate rows, and 128 paired snapshot proofs. Source-cell throughput was
4.9-5.2 tokens/s for Gemma and 20.2 tokens/s for Qwen.

`tests/llb/bench/test_agentic_controller_preamble.py` checks both backend transforms, the exact
two-family/two-seed design, snapshot refusal, every gate, and an end-to-end fake run. The existing
controller-channel tests protect backward compatibility.
Validation on 2026-08-02: `make ci` passed 2,493 tests with 45 opt-in/slow tests deselected, and
`make lint-md` passed.

CrewAI is optional and lazy-imported. The adapter wraps the candidate completion function as a
CrewAI LLM, builds tools from the benchmark tool definitions, and disables telemetry/tracing for a
local no-egress run.

The `[crewai]` extra is a standalone install lane in `uv`: upstream CrewAI pins older Chroma,
LanceDB, and `tomli` ranges than the repo's RAG/vector/dev extras. `pyproject.toml` declares those
extra conflicts so `uv lock` stays resolvable while `uv pip install -e ".[crewai]"` still works for
host validation.

## Agent Context-Management Policies

`bench-agentic-context` ranks how the agent spends its context window for ONE fixed model over
one task set. It is the agent-side sibling of the chain-context lane below: the model, the task set,
the tool world, the success checks, and the step budget are held FIXED and only the
context-management policy varies, so every difference it reports is attributable to context
handling. Every policy runs a fresh episode through the pure `loop` harness by default; the same
`policy`/`budget` kwargs transfer onto `langgraph` via the widened `Harness` protocol (see
[Agentic Harness Comparison](#agentic-harness-comparison)), while CrewAI records the policy as
unsupported because it owns its own transcript.

Core locations:

- `src/llb/bench/agentic/context.py`: the policy vocabulary, observation trimming, transcript
  assembly, compaction, and the per-episode telemetry;
- `src/llb/bench/agentic/context_budget.py`: the per-step prompt budget (`ContextBudget`) and its
  resolution from the declared + probed usable window;
- `src/llb/backends/served_window.py`: per-backend probe of the window the runtime is actually
  serving (Ollama `/api/ps`, vLLM `/v1/models`, llama.cpp `/props`);
- `src/llb/bench/agentic/episode.py`: `build_agent_prompt_lines` (the policy seam) and the
  policy-aware `run_episode`;
- `src/llb/bench/agentic_context.py`: the four-policy run + persistence;
- `src/llb/bench/agentic_context_report.py`: the paired reading, the policy table, and the
  recommendation;
- `src/llb/bench/agentic_context_sweep.py`: the constant-grid sweep (cap / head-share / keep_last_n)
  and pin/expose/inapplicable verdicts;
- `src/llb/board/agentic_context.py`: one-model policy comparison board rows;
- `src/llb/prompts/templates/bench/agentic/compact_summary.*`: the reviewable compaction prompt.

The four policies (each a fresh episode over the identical task set):

- `full` -- the whole transcript verbatim; today's shipped behavior and the BASELINE row every
  other policy is paired against. Its prompt is byte-identical to the pre-policy loop's, so the
  baseline reproduces the recorded agentic rows exactly;
- `observation_cap` -- every observation trimmed to a char budget, HEAD and TAIL kept around an
  explicit elision marker naming the dropped char count, so the model can tell it is reading a
  fragment rather than a short tool result. When trimmed, a machine-computed aggregate header
  (`[агрегат: hits=N chars=M docs=...]`) is prepended outside the cap so a count question stays
  answerable after a middle-of-list loss (`src/llb/bench/agentic/context_aggregate.py`);
- `keep_last_n` -- only the last N steps survive, with a marker line announcing how many were
  dropped so a missing step is visible instead of looking like it never happened;
- `compact` -- once the prompt crosses a share of the usable window, a model-written running summary
  replaces the older steps (the agent-loop counterpart of the chain lane's `summary`). One recent
  step stays verbatim; when the prompt was blown by that most recent observation there are no older
  steps to fold, and the whole transcript is summarized rather than letting the policy degenerate
  into `full`. Live steps also share the `observation_cap` trim (same char budget and aggregate
  header), so a fat search hit does not re-blow every later prompt. When the summary already
  carries machine hit-count facts, a finish cue names the known `hits=` value and tells the model
  to call `finish` instead of searching again. The summary marker carries the count of steps it
  stands in for, so a folded step is never silently absent. Search observations being folded also
  inject their aggregate headers into the summary text itself (not only into the summarizer prompt),
  so hit counts do not depend on the free-text summary remembering a number. Two rules keep the
  policy honest: at most ONE compaction per step (if the compacted prompt still does not fit, the
  guard is what ends the episode, not another round of summarizing), and the summarize call's INPUT
  is ITSELF capped -- its input is the transcript that just blew the step prompt, so an uncapped
  summarizer is the one call in the loop guaranteed to overflow, and it would return a silently
  truncated summary the policy then trusts for the rest of the episode. Which bound is a policy
  field, `summary_input_cap`: the shipped `window` is the resolved prompt budget minus the summary
  template (including the elision marker the trim writes on top of its cap), so the folded
  transcript is summarized at its OWN size whenever it fits; the legacy `trigger`
  (`compact_share * guard`) is kept selectable because the published fold-step, trigger-collapse,
  and boundary-surface evidence was measured under it -- see
  [the summarize-input cap](#the-summarize-input-cap-is-step-aligned). An empty summary is treated
  as a no-op rather than folding those steps away with nothing standing in for them.

Underneath all four sits the guard the loop never had. `ContextBudget` resolves the usable prompt
budget ONCE per run from the DECLARED window (host planner cap, model window, `max_model_len`,
explicit `context_budget` via `effective_max_context` in `src/llb/optimize/tuning_space.py`) and a
LIVE probe of what the backend is serving (`src/llb/backends/served_window.py`). The budget is the
MINIMUM of those two; `budget_source` names which side bound it (`declared` or `served`), and both
`declared_max_model_len` and `served_max_model_len` are recorded in the
`$DATA_DIR/agentic-context/<run>/` and `$DATA_DIR/agentic/<run>/` manifests. A probe miss falls back
to the declared window and records `served_max_model_len=null` with `budget_source=declared`. This
closes the Ollama hole where a GGUF advertising 131072 still serves `num_ctx=4096` by default: the
guard no longer approves prompts the backend will silently truncate. For Ollama, `bench-agentic` /
`bench-agentic-context` always drive the native `/api/chat` launcher and pass `num_ctx` from
`--max-model-len` / `context_budget`, warming the model before the probe so `/api/ps` reports the
window the run will actually use. The same applies when the operator passes `--base-url` pointing
at Ollama's OpenAI-compatible `/v1` endpoint: `drive_with_backend` detects that the URL resolves
to the same host as `ollama_host` (`is_ollama_base_url` in `src/llb/backends/served_window.py`)
and routes the call through the native launcher on that host rather than the OpenAI-compat
`local_complete` path, which silently ignored `extra_body.options.num_ctx` on some Ollama builds.
When the URL points at a different host (a non-Ollama OpenAI-compat backend), the generic
`local_complete` path is used unchanged. Each step's prompt is checked before the call. A prompt
that does not fit is NEVER SENT: the episode terminates as `context_overflow` -- the status already
in the shared taxonomy (`src/llb/eval/common.py`) that the context-ablation lane raises for the same
reason -- so an unusable configuration is a typed outcome instead of a wrong answer. An unresolvable
window (no model spec, no served cap, no explicit budget, no probe) refuses nothing, matching
`fits_context_chars`: an unknown model never silently declares a prompt unusable.

Per-episode telemetry rides ALONGSIDE the headline and is what makes the overflow observable after
the fact: `max_prompt_tokens`, `total_prompt_tokens`, `total_model_input_tokens`,
`compaction_prompt_tokens`, `n_model_calls`, `observation_bytes` (counted BEFORE any policy trim,
so `full` and `observation_cap` stay comparable on it), `n_compactions`, and
`n_trimmed_observations` per case row. `total_prompt_tokens` counts assembled controller prompts,
including a locally refused final prompt; `total_model_input_tokens` counts only prompts actually
sent to the model and includes compact-summary prompts. A policy changes what the model SEES, never
what the run reports the agent did -- the persisted transcript and the trajectory judge read the
full executed record even when `compact` has folded it out of the prompt.

Every non-baseline policy carries a paired delta against `full` on four metrics -- completion,
steps, tool calls, and prompt tokens -- over SHARED bootstrap index sets, so an interval is about
the DIFFERENCE and not about two lanes' separate sampling noise. The statistics are
`llb.rag.fusion_evidence` wholesale (`paired_comparison`, `bootstrap_index_sets`,
`apply_evidence_gate`), including the `insufficient_evidence` relabel when a positive interval rests
on too few differing tasks for the reporting level to be reachable. The recommendation names the
best policy per model only on a SEPARATED completion delta; otherwise it states that the policies
are flat at this task count and falls back to the cheapest prompt among policies that do not
overflow more often than the baseline.

```bash
llb bench-agentic-context --tasks <tasks.json> --model <model> --backend <backend> \
  --policies full,observation_cap,keep_last_n,compact \
  --observation-cap-chars 800 --observation-head-share 0.6 --keep-last-n 3
llb bench-agentic-context-compare --model <model>
llb bench-agentic-context-sweep --tasks <tasks.json> --model <model> --backend <backend>
make bench-agentic-context MODEL=<model> BACKEND=<backend> \
  AGENT_CONTEXT_POLICIES=full,observation_cap,keep_last_n,compact
make bench-agentic-context-sweep MODEL=<model> BACKEND=<backend> \
  AGENT_CONTEXT_SWEEP_TASKS=<tasks.json> AGENT_CONTEXT_SWEEP_MAX_MODEL_LEN=8192
```

`AGENT_CONTEXT_MAX_PROMPT_CHARS` (or `--max-prompt-chars`) forces the budget instead of resolving
it, which is how the guard is exercised on purpose. `AGENT_CONTEXT_MAX_MODEL_LEN` (or
`--max-model-len`) is the declared window forwarded to Ollama as `num_ctx` and compared against the
probed served window. `AGENT_CONTEXT_OBSERVATION_HEAD_SHARE` (or `--observation-head-share`) is the
fraction of the observation-cap budget kept from the head. Each policy persists its OWN bundle under
`$DATA_DIR/agentic-context/<timestamp>/` tagged with the policy (mirroring the per-harness and
per-chain-policy bundles); provenance records the policy, its settings, the `task_set_digest` that
proves both policies ran the same set, the resolved `max_prompt_chars`, `declared_max_model_len`,
`served_max_model_len`, `budget_source`, the overflow count, and the `paired_vs_full` block. The
bundles live under their own method root, so they are never mixed into the harness comparison or the
category composite, which both read `agentic`.

CI drives every policy, the compaction path, and the guard over the fake endpoint with no GPU
(`tests/llb/bench/test_agentic_context.py`, `tests/llb/backends/test_served_window.py`), including
the assertion that the `full` policy's prompt is byte-identical to the pre-policy loop's, that no
episode in any policy sends a prompt over the resolved window, and that a declared window larger
than a probed one is bound by the probe. The constant-sweep lane's trim arithmetic and pin/expose
verdicts are covered in `tests/llb/bench/test_agentic_context_sweep.py`.

CUDA host smoke (2026-07-29, MamayLM-Gemma-3-12B-IT-v2.0 on Ollama): after
`OllamaLauncher.ensure_num_ctx(8192)`, `/api/ps` reported `context_length=8192` and
`resolve_context_budget(..., probe=True)` with `--max-model-len 32768` bound to
`budget_source=served` / `served_max_model_len=8192`. A `make bench-agentic-context` pass with
`full,observation_cap` persisted those provenance fields on the agentic-context manifests.

`--base-url` routing (2026-07-29): unit test `test_drive_with_backend_routes_ollama_base_url_through_native_launcher`
in `tests/llb/backends/test_served_window.py` confirms that when `--base-url` resolves to the same
host as `ollama_host` and `--max-model-len` is set, `drive_with_backend` routes through
`OllamaLauncher` (native `/api/chat`) with the correct `num_ctx` instead of `local_complete`.

### Context-policy evidence on the 16 GB RTX 4060 Ti host

Run date 2026-07-28. `MamayLM-Gemma-3-12B-IT-v2.0` on Ollama, 24 tasks (the committed 4-task UA
seed plus 20 generated `search` tasks over the 250-document `ua_squad_postedited_v1` corpus, whose
observations run
20k-36k chars), all four policies in one ~46 min invocation, 377 model calls at 5.7 tok/s. Ollama
serves a 4096 window here regardless of the GGUF advertising 131072, so the run declared
`--max-model-len 4096` and the guard resolved a 9216-char prompt budget. Bundles under
`.data/agentic-context/20260728T2034*` (one per policy).

| policy | completion | steps | max prompt tok | overflow | reliability | d(prompt-tok) vs `full` |
| --- | --- | --- | --- | --- | --- | --- |
| `full` | 0.458 | 2.42 | 3950 | 10 | 0.417 | baseline |
| `observation_cap` | 0.458 | 4.33 | 1527 | 0 | 0.667 | -2423 [-3679, -1208] |
| `keep_last_n` | 0.458 | 2.42 | 3930 | 10 | 0.417 | -20 [-53, -2] |
| `compact` | 0.458 | 4.42 | 997 | 0 | 0.417 | -2953 [-4432, -1478] |

The headline reading is FLAT and is recorded as flat: all four policies complete the identical 11 of
24 tasks, item for item, so no policy separates from `full` on completion at this task count and the
recommendation names `compact` on CONTEXT COST alone. The cost side does separate -- `compact` runs
on a quarter of `full`'s prompt with its interval clear of zero.

The result worth reading is WHY the completion ties, and it is the whole argument for the typed
status. The 10 `search-locate` tasks answer at step 2-3 under every policy, before the transcript
grows enough for context management to matter. The 10 `search-count` tasks fail under every policy --
but for two DIFFERENT reasons that the completion number alone cannot tell apart: `full` and
`keep_last_n` overflow at step 1 and never get to try, while `observation_cap` and `compact` run all
six steps on evidence they can see and still count wrong. Without `context_overflow` those twenty
failures are one indistinguishable bucket, and the pre-policy loop reported exactly that.

Two policy findings fall out. `keep_last_n` is nearly a no-op here (-20 prompt tokens, the same 10
overflows): with a 6-step budget and 3 steps kept, the one oversized observation that blew the prompt
is always INSIDE the kept window, so dropping older steps cannot reach it -- the policy is aimed at
long transcripts, and this failure is a single fat observation. And the two policies that do fit the
window are lossy in a task-dependent way: a `count` question needs the whole hit list, which is
precisely what a positional trim and a summary destroy, so surviving the window and answering
correctly are not the same win.

### Aggregate-safe trimming

Trim and compaction now carry machine-computed aggregate headers
(`src/llb/bench/agentic/context_aggregate.py`): hit count, total length, and matched doc ids are
prepended when an observation is trimmed, and the same facts are injected into a compacted summary
so a free-text summarizer cannot drop them. The report breaks completion out by task kind
(`count` / `locate` / `other`) and states a vs-pre-header delta on the count slice.

`compact` also applies the same live observation-cap trim (and stamps
`n_trimmed_observations`), and when a compacted summary already carries `hits=` facts a finish cue
names that count and steers the model to call `finish` instead of searching again
(`src/llb/bench/agentic/context.py`). Without those, compact burned the 6-step budget on repeated
compactions and never finished count tasks even after summary injection.

Blackwell / RTX 4060 Ti host evidence (2026-07-29 pre-recovery; 2026-07-30 post-recovery):
`MamayLM-Gemma-3-12B-IT-v2.0` on Ollama, `--max-model-len 8192` (prompt budget 21504 chars), the
same 24-task shape (4 seed + 10 `search-count` + 10 `search-locate` over 250 UA docs). Pre-recovery
bundles under `.data/agentic-context/20260729T1515*` / `20260729T155227*`. Post-recovery bundles
under `.data/agentic-context/20260730T0952*` (266 calls, 3.6 tok/s).

Pre-recovery (aggregate headers on trim + summary injection only):

| policy | completion | count | locate | other | overflow | vs pre-header count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 0.458 | 0.000 | 1.000 | 0.250 | 10 | +0.000 |
| `observation_cap` | **0.875** | **1.000** | 1.000 | 0.250 | 0 | **+1.000** |
| `keep_last_n` | 0.458 | 0.000 | 1.000 | 0.250 | 10 | +0.000 |
| `compact` | 0.458 | 0.000 | 1.000 | 0.250 | 0 | +0.000 |

Post-recovery (compact live trim + finish cue):

| policy | completion | count | locate | other | overflow | vs pre-header count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 0.458 | 0.000 | 1.000 | 0.250 | 10 | +0.000 |
| `observation_cap` | **0.875** | **1.000** | 1.000 | 0.250 | 0 | **+1.000** |
| `keep_last_n` | 0.458 | 0.000 | 1.000 | 0.250 | 10 | +0.000 |
| `compact` | **0.875** | **1.000** | 1.000 | 0.250 | 0 | **+1.000** |

Verdict: **aggregate-safe trimming recovered the count slice under `observation_cap`** (10/10
count tasks finish with the correct integer; paired d(completion) vs `full` on the count slice is
+1.000 [+1.000, +1.000]). **`compact` now recovers the same count slice** once live steps share
the observation-cap trim: 10/10 count tasks complete, paired d(completion) vs `full` on the count
slice is +1.000 [+1.000, +1.000], and mean prompt tokens match `observation_cap` (1362). On this
6-step count-heavy set every compact count episode finished with `n_compactions=0` -- the live trim
kept prompts under the compact trigger, so the finish cue was latent insurance rather than the
active path. Operators who prefer compact for longer transcripts can keep it on count-heavy short
episodes; for this shape `observation_cap` is the simpler equivalent. `keep_last_n` and `full`
still overflow the ten count tasks at step 1-2.

#### Compact versus cap with active compaction

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
locations are `src/llb/bench/agentic_compact_vs_cap.py`,
`src/llb/bench/agentic_compact_vs_cap_report.py`,
`src/llb/cli/bench/category_agentic_compact_vs_cap.py`, and
`tests/llb/bench/test_agentic_compact_vs_cap.py`.

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

##### Memory-dependent transcript

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

Core locations are `src/llb/bench/agentic_memory_transcript.py` (task builder),
`src/llb/bench/tool_world.py` (one-way token workflow),
`src/llb/bench/agentic/context.py` (typed memory folding and finish cue), the focused compact-vs-cap
runner and report modules, `make/eval/categories-platform.mk`, and focused tests in
`tests/llb/bench/test_agentic_memory_transcript.py` and
`tests/llb/bench/test_agentic_compact_vs_cap.py`.

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

##### Non-Qwen depth/trigger transfer

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
in `src/llb/bench/agentic_memory_transfer.py`, aggregate rendering/persistence in
`src/llb/bench/agentic_memory_transfer_report.py`, and the CLI in
`src/llb/cli/bench/category_agentic_memory_transfer.py`. Focused contracts are in
`tests/llb/bench/test_agentic_memory_transfer.py`; token-chain task construction shares
`src/llb/bench/agentic_memory_transcript.py` with the memory-dependent lane.

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

##### Second-family replication and cap-fitting boundary

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
`src/llb/bench/agentic_memory_transfer_cells.py`; the original transfer runner now reuses that
module. Replication validation, execution, and analysis live in
`src/llb/bench/agentic_memory_replication.py`, rendering and persistence in
`src/llb/bench/agentic_memory_replication_report.py`, the CLI in
`src/llb/cli/bench/category_agentic_memory_replication.py`, and focused contracts in
`tests/llb/bench/test_agentic_memory_replication.py`. The boundary analyzer records a
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

##### Cap-fitting boundary surface

`make bench-agentic-context-compact-memory-boundary-surface` answers what one cap-fitting cell
cannot: WHERE compact stops repaying its summary call. It pins the family the replication qualified,
holds the typed memory tasks, observation cap, `compact_share`, activation floor, task count, and
summarizer-inclusive accounting fixed, and varies ONLY transcript depth and the prompt guard over a
predeclared grid of cap-fitting cells. The committed design is
`samples/benchmarks/agentic_compact_memory_boundary_surface_design.json`, and the pinned family is
re-qualified against the unchanged token-chain control before any cell runs.

A cap-fitting cell is usable only inside a narrow band, and that band needs NO model to compute. The
memory-dependent tool world is deterministic, so an oracle controller that always plays the next
workflow token reproduces the exact prompt sequence a perfect controller would send
(`src/llb/bench/agentic_memory_boundary_probe.py`): depth 6 peaks at 8,374 prompt chars and depth 10
at 11,926, and the probe's cap totals (13,258 and 27,343 model-input tokens per task) are the
numbers the host then measured. A guard BELOW the peak overflows cap; a guard at or above
`peak / compact_share` never lets compact fire. Design validation refuses any cell outside that open
band, a depth that does not predeclare cells on both sides of the crossover, a grid that drops the
replication's anchor geometry, and a declared window too narrow to carry the widest guard -- all in
CI, with no GPU.

The interpolation rule is predeclared with the grid: read the compact-minus-cap total model-input
delta on the guard axis, take the FIRST adjacent pair of cost-separated cells whose means have
opposite signs, and interpolate linearly to the zero crossing. A cell whose cost sign is not
readable is skipped rather than blocking a bracket around it, and a depth with no sign change
reports a BOUND (the crossing lies above or below the tested guards) instead of extrapolating. Every
cell must also keep its preconditions -- zero cap overflows, zero compact overflows, compaction
above the activation floor, paired completion, and both policies above the cell completion floor --
or it is reported invalid with the named reason instead of bending the crossover.

Core locations are `src/llb/bench/agentic_memory_boundary_probe.py` (oracle probe and usable band),
`src/llb/bench/agentic_memory_boundary_gate.py` (the direction-aware lower-is-better cost gate,
shared with the replication above), `src/llb/bench/agentic_memory_boundary_surface_cells.py` (grid
contract and per-cell validity), `src/llb/bench/agentic_memory_boundary_crossover.py` (the
interpolation and the routing lines), `src/llb/bench/agentic_memory_boundary_surface.py` (design,
run, and analysis), `src/llb/bench/agentic_memory_boundary_surface_report.py`,
`src/llb/cli/bench/category_agentic_memory_boundary_surface.py`, and
`tests/llb/bench/test_agentic_memory_boundary_surface.py`, whose fake-model pass over the committed
grid proves every predeclared cell keeps cap fitting and compact firing.

```bash
make bench-agentic-context-compact-memory-boundary-surface
```

CUDA host evidence (2026-08-02, RTX 4060 Ti 16 GB): `mistral-small3.1:24b` on Ollama with
`num_ctx=8192`, seven depth-matched memory tasks per cell, `compact_share=0.5`, an 800-char
observation cap, 88 episodes at 11.21 tok/s over about 50 minutes. The pinned family re-passed the
unchanged depth-10 control at 4/4. Every cell completed 7/7 under both policies with zero overflows
and exactly one compaction per compact episode, so each cost delta is one summary call against
smaller later controller prompts. The aggregate is
`$DATA_DIR/agentic-compact-memory-boundary-surface/20260802T154634.305722Z-c668820b6c4d/manifest.json`;
its source cell bundles are under `.data/agentic-compact-vs-cap/20260802T1506*` through
`.data/agentic-compact-vs-cap/20260802T1546*`.

| cell | depth | guard | cap input tok | compact input tok | paired d(input tok) | cost pairs | side |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `surface-d6-g12000` | 6 | 12000 | 13258.0 | **12466.0** | -792.0 [-815.3, -762.0] | 0/7 | compact |
| `surface-d6-g14000` | 6 | 14000 | 13258.0 | **13132.1** | -125.9 [-138.4, -112.7] | 0/7 | compact |
| `surface-d6-g15500` | 6 | 15500 | **13258.0** | 14312.6 | +1054.6 [+1049.9, +1061.7] | 7/0 | cap |
| `surface-d10-g14000` | 10 | 14000 | 27343.0 | **22884.7** | -4458.3 [-4501.4, -4411.4] | 0/7 | compact |
| `surface-d10-g20000` | 10 | 20000 | 27343.0 | **24709.6** | -2633.4 [-2652.7, -2614.9] | 0/7 | compact |
| `surface-d10-g23000` | 10 | 23000 | **27343.0** | 28867.9 | +1524.9 [+1517.3, +1533.9] | 7/0 | cap |

Every cost row is unanimous across its seven pairs (two-sided exact sign-test p = 0.015625), every
completion delta is +0.000, and compact adds exactly +1.000 model call per task in all six cells.
All six cells landed on the side the design predeclared. The interpolated crossings are:

| depth | cap peak prompt | bracket | crossover guard | guard / peak |
| ---: | ---: | --- | ---: | ---: |
| 6 | 8374 | [14000, 15500] | 14160 | 1.69 |
| 10 | 11926 | [20000, 23000] | 21900 | 1.84 |

Routing rule: for memory-dependent transcripts, use `compact` while the prompt guard is below about
1.7-1.8x the transcript's cap peak prompt, and `observation_cap` above it. Verdict: **surface
mapped**. The replication's single cap-fitting cell was not a universal result and not a knife-edge
one either -- it sits inside a measured compact-cheaper region that ends at a crossover both tested
depths agree on in peak-relative terms (spread 0.15x). The mechanism is visible in the numbers: cap
costs exactly what the deterministic probe says regardless of guard, while compact's cost rises with
the guard because a later trigger folds a bigger transcript and leaves fewer steps to spend the
smaller prompt on. This does not change the shipped default; the medium-search shape still prefers
`observation_cap`.

##### The routing rule lives on the trigger axis

`make bench-agentic-context-compact-trigger-collapse` closes the axis question the surface leaves
open. The surface swept the prompt guard at ONE `compact_share`, but the policy never reads the
guard directly: it folds when the prompt crosses `int(guard * compact_share)`, so the reported
guard is a stand-in for that trigger. The study measures the difference -- FAMILIES of cells that
hold the trigger fixed while moving share and guard inversely, plus one contrast family that holds
the guard at 12,000 chars and moves the trigger, which is the positive control that the measurement
can see a trigger change at all. The committed design is
`samples/benchmarks/agentic_compact_trigger_guard_collapse_design.json`.

Equivalence is predeclared on the scale the operator pays rather than on interval overlap: the
paired intervals here are tighter than any difference worth acting on, so overlap would reject a
practical equivalence over a few tokens. A family collapses when its SPREAD of compact-minus-cap
total model-input tokens stays within 2% of what the cap baseline costs at that depth AND every
member lands on the same cost side; the contrast family must EXCEED that same band or the study
reports `no_resolving_power` instead of a collapse. The probe also predicts, with no model, which
step each trigger folds at (`first_fold_step` over the deterministic cap prompt sequence), which is
the mechanism the claim rests on.

Core locations are `src/llb/bench/agentic_memory_trigger_collapse_design.py` (family/axis contract
and the cap-fitting band per pair), `src/llb/bench/agentic_memory_trigger_collapse_reading.py`
(vocabulary, fold-step annotation, family spread, and the reading),
`src/llb/bench/agentic_memory_trigger_collapse.py` (run and analysis),
`src/llb/bench/agentic_memory_trigger_collapse_report.py`,
`src/llb/cli/bench/category_agentic_memory_trigger_collapse.py`, and
`tests/llb/bench/test_agentic_memory_trigger_collapse.py`.

```bash
make bench-agentic-context-compact-trigger-collapse
```

CUDA host evidence (2026-08-02, RTX 4060 Ti 16 GB): `mistral-small3.1:24b` on Ollama with
`num_ctx=8192`, the same seven memory tasks per cell, eight cells at 10.64 tok/s over about an hour,
the pinned family re-passing the control at 4/4. Every cell completed 7/7 under both policies with
zero overflows, one compaction per compact episode, and +1.000 model calls per task. The
fold-annotated aggregate is
`$DATA_DIR/agentic-compact-trigger-guard-collapse/20260802T171326.479910Z-eed680be10aa/manifest.json`.

| family | kind | depth | share / guard | trigger | fold step | d(input tok) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `d6-trigger-7000` | equal trigger | 6 | 0.40 / 17500 | 7000 | 6 | -125.9 |
| `d6-trigger-7000` | equal trigger | 6 | 0.50 / 14000 | 7000 | 6 | -125.9 |
| `d6-trigger-7000` | equal trigger | 6 | 0.60 / 11667 | 7000 | 6 | -125.9 |
| `d6-guard-12000` | equal guard | 6 | 0.40 / 12000 | 4800 | 4 | -873.6 |
| `d6-guard-12000` | equal guard | 6 | 0.50 / 12000 | 6000 | 5 | -792.0 |
| `d6-guard-12000` | equal guard | 6 | 0.60 / 12000 | 7200 | 6 | -125.9 |
| `d10-trigger-10000` | equal trigger | 10 | 0.45 / 22224 | 10000 | 9 | -2633.4 |
| `d10-trigger-10000` | equal trigger | 10 | 0.60 / 16667 | 10000 | 9 | -2633.4 |

Both equal-trigger families have a spread of **0.0 tokens** -- bit-identical deltas across a 1.5x
and a 1.3x range of prompt guards -- against bands of 265.2 and 546.9 tokens, while the contrast
family moved 747.7 tokens across the same band. Verdict: **the trigger ratio collapses the
surface**. Share and guard act only through their product, so a crossover measured at one share
converts to any other: at depth 6 the 14,160-char crossover guard is a 7,080-char trigger (0.85x the
cap peak prompt) and at depth 10 the 21,900-char guard is a 10,950-char trigger (0.92x), giving the
portable form -- use `compact` while `compact_share * guard` stays below about 0.85-0.92x the
transcript's cap peak prompt, and `observation_cap` above it.

The fold step says why, and it is the more useful statement of the rule: the trigger reaches the
transcript ONLY by choosing which step compacts, so the cost delta is a STEP function of the
trigger, not a smooth one. The contrast family's 7,200-char trigger folds at step 6 exactly like the
7,000-char triggers and reproduces their -125.9 to the token; its 4,800- and 6,000-char triggers
fold at steps 4 and 5 and cost -873.6 and -792.0. Two independent runs agree to the token as well:
the surface's own (0.5, 14000) and (0.5, 20000) cells are the trigger-7000 and trigger-10000 values
above. An operator therefore does not need a grid for a new window -- only the trigger that lands on
the intended fold step. The shipped `compact_share` is unchanged.

##### The crossover is a fold step, not a char guard

`make bench-agentic-context-compact-fold-step` restates the crossover on the axis the mechanism
actually has. The surface interpolated a char guard, and the trigger collapse then showed why that
number cannot be read literally: the trigger reaches the transcript ONLY by choosing which step
folds, so every trigger inside one step's interval produces the identical transcript. The crossover
is the boundary between two fold steps, and the interpolated char value is an artifact of fitting a
continuous rule to a discrete mechanism. The committed design is
`samples/benchmarks/agentic_compact_fold_step_crossover_design.json`.

The geometry is the inverse of the fold-step prediction and needs no model.
`fold_step_trigger_interval` returns the half-open `[low, high)` trigger interval whose every value
folds at one step -- `low` is the largest earlier step prompt, `high` is the step's own prompt --
and `fold_step_guard_interval` converts it to prompt guards through the runtime's own truncating
`int(guard * share)` rather than a float inverse. A step whose prompt does not exceed the running
maximum before it is UNREACHABLE (no trigger selects it), so `reachable_fold_steps` is the ladder a
design is placed against.

Placement is what lets the grid tell a step change apart from a smooth slide, and all four rules are
checked in CI with no GPU: every declared cell must fold at the step it claims, the tested steps must
be ADJACENT on the reachable ladder, the guards inside one step must span at least half of that
step's guard interval (otherwise "same step, same cost" is measured over two nearly identical
guards), and the guards on either side of a step change must straddle it within 8 chars (otherwise
the flip is localized no better than the old bracket was). Cell preconditions -- cap fits, compact
fires above the activation floor, completion paired -- are the surface's unchanged gate, and a cell
whose measured fold step drifts from its declared one aborts the analysis rather than being re-read.

Core locations are `src/llb/bench/agentic_memory_boundary_probe.py` (the trigger/guard interval
inverse, the reachable ladder, and `compaction_trigger_chars`, now shared with the collapse study),
`src/llb/bench/agentic_memory_fold_step_design.py` (the placement contract),
`src/llb/bench/agentic_memory_fold_step_rows.py` (step and depth rows),
`src/llb/bench/agentic_memory_fold_step_reading.py` (vocabulary, readings, routing lines),
`src/llb/bench/agentic_memory_fold_step.py` (run and analysis),
`src/llb/bench/agentic_memory_fold_step_report.py`,
`src/llb/cli/bench/category_agentic_memory_fold_step.py`, and
`tests/llb/bench/test_agentic_memory_fold_step_crossover.py`.

```bash
make bench-agentic-context-compact-fold-step
```

CUDA host evidence (2026-08-02, RTX 4060 Ti 16 GB): `mistral-small3.1:24b` on Ollama with
`num_ctx=8192`, seven depth-matched memory tasks per cell, `compact_share=0.5`, eight cells at
10.56 tok/s over about 68 minutes. The pinned family re-passed the unchanged depth-10 control at
4/4; every cell
completed 7/7 under both policies with zero overflows, one compaction per compact episode, and all
eight landed on the side the design predeclared. The aggregate is
`$DATA_DIR/agentic-compact-fold-step-crossover/20260802T185212.038607Z-24e73063cba6/manifest.json`.

| cell | depth | guard | trigger | fold step | cap tok | compact tok | d(input tok) | side |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fold-d6-step6-lo` | 6 | 13136 | 6568 | 6 | 13258.0 | **13132.1** | -125.9 | compact |
| `fold-d6-step6-hi` | 6 | 14910 | 7455 | 6 | 13258.0 | **13132.1** | -125.9 | compact |
| `fold-d6-step7-lo` | 6 | 14912 | 7456 | 7 | **13258.0** | 14312.6 | +1054.6 | cap |
| `fold-d6-step7-hi` | 6 | 16746 | 8373 | 7 | **13258.0** | 14312.6 | +1054.6 | cap |
| `fold-d10-step10-lo` | 10 | 20240 | 10120 | 10 | 27343.0 | **26434.4** | -908.6 | compact |
| `fold-d10-step10-hi` | 10 | 22014 | 11007 | 10 | 27343.0 | **26541.0** | -802.0 | compact |
| `fold-d10-step11-lo` | 10 | 22016 | 11008 | 11 | **27343.0** | 28698.4 | +1355.4 | cap |
| `fold-d10-step11-hi` | 10 | 23040 | 11520 | 11 | **27343.0** | 28878.6 | +1535.6 | cap |

Verdict: **fold-step boundary confirmed** at both depths. The cost side changes between two guards
**2 chars apart** -- 14910 and 14912 at depth 6, 22014 and 22016 at depth 10 -- while guards up to
1834 chars apart inside one step stay on the same side. The step change moves 1180.4 tokens
at depth 6 and 2300.8 at depth 10, against within-step bands of 265.2 and 546.9.

| depth | last compact-cheaper fold step | trigger interval | guard interval (share 0.5) | within-step residual |
| ---: | ---: | --- | --- | ---: |
| 6 | 6 | `[6568, 7456)` | `[13136, 14912)` | 0.0 tok |
| 10 | 10 | `[10120, 11008)` | `[20240, 22016)` | 180.1 tok |

The routing rule an operator applies exactly: **fold no later than step k** -- keep
`compact_share * guard` below step k's own cap prompt (7456 chars at depth 6, 11008 at depth 10),
which at `compact_share=0.5` is a guard below 14912 and 22016 chars. Both interpolated crossovers
land INSIDE the cheap step and name a point at which nothing changes: 14160 sits 752 chars below the
depth-6 step change (5.0% low) and 21900 sits 116 chars below the depth-10 one. Neither is wrong as
an approximation -- both fall in the correct step -- but only the step boundary is where the cost
actually moves, and only it converts to another `compact_share` without re-deriving anything.

One term survives inside a step, and the run isolates it. At depth 6 the whole cost is bit-identical
across the guard interval; at depth 10 it moves 180.1 tokens, of which 171.0 is the summarize call
and 9.1 is later controller prompts. The cause is that the summarize call's input cap was the trigger
at the time of this run, so a larger trigger inside one step fed the summarizer more of the folded
transcript -- and the summary it returned was then carried by every later prompt. Depth 6 folds a
transcript smaller than either cap, so nothing was trimmed and the residual is exactly zero. The
residual is 8% of the depth-10 step change and stays far inside the equivalence band, so it does not
move the boundary; each cell row records `compact_mean_controller_prompt_tokens` and
`compact_mean_compaction_prompt_tokens` so the split is readable rather than inferred. This does not
change the shipped `compact_share` or the guard-axis interpolation the surface publishes. The bound
that produced the residual is what
[the summarize-input cap](#the-summarize-input-cap-is-step-aligned) then replaced; this study's
design pins `summary_input_cap: "trigger"` so the numbers above reproduce unchanged.

##### The summarize-input cap is step-aligned

`make bench-agentic-context-compact-summary-input-cap` closes the one term the fold-step study left
moving. The compact policy has to bound the summarize call's input -- that input is the transcript
that just blew the step prompt, so an uncapped summarizer is the one call in the loop guaranteed to
overflow -- but the bound it used, the compaction trigger, is the ONLY part of the compact cost that
is not a step function of the fold step. Two guards inside one step fold the identical transcript and
send bit-identical controller prompts, yet feed the summarizer different amounts of it, and the
summary that comes back is then carried by every later prompt. The bound also ELIDES the folded
transcript head-and-tail once it outgrows the trigger, so a transcript that would have fit the window
was summarized with its middle missing.

The shipped bound is now `summary_input_cap="window"`: the resolved prompt budget minus the summary
template's own overhead, which includes the elision marker `trim_observation` writes ON TOP of the
cap it is given (a bound that ignores the marker sends a summarize prompt a few chars over the
window -- exactly the silent truncation the cap exists to prevent). It is a property of the resolved
budget alone, so it does not move with `compact_share` and the folded transcript is summarized at its
own size whenever it fits. The legacy `trigger` bound stays selectable, and the boundary-surface,
trigger-collapse, replication, transfer, and fold-step designs all pin it explicitly so their
published numbers reproduce against the current runtime instead of silently re-measuring a different
summarizer. The committed design is
`samples/benchmarks/agentic_compact_summary_input_cap_design.json`.

The study is two ARMS over ONE fold-step ladder -- the same depth-10 ladder the fold-step crossover
published, with the two bounds as the only difference -- and it reads two independent things: whether
the step-aligned bound drives the within-step residual to zero WITHOUT moving the fold step the
routing rule is stated on, and whether the span the trigger bound elided was carrying completion.
The second question needs an elision to exist, and that is decided with no model at all:
`compact_fold_input_probe` walks the deterministic tool world with an oracle controller and a fixed
summary reply, and reports what each arm offers the summarizer and how much its bound elides. Design
validation refuses a ladder whose reference arm elides nothing (no trimmed span to price), a
step-aligned arm that elides anything (it is not step-aligned), and a step-aligned arm whose
summarize input is not identical across the guards inside one step. The fold-step placement rules --
declared step, adjacency on the reachable ladder, within-step guard span, straddle gap -- are shared
verbatim with the crossover study (`src/llb/bench/agentic_memory_fold_step_placement.py`), so a
residual measured here is on exactly the scale that study publishes.

Core locations are `src/llb/bench/agentic/context.py` (`SUMMARY_INPUT_CAPS`, the elision telemetry),
`src/llb/bench/agentic/context_budget.py` (`summary_input_cap_chars`),
`src/llb/bench/agentic/episode.py` (the bound resolver),
`src/llb/bench/agentic_memory_boundary_probe.py` (`compact_fold_input_probe`),
`src/llb/bench/agentic_memory_fold_step_placement.py` (shared placement rules),
`src/llb/bench/agentic_memory_summary_cap_design.py`,
`src/llb/bench/agentic_memory_summary_cap_reading.py`,
`src/llb/bench/agentic_memory_summary_cap_rows.py`,
`src/llb/bench/agentic_memory_summary_cap.py`,
`src/llb/bench/agentic_memory_summary_cap_report.py`,
`src/llb/cli/bench/category_agentic_memory_summary_cap.py`, and
`tests/llb/bench/test_agentic_memory_summary_cap.py`.

```bash
make bench-agentic-context-compact-summary-input-cap
```

CUDA host evidence (2026-08-05, RTX 4060 Ti 16 GB): `mistral-small3.1:24b` on Ollama with
`num_ctx=8192`, the fold-step study's seven depth-10 memory tasks per cell, `compact_share=0.5`,
eight cells (four guards under each bound) at 10.63 tok/s over about 79 minutes. The pinned family
re-passed the unchanged depth-10 control at 4/4. Every cell completed 7/7 under both policies with
zero overflows, exactly one compaction per compact episode, and all eight landed on the side the
design predeclared. The aggregate is
`$DATA_DIR/agentic-compact-summary-input-cap/20260805T185837.832318Z-0f86b57558a1/manifest.json`.

| arm | cell | guard | fold step | summarizer offered | elided | compact tok | d(input tok) | side |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `trigger` | `cap-d10-step10-lo` | 20240 | 10 | 10494 | **374** | 26434.4 | -908.6 | compact |
| `trigger` | `cap-d10-step10-hi` | 22014 | 10 | 10494 | 0 | 26541.0 | -802.0 | compact |
| `trigger` | `cap-d10-step11-lo` | 22016 | 11 | 11802 | **794** | 28698.4 | +1355.4 | cap |
| `trigger` | `cap-d10-step11-hi` | 23040 | 11 | 11802 | **282** | 28878.6 | +1535.6 | cap |
| `window` | `cap-d10-step10-lo` | 20240 | 10 | 10494 | 0 | **26541.0** | -802.0 | compact |
| `window` | `cap-d10-step10-hi` | 22014 | 10 | 10494 | 0 | **26541.0** | -802.0 | compact |
| `window` | `cap-d10-step11-lo` | 22016 | 11 | 11802 | 0 | **28953.3** | +1610.3 | cap |
| `window` | `cap-d10-step11-hi` | 23040 | 11 | 11802 | 0 | **28953.3** | +1610.3 | cap |

Verdict: **the step-aligned cap is an exact step function**. The within-step residual goes from
180.1 tokens to **exactly 0.0** -- both guards inside fold step 10 and both inside fold step 11 now
cost the same to the token, controller and summarizer alike -- while the boundary the routing rule is
stated on stays at fold step 10 and the step change grows slightly, from 2300.8 to 2412.3 tokens
against the same 546.9-token band. Every measured summarizer input and elided span reproduces the
model-free probe's prediction to the character, so the mechanism was settled before the GPU ran and
the run only confirmed it costs what the geometry says.

| arm | step-10 spread | step-11 spread | residual (summarizer / controller) | last compact-cheaper step |
| --- | ---: | ---: | --- | ---: |
| `trigger` | 106.6 | 180.1 | 180.1 (171.0 / 9.1) | 10 |
| `window` | **0.0** | **0.0** | **0.0 (0.0 / 0.0)** | 10 |

The elision was free: the reference arm cut up to 794 chars out of the summarizer's input and the
paired compact completion between the arms is +0.000 [+0.000, +0.000] over 28 pairs (0 wins, 0
losses, 28 ties, sign-test p = 1.0000) -- a `flat` reading, so the trimmed span carried nothing the
summary needed on this shape. Pin the cap for predictability, not for completion.

What the trigger cap WAS doing is visible in the compact column: a trimmed summarize input is a
smaller prompt, so the elision quietly discounted compact's own measured cost -- by 106.6 tokens at
fold step 10 and 180.1 at fold step 11, always in compact's favor, and always at the cells the
routing rule is read from. The `window` numbers are the undiscounted ones. Both arms still land on
the predeclared sides at every guard, so the depth-10 fold-step crossover is unchanged; what that
discount does to every OTHER published crossover is settled in
[the restatement](#published-crossovers-under-the-shipped-cap) below.

##### Published crossovers under the shipped cap

`make bench-agentic-context-compact-crossover-restatement` answers the question the step-aligned
bound leaves behind: every compact routing number an operator applies was measured under the retired
trigger bound, which discounted compact's own cost wherever it actually trimmed the folded
transcript. Re-running four studies to find out where that mattered would be the expensive answer.
The cheap one is exact.

The bound reaches a run through exactly ONE prompt -- the summarize call -- so a cell whose
summarize input is identical under both bounds sends bit-identical prompts under both, and its
published cost cannot have moved. `compact_fold_input_probe` decides that per cell with no model, so
the study opens with a model-free AUDIT of every published cell in the boundary surface, the trigger
collapse, and the fold-step crossover, and re-measures only the cells the audit calls
bound-sensitive. The committed design is
`samples/benchmarks/agentic_compact_crossover_restatement_design.json`; it names each audited study
by its in-repo design path and every crossover that study published, and validation refuses a design
whose path is missing, declares a different study kind, publishes a crossover at a depth the study
does not test, or omits the fold step a crossover lands in.

The invariance criterion is the fold step, not a char tolerance. The fold-step study established
that the cost changes only at a step boundary, so a restated guard that moves INSIDE one step's
guard interval names a point at which nothing changes; only a guard that crosses a step boundary
withdraws a published number. `--audit-only` reports the audit and stops, which is the GPU-free way
to ask "does this bound change invalidate my evidence" before spending anything.

Core locations are `src/llb/bench/agentic_memory_cap_audit.py` (geometry extraction per study shape,
both-bound probe, invariance verdict), `src/llb/bench/agentic_memory_crossover_restatement_design.py`,
`src/llb/bench/agentic_memory_crossover_restatement_reading.py`,
`src/llb/bench/agentic_memory_crossover_restatement_rows.py` (substitute, re-interpolate, and place
the restated guard on the step ladder), `src/llb/bench/agentic_memory_crossover_restatement.py`,
`src/llb/bench/agentic_memory_crossover_restatement_report.py`,
`src/llb/cli/bench/category_agentic_memory_crossover_restatement.py`, and
`tests/llb/bench/test_agentic_memory_crossover_restatement.py`.

```bash
make bench-agentic-context-compact-crossover-restatement
make bench-agentic-context-compact-crossover-restatement AGENT_CONTEXT_COMPACT_CROSSOVER_AUDIT_ONLY=1
```

The audit is the result. Of the 22 published cells across the three studies, **18 are
bit-identical under both bounds** and needed no run at all: every depth-6 cell folds a transcript
neither bound trims, and so do most depth-10 cells. Four are bound-sensitive, and three of those are
the depth-10 fold-step cells the summarize-input-cap study had already re-measured. **One cell**
(`surface-d10-g23000`, 302 chars elided) was left, so the whole GPU cost of restating four studies'
worth of routing numbers was 14 episodes.

CUDA host evidence (2026-08-05, RTX 4060 Ti 16 GB): `mistral-small3.1:24b` on Ollama with
`num_ctx=8192`, seven depth-10 memory tasks, `compact_share=0.5`, one re-measured cell at 10.55
tok/s over about 12 minutes including the control. The pinned family re-passed the unchanged
depth-10 control at 4/4; the re-measured cell completed 7/7 under both policies with zero overflows
and one compaction per compact episode. The aggregate is
`$DATA_DIR/agentic-compact-crossover-restatement/20260805T192757.795491Z-2bc079197412/manifest.json`.

| study | depth | form | published | restated | fold step | basis |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| boundary surface | 6 | interpolated guard | 14160 | unchanged | 7 | every cell bound-invariant |
| boundary surface | 10 | interpolated guard | 21900 | **21862** | 10 | re-measured cell |
| trigger collapse | 6 | portable ratio | 0.85x | unchanged | 6 | every cell bound-invariant |
| trigger collapse | 10 | portable ratio | 0.92x | 0.92x | 10 | every cell bound-invariant |
| fold step | 6 | fold-step boundary | 14912 | unchanged | 6 | every cell bound-invariant |
| fold step | 10 | fold-step boundary | 22016 | unchanged | 10 | already re-measured |

Verdict: **every published crossover holds under the shipped cap**. Exactly one number moves at all
-- the depth-10 interpolated guard, by **-38 chars** (21900 -> 21862, ratio 1.84 -> 1.83) once the
re-measured cell's undiscounted cost (+1610.3 instead of +1524.9 tokens) enters the interpolation --
and it lands at the same place on the ladder, inside fold step 10's guard interval
`[20240, 22016)` where every guard costs the same. The direction is the one the mechanism predicts:
removing a discount that flattered compact pulls the crossing DOWN, toward compact being preferred
over a slightly narrower band of guards.

The re-measured cell also cross-checks the step function across two independent runs on different
days: guard 23000 here costs 28953.3 compact tokens, the identical value guards 22016 and 23040
produced in the summarize-input-cap study. Three guards spanning 1024 chars, one fold step, the same
cost to the token.

The collapse's portable ratio needs one extra step to read, because it is DERIVED from the surface's
interpolated guard rather than measured directly: at depth 10 the restated 21862-char guard is a
10931-char trigger against an 11926-char cap peak, so the ratio moves 0.918x -> 0.917x and the
published 0.85-0.92x band is unchanged at the precision it is stated in. The collapse's own eight
cells are all bound-invariant, so the equal-trigger spreads and the contrast family stand as
measured.

The trigger collapse gains something from the change rather than merely surviving it. Its claim is
that `compact_share` and the prompt guard act ONLY through their product, and the retired bound was
the one place where share entered independently (the summarize input was capped at
`compact_share * guard`). Under the shipped bound that term is gone, so the collapse holds by
construction and not only by measurement.

##### What a policy-constant change invalidates

`make bench-agentic-policy-change-audit` generalizes the mechanism above from ONE bound to any agent
context-policy constant. A context policy is a pure function of the deterministic tool world, so
fixing the geometry and the controller fixes the exact sequence of prompts an episode sends before
any model runs. The audit therefore replays every published cell under both values of the changed
field with an oracle controller, records every prompt each replay sends -- controller prompts and
summarize calls alike, by recording through the injected `complete`, which is the seam they all pass
through -- and compares the sequences byte for byte.

Two properties make a replay a statement about a real run. The summarize call is answered with a
FIXED summary so the replay is deterministic, which can only hide downstream divergence, never
invent it: identical summarize prompts mean a temperature-0 model returns the same summary, so the
later controller prompts are identical too, and "all prompts identical under the replay" implies
"all prompts identical under the served model". And BOTH arms of a cell are replayed
(`observation_cap` and `compact`), because a published number is a compact-minus-cap delta and a
change that moves either arm moves it.

One case needs its own verdict rather than a comparison. A cell that declares the audited field
ITSELF -- the trigger collapse sweeps `compact_share` cell by cell -- is not describable by the
change: replaying it at another value measures a different cell, not the published one. Those report
`cell_pins_the_field` and are excluded from the counts. A value inherited from `held_fixed` is not
the same thing; that is the study's inherited setting, and whether its number holds at another value
is exactly the counterfactual the audit answers.

Core locations are `src/llb/bench/agentic_policy_change_replay.py` (replay, digest, and the
per-arm comparison), `src/llb/bench/agentic_policy_change_audit.py` (the auditable fields, the
per-study cell geometry, and the verdict),
`src/llb/bench/agentic_policy_change_audit_report.py`,
`src/llb/cli/bench/category_agentic_policy_change_audit.py`, and
`tests/llb/bench/test_agentic_policy_change_audit.py`. The summarize-bound audit
(`src/llb/bench/agentic_memory_cap_audit.py`) is now ONE use of this mechanism rather than a second
one: it supplies the elision diagnostic that explains the verdict, and CI asserts the two agree cell
for cell.

```bash
make bench-agentic-policy-change-audit \
  POLICY_FIELD=observation_cap_chars POLICY_BASELINE=800 POLICY_CANDIDATE=1600
```

Every auditable field against the 22 published cells of the three cap-fitting studies (2026-08-05,
no GPU, about 0.7 s per field; audits land under `$DATA_DIR/agentic-policy-change-audit/<run>/`):

| field | change | invariant | invalidated | not applicable |
| --- | --- | ---: | ---: | ---: |
| `observation_cap_chars` | 800 -> 400 | 0 | **22** | 0 |
| `observation_cap_chars` | 800 -> 1600 | 0 | **22** | 0 |
| `observation_head_share` | 0.6 -> 0.5 | 0 | **22** | 0 |
| `keep_last_n` | 3 -> 1 | **22** | **0** | 0 |
| `compact_share` | 0.5 -> 0.45 | 2 | 12 | 8 |
| `compact_keep_recent` | 1 -> 2 | 0 | **22** | 0 |
| `summary_input_cap` | trigger -> window | **18** | 4 | 0 |

Two readings an operator can act on. **`keep_last_n` is free**: the constant sweep EXPOSES keep=1 as
cheaper on prompt tokens, and this says taking that up costs no published compact evidence at all,
because no cap-fitting cell runs the `keep_last_n` policy. **The observation-trim constants are
not**: `observation_cap_chars` and `observation_head_share` change both arms of every cell from
model call 2 -- the first prompt that carries a trimmed observation -- so re-pinning either one
retires all three studies at once.

The `compact_share` row also reproduces the fold-step mechanism from a direction that owes it
nothing. Of the 14 applicable cells, the two that survive 0.5 -> 0.45 are `fold-d6-step6-hi` and
`fold-d6-step7-hi` -- the HIGH guard in each fold step, where a smaller share still lands the trigger
inside the same step's interval and folds the identical transcript. At the low guards the trigger
drops into the previous step and everything downstream changes. A byte-level prompt comparison that
knows nothing about fold steps rediscovers exactly where they are.

### Agent context-policy constants

Three constants decide what `observation_cap` and `keep_last_n` do:
`DEFAULT_OBSERVATION_CAP_CHARS` (800), `OBSERVATION_HEAD_SHARE` (0.6), and `DEFAULT_KEEP_LAST_N`
(3). All three are CLI / Make knobs (`--observation-cap-chars`, `--observation-head-share`,
`--keep-last-n` / `AGENT_CONTEXT_*`). `make bench-agentic-context-sweep` walks one-dimensional
grids for each axis, pairs every non-shipped cell against the shipped value over shared bootstrap
index sets, and states a pin / expose / inapplicable verdict per constant without rewriting the
defaults.

```bash
make bench-agentic-context-sweep MODEL=<model> BACKEND=<backend> \
  AGENT_CONTEXT_SWEEP_TASKS=<tasks.json> AGENT_CONTEXT_SWEEP_MAX_MODEL_LEN=8192
```

Core locations: `src/llb/bench/agentic_context_sweep.py` (grids, pairing, verdicts),
`src/llb/cli/bench/category_agentic_context_sweep.py`, CI span arithmetic in
`tests/llb/bench/test_agentic_context_sweep.py`. Bundles land under
`$DATA_DIR/agentic-context-sweep/<run>/` (one per setting plus a summary manifest).

CUDA host evidence (2026-07-30, RTX 4060 Ti 16 GB): `MamayLM-Gemma-3-12B-IT-v2.0` on Ollama,
`--max-model-len 8192` (prompt budget 21504 chars), the same 24-task large-observation set, 8
unique cells (shipped `observation_cap` shared across the cap and head-share baselines), ~73 min,
562 calls at 3.8 tok/s. Summary under `.data/agentic-context-sweep/20260730T125612*`.

| axis | setting | completion | mean prompt tok | overflow | d(compl) vs shipped | d(prompt) vs shipped |
| --- | --- | ---: | ---: | ---: | --- | --- |
| cap | 400 | 0.708 | 1280 | 0 | -0.167 [-0.333, -0.042] | -83 [-283, +104] |
| cap | **800** | **0.875** | 1362 | 0 | baseline | baseline |
| cap | 1600 | 0.875 | 1323 | 0 | +0.000 [+0.000, +0.000] | -40 [-206, +135] |
| head | 0.5 | 0.875 | 1305 | 0 | +0.000 [+0.000, +0.000] | -58 [-225, +91] |
| head | **0.6** | **0.875** | 1362 | 0 | baseline | baseline |
| head | 0.7 | 0.875 | 1426 | 0 | +0.000 [+0.000, +0.000] | +63 [+0.000, +190] |
| keep | 1 | 0.500 | 3908 | 8 | +0.042 [+0.000, +0.125] | **-474 [-1145, -21]** |
| keep | 2 | 0.500 | 4379 | 10 | +0.042 [+0.000, +0.125] | -3 [-6, -0.5] |
| keep | **3** | 0.458 | 4382 | 10 | baseline | baseline |

Verdicts:

- **pin** `observation_cap_chars=800`: cap=400 separates worse on completion; cap=1600 is flat.
  Keep the measured default.
- **pin** `observation_head_share=0.6`: 0.5 / 0.7 are flat on completion (prompt deltas do not
  separate clear of zero either). Keep the measured default; the knob stays operator-visible.
- **expose** `keep_last_n=3`: keep=1 is flat on completion but separates cheaper on prompt tokens
  (-474 [-1145, -21]) and drops overflows from 10 to 8. The knob was already CLI-exposed; the
  shipped default stays 3 because the completion reading is flat -- operators who care about
  prompt cost on this short-episode shape can pass `--keep-last-n 1`. The earlier "nearly a
  no-op at keep=3" reading still holds for the shipped cell; keep=1 is the setting that starts
  to reach the blowup.

### keep_last_n on longer transcripts

The short-episode expose of keep=1 was a cost reading on a 6-step fat-observation set. The
follow-up lane asks whether that cheaper cell stays flat on completion once transcripts grow past
the shipped keep. `make bench-agentic-context-keep-long` builds a medium-observation search set
from the fat count/locate tasks (`prepare-agentic-long-transcript --from-search-tasks`, capping
matching/other docs and rebinding success), then sweeps `keep_last_n` alone at
`AGENT_CONTEXT_KEEP_LONG_MAX_STEPS=12`.

```bash
make bench-agentic-context-keep-long MODEL=<model> BACKEND=<backend> \
  AGENT_CONTEXT_SWEEP_MAX_MODEL_LEN=8192
```

Core locations: `src/llb/bench/agentic_long_transcript.py` (medium-search shrink + synthetic
pipelines for CI), `make bench-agentic-context-keep-long`, axes filter
`AGENT_CONTEXT_SWEEP_AXES=keep_last_n`.

CUDA host evidence (2026-07-30, RTX 4060 Ti 16 GB): `MamayLM-Gemma-3-12B-IT-v2.0` on Ollama,
`--max-model-len 8192`, 14 medium search tasks (from the 24-task UA-squad set), keep grid only,
`max_steps=12`, ~41 min, 362 calls at 4.7 tok/s. Bundles under
`.data/agentic-context-sweep/20260730T1855*`.

| setting | completion | mean steps | mean prompt tok | overflow | d(compl) vs keep=3 | d(prompt) vs keep=3 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| keep=1 | **0.643** | 7.86 | 1000 | 0 | +0.214 [+0.000, +0.429] | **-283 [-394, -180]** |
| keep=2 | **0.643** | 8.64 | 1142 | 0 | +0.214 [+0.000, +0.429] | -141 [-197, -90] |
| keep=3 | 0.429 | 9.36 | 1283 | 0 | baseline | baseline |

Verdict: **expose stays the right call on longer transcripts**. Mean steps (7.9-9.4) clear the
shipped keep, so older steps are actually dropped -- this is not the short-episode no-op. keep=1
(and keep=2) stay flat on completion against keep=3 (CI lower bound touches zero) while separating
cheaper on prompt tokens; point completion is higher and mean steps are lower under keep=1, but
without a separated completion win the shipped default stays 3. Operators who already pass
`--keep-last-n 1` for cost on the short set can keep doing so on this longer-transcript shape.

Synthetic file/db pipeline tasks remain in the module for CI over a fake `complete`; live MamayLM
loops on repeated `read_file` for those prompts, so the CUDA evidence path is the medium-search
shrink above.

## Context-Policy Comparison

`bench-chain-context` ranks context-management POLICIES for ONE fixed model over a verified
chain-of-questions set. It holds the model, the chain set, retrieval, and the scoring fixed and
varies only the policy -- the row label -- exactly as the agentic harness comparison holds
everything fixed and varies the harness. It answers "which harness and context policy improves
scores, and how should system prompts be sequenced" over multi-step questions where each step's
answer depends on the prior steps.

Core locations:

- `src/llb/bench/chain_context_policy.py`: policy execution and per-step context assembly;
- `src/llb/bench/chain_context.py`: run orchestration and persistence;
- `src/llb/bench/chain_context_report.py`: result contract, provenance projection, digest, and
  recommendation rendering;
- `src/llb/board/chain_context.py`: one-model policy comparison board rows under
  `TIER_CHAIN_CONTEXT` (mirrors `board/harnesses.py`);
- `src/llb/prompts/templates/bench/chain_context/`: the reviewable role/instruction prompt-system
  templates and the step scaffold.

The four policies (each a fresh retrieval per step PLUS a different memory of the prior steps):

- `fresh` -- fresh retrieval per step, NO prior-step carryover (the naive baseline);
- `history` -- the accumulated full (question, answer) transcript;
- `summary` -- a running model-written summary of the prior steps (one extra model call per step);
- `roles` -- a staged role/system-prompt sequence (librarian -> analyst -> answerer) built from
  the prompt-system role templates, plus the accumulated transcript. First step is the librarian,
  last is the answerer, middle steps are analysts; a 2-step chain is librarian -> answerer.

```bash
llb bench-chain-context --chains <chains.jsonl> --corpus <corpus-dir> \
  --model <model> --backend <backend> --policies fresh,history,summary,roles
make bench-chain-context CHAIN_CONTEXT_MODEL=<model> CHAIN_CONTEXT_BACKEND=<backend>
```

Retrieval reaches the store through the injectable `Retriever` seam (`retrieve(question, k)`), and
the model through the same injectable `complete` (prompt -> raw text) every category uses, so the
exact context assembled per policy per step is unit-tested over a FAKE endpoint with no GPU
(`tests/llb/bench/test_chain_context.py`). Each step's answer is scored objectively against its
reference answer (`scoring.correctness` token-F1); the headline is FINAL-answer correctness per
chain (does the chain end right), with per-step correctness recorded alongside, both with bootstrap
CIs. Each policy persists its OWN run bundle under `$DATA_DIR/chain-context/<timestamp>/` tagged
with the policy (mirroring the per-harness agentic bundles); provenance records the policy, the
`prompt_system_ids` (the role/instruction template ids), and the `chain_set_digest`. Verified-data
stamping (`--data-verified` + `--verification-ref`) follows the same category-suite gate as the
other benchmarks. The board loader ranks all policies together, and `llb recommend` gains a
"Context policy" section naming the best policy per model when bundles exist.

CUDA evidence (2026-07-11, RTX 4060 Ti 16 GB): the committed 20-chain fixture (40 steps) run
through `MamayLM-Gemma-3-12B-IT-v2.0` on Ollama, all four policies in one ~11 min invocation,
reliability 1.000 for each. Final-answer correctness ranked `roles` **0.789** [0.635, 0.915] >
`history` 0.625 > `summary` 0.534 > `fresh` 0.431: the naive no-carryover baseline is worst, any
memory of the prior steps helps, and the staged librarian -> analyst -> answerer sequence wins with
its CI resolved above the rest. Run bundles under `.data/chain-context/20260711T1938*` (one per
policy). The discriminating spread is the whole reason to measure the policy rather than assume one.

## Judge Diagnostics

`src/llb/scoring/judge_diag.py` classifies zero-valued judge outcomes so a diagnostic score can be
read correctly:

- `empty_answer`: candidate produced nothing useful;
- `malformed_judge_json`: judge endpoint failed strict JSON expectations;
- `judge_transport_error`: judge endpoint failed transport;
- `zero_score`: judge returned a valid zero.

`run_gated_judge` attaches diagnostics to category judge outcomes, and category manifests carry the
summary under judge metadata. `bench-agentic` also echoes the diagnostic summary.

Before a long judged run, use:

```bash
llb judge-smoke --judge-model <model> --judge-base-url <url>
```

The smoke check runs one grounded case and exits non-zero with a reason when the judge cannot
return a well-formed non-zero strict-JSON score.

## Prompt-System Packages

`src/llb/prompt_system/` builds reviewable RAG prompt-system candidates from a corpus. The package
is deterministic and manifest-addressable so prompt changes become explicit experiment variables.

Important modules:

- `corpus.py`: reads `.md`/`.txt`, keeps exact spans, selects anthology passages, builds metadata;
- `budget.py`: token-budget planning and section trimming;
- `template.py`: prompt fields and `PromptPackage.apply`;
- `tuning.py`: candidate grid and deduplication;
- `knowledge_tree_source.py`: ontology/graph loading and source identity;
- `knowledge_tree_render.py`: deterministic community ordering and strict depth/token-budget
  rendering;
- `review.py`: approve, pin, reject, and persist candidate review state;
- `manifest.py`: corpus, mapping, template digests, and stable prompt-system ids;
- `selection.py`: resolves a selected package for `run-eval`.

```bash
llb prompt-system-prepare --corpus-root <dir> --out-dir <review-dir>
llb prompt-system-review --run-dir <review-dir> --action summary
llb prompt-system-review --run-dir <review-dir> --action pin --id <prompt-id>
llb run-eval --prompt-system <prompt-id> --prompt-package <review-dir> ...
llb prompt-system-compare --lane rag --model <model>
```

`run-eval` prepends the selected prompt package to the normal RAG generation prompt and records
`prompt_system_provenance` in the manifest. Board loaders can rank one model across prompt-system
ids for RAG or agentic lanes.

Knowledge-tree generation is opt-in and consumes an existing ontology draft bundle, a persisted
graph store, or both. It adds the corpus vocabulary, size-ranked entity communities, and optional
community summaries as a system-prompt block. Every ordinary prompt candidate remains as a
no-tree control; tree candidates record their exact control id plus the source digest, requested
depth, requested/effective token budget, and rendered token count. Tree tokens consume the same
overall prompt allowance as anthology/metadata/mapping context.

```bash
make prompt-system-prepare \
  PROMPT_SYSTEM_CORPUS=<corpus-dir> \
  PROMPT_SYSTEM_GRAPH_DIR=<graph-store> \
  PROMPT_SYSTEM_TREE_DEPTHS=1,2,3 \
  PROMPT_SYSTEM_TREE_BUDGETS=128,256
make run-eval PROMPT_SYSTEM_ID=<control-id> PROMPT_PACKAGE=<review-dir>
make run-eval PROMPT_SYSTEM_ID=<tree-id> PROMPT_PACKAGE=<review-dir>
make prompt-system-compare MODEL=<model> PROMPT_SYSTEM_LANE=rag
```

`PROMPT_SYSTEM_ONTOLOGY_BUNDLE=<draft-bundle>` is the alternative source knob. When only that
source is supplied, preparation reuses the existing graph builder and deterministic community
detector in memory; it does not run extraction. `prompt-system-compare` ranks all evaluated ids and
prints the best evaluated tree against its matched control, including the paired objective delta,
bootstrap CI when per-case series align, and `helps`, `hurts`, or `inconclusive` conclusion.

Local evidence on 2026-07-18 used the committed IP-regulation final split (n=4) and
`MamayLM-Gemma-3-12B-IT-v2.0` Q4_K_M on the RTX 4060 Ti 16 GB host. Control `2af73c060984`
scored 0.685; depth-2, 256-token tree `e6176121770b` scored 0.671. The paired delta was -0.0145
with CI [-0.0339, +0.0000], so the comparison is inconclusive and does not support pinning the
tree on this tiny fixture. Artifacts are under `.data/knowledge-tree-ab/`: prompt candidates in
`prompt-system/evidence/`, control run
`run-eval/20260718T170430.216852Z-20da112f09c7/`, and tree run
`run-eval/20260718T170519.951424Z-cdfe1dc48c62/`. Retrieval held recall@5=1.000 and MRR=1.000
for both. `make ci` passed with 1,477 tests and 42 slow tests deselected.

## Sample Prompt Assets

The IP regulation samples provide a small checked prompt-system fixture:

- `samples/goldsets/ip_regulation_uk/`;
- `samples/prompt_system/ip_regulation_uk/`;
- `samples/prompt_system/ip_regulation_uk/tuned/`;
- `samples/prompt_system/ip_regulation_uk/graph/`.

These samples are useful for local prompt-system mechanics and board rendering. Treat tuning wins
as provisional until a held-out final split confirms them; the prompt-system lane exists to make
that split discipline visible.

## Local Self-Improvement Loop

The self-improvement workflow closes the loop from a measured local RAG run to an adapter-backed
candidate row. It is file-driven and split-guarded:

- `src/llb/finetune/dataset.py` exports SFT records and optional DPO preference pairs from a
  finalized tuning-split run bundle. The exporter renders `eval.rag.chat` messages through the
  same prompt path as `run-eval`, writes `sft.jsonl`, `dpo.jsonl`, and `dataset_manifest.json`,
  and records the item ids, split counts, source run, and dataset digest.
- `src/llb/finetune/trainer.py` selects and orchestrates LoRA/QLoRA trainer backends, while
  `training_runtime.py` owns dataset/tokenizer/model preparation and the shared TRL loop.
  `--trainer fake` writes deterministic CI artifacts; the real path lazy-imports PEFT, TRL,
  Transformers, and Datasets from the `[finetune]` extra and saves an adapter plus
  `adapter_manifest.json`.
  `--trainer unsloth` selects the Unsloth-accelerated path (`unsloth_train_adapter`): same SFT
  loop, dataset contract, and manifest, but the base model is loaded and LoRA-wrapped through
  `FastLanguageModel` for roughly 2x faster single-GPU training. Unsloth is intentionally not a
  project extra (it pins a hardware-matched torch/triton stack, same policy as marker); install
  `unsloth` manually in the CUDA training environment. Unknown `--trainer` values exit with the
  accepted list; the manifest records the concrete trainer that ran (`peft-trl`, `unsloth`, or
  `fake`), never `auto`. Covered by dispatch/missing-dependency tests in
  `tests/llb/finetune/test_finetune.py`.
- `src/llb/finetune/guard.py` enforces the contamination invariant before `run-eval` launches a
  backend: adapter manifests may contain only tuning-split training ids, may not intersect
  calibration/final eval ids, and a tuned model cannot judge itself.
- `src/llb/finetune/loop.py` orchestrates base final eval, per-round tuning eval, miss analysis,
  dataset export, adapter training, adapter final eval, stop/accept logic, `state.json`, and
  `report.md`.
- `src/llb/finetune/campaign/run.py` schedules the loop ingredients across a `--models` roster with
  planner skip reasons, a shared campaign SFT export, per-model preference exports, VRAM reclaim
  between roster entries, `campaign.progress.jsonl` resume, and a tunability `report.md`.
- `src/llb/finetune/distill/run.py` runs local text-level distillation: a teacher answers verified
  tuning items through the normal RAG backend seam, deterministic correctness gates decide which
  answers become SFT targets, the same student is trained on teacher targets and reference targets,
  and the report compares the two adapters over the same held-out items.
- `src/llb/finetune/registry/`, `lifecycle.py`, and `serving.py` make adapters first-class,
  traceable artifacts (see [Adapter Registry And Lifecycle](#adapter-registry-and-lifecycle)).
- `src/llb/finetune/hparam_search/search.py` searches the LoRA space per model and feeds the winning
  config back as the trainer's defaults (see
  [Hyperparameter Search](#hyperparameter-search)).
- `src/llb/finetune/naming.py` holds `model_slug`, the one filesystem name a model gets across the
  campaign and hyperparameter artifact trees.

Commands:

```bash
llb export-finetune-set --run-dir <tuning-run> --goldset <goldset> --out <dataset-dir>
llb finetune-adapter --dataset <dataset-dir> --model <model> --seed <seed>
llb self-improve --model <model> --backend vllm --goldset <goldset> --rounds 2
llb finetune-campaign --models <m1,m2> --backend vllm \
  --goldset <goldset> --corpus <corpus-dir> --rounds 1
llb distill --teacher <teacher> --student <student> --backend vllm \
  --goldset <goldset> --corpus <corpus-dir> --gate 0.8
make self-improve MODEL=<model> BACKEND=vllm GOLDSET=<goldset> ROUNDS=2
make finetune-campaign MODELS=<m1,m2> BACKEND=vllm GOLDSET=<goldset> CORPUS=<corpus-dir>
make distill TEACHER=<teacher> STUDENT=<student> BACKEND=vllm GOLDSET=<goldset>
```

Artifacts live under `$DATA_DIR/self-improve/<timestamp>/round-<n>/` for campaign state and under
`$DATA_DIR/run-eval/` for canonical board bundles. Round directories carry `dataset/`, `adapter/`,
`run` and `run-final` pointers, plus per-round reports.

Multi-model campaign artifacts live under `$DATA_DIR/finetune-campaign/<timestamp>/`. The campaign
root contains `shared-dataset/dataset_manifest.json`, `campaign.progress.jsonl`, `report.md`, and
one directory per roster model. Each model directory records base-final and per-round tuning/final
run pointers, miss analysis, a per-model preference dataset, and the final adapter. Resume replays
`campaign.progress.jsonl` and does not retrain a completed roster entry.

Distillation artifacts live under `$DATA_DIR/distill/<timestamp>/`: `teacher_outputs.jsonl`,
`dataset/` for accepted teacher-answer SFT targets, `reference_dataset/` for the same item ids with
reference-answer targets, `adapter/`, `reference_adapter/`, `comparison/`, `distill_manifest.json`,
and `report.md`. The distillation manifest and accepted dataset manifest record the teacher model,
student model, gate threshold, accepted item ids, and per-item gate scores. The distilled adapter is
registered with its paired comparison delta; the reference adapter stays local comparison evidence.

Adapter-backed `run-eval` rows are labeled `<base>+adapter-<digest>` in manifests and board loaders.
`llb recommend` appends a self-improvement section when a campaign `state.json` exists and a
fine-tune campaign section when `$DATA_DIR/finetune-campaign/*/campaign.progress.jsonl` exists. The
campaign section ranks completed models by final-split delta, then shorter training wall-clock, then
lower peak VRAM; skipped models remain visible with the planner reason.

Tests:

```bash
uv run pytest tests/llb/finetune/test_finetune.py \
  tests/llb/finetune/test_distill.py \
  tests/llb/finetune/test_adapter_registry.py \
  tests/llb/board/test_recommend.py
```

The campaign implementation is covered by fake eval/trainer/planner tests for scheduling order,
planner skip reasons, shared dataset digest reuse, JSONL resume, and report ranking.
The distillation implementation is covered by fake teacher/trainer/comparison tests for gate
exclusion, tuning-only teacher generation, identity and judge-teacher refusals, report math,
registry registration, and contamination-guard compatibility.

## Hyperparameter Search

`src/llb/finetune/hparam_search/search.py` searches the LoRA configuration space for one model with a
bounded budget, so fine-tuning stops guessing rank, alpha, learning rate, epochs, target modules,
or batch geometry.

The search space also covers the effective batch axis (finetune-hparams-effective-batch-axis):
`per_device_train_batch_size` x `gradient_accumulation_steps` ride ONE `batch_geometry`
categorical (`1x4` the trainer default, `1x8`, `2x4`, `2x8`) rather than two independent draws --
independent draws would mostly differ only in a VRAM/wall-clock trade at the same effective batch,
wasting budget on gradient-equivalent points -- and `max_length` (512/1024/2048) is sampled beside
it. Effective batch size interacts strongly with the learning rate, so the recorded best config is
internally consistent: `hparams_manifest.json` carries the batch geometry the learning rate was
chosen under, and an operator changing the batch size knows they left the searched optimum. The
sampled record always satisfies `effective_batch_size == per_device * grad_accum` (unit-tested).

Dependency contract: the `[finetune]` and `[dev]` extras include Optuna. GitHub CI installs
`.[dev]`, so pure hparam slice/guard tests plus small fake-trainer manifest integrations stay in
the lightweight `make ci` suite without pulling the CUDA training stack. Multi-trial hparam
resume/prune simulations and multi-entry fine-tune campaign ranking/resume simulations are marked
`slow`; they run in the full local `make test` suite.

```bash
llb finetune-hparams --model <m> --dataset <tuning-dataset> --backend vllm \
  --goldset <goldset> --max-trials 8 [--max-hours 2] [--seed 13] [--dev-fraction 0.25] \
  [--stratify-by-base-score <scored-base-run-dir>]
llb finetune-hparams ... --resume <study-dir>
make finetune-hparams MODEL=<m> DATASET=<dir> GOLDSET=<g> MAX_TRIALS=8 \
  HPARAMS_STRATIFY_RUN=<scored-base-run-dir>
```

Artifacts land under `$DATA_DIR/finetune-hparams/<model-slug>/<timestamp>/`: `study.db` (the
persistent Optuna study), `trials.jsonl` (a live progress log), `trials/trial-<n>/` (the trial's
train-slice dataset and adapter), and `hparams_manifest.json` (best config, study seed, dev slice,
budget, and the full trial table).

### Split discipline

The discipline of `optimize/tuner.py` extends one level down. That tuner searches RAG and serving
knobs on the tuning split while `final` stays held out; here the search space is the LoRA config
itself, and the held-out set is carved from *inside* the tuning split:

- `carve_dev_slice` seeds a deterministic, disjoint train/dev partition of the dataset's item ids.
  Each trial trains only on the train sub-slice and is scored only on the dev sub-slice, so a trial
  never sees its own evaluation items.
- `--stratify-by-base-score <scored-base-run-dir>` (make: `HPARAMS_STRATIFY_RUN=`) replaces the
  uniform draw with a stratified one: `carve_stratified_dev_slice` buckets the tuning items by
  the base model's per-item `objective_score` from the given run bundle's `scores.jsonl`
  (`high` >= 0.5, `low` > 0, `zero`, `unscored`) and draws the dev slice proportionally per
  bucket with a floor of one, answerable buckets first -- so a small dev slice always carries
  items the base model can answer and the trial objective can discriminate (the failure the
  first CUDA search hit: a uniform 3-item slice with one answerable item tied every trial at
  0.0000). A population the base model scores 0.0 everywhere is REFUSED -- no slice can rank
  trials against a constant objective. The same disjointness and seeded determinism hold, and
  `hparams_manifest.json` records an additive `dev_slice.strata` block (the source run plus
  per-bucket population/dev counts and mean base score). The default without the flag stays the
  uniform slice. Committed fixture: `samples/finetune/base-score-run/scores.jsonl` (12 items, 3
  answerable), used by `tests/llb/finetune/test_finetune_hparams.py` to prove the stratified
  slice holds an answerable item at every seed where the uniform slice misses.
- `assert_tuning_only` refuses the search outright when the dataset's `split_counts` name any split
  but `tuning`, and -- when a goldset is available -- when its item ids intersect the real
  calibration/final ids. A dataset manifest is operator-writable, so its split counts alone are not
  proof (the same lesson the registry records for adapter manifests).
- The default objective scores the trial adapter through `run_eval` over the dev items only. It
  refuses a non-vLLM backend and a missing goldset BEFORE the study is created: the first trial
  fine-tunes a model before it ever reaches the objective, so a late refusal would waste a full
  training run.

### Budget and resume

`--max-trials` caps the trial count; `--max-hours` caps wall clock. A trial is atomic (a whole
fine-tune), so the wall-clock budget is checked BETWEEN trials through an Optuna callback -- one
in-flight trial may overrun the deadline and is never killed mid-training. An aborted study records
`budget_exhausted: true` and stays resumable: the SQLite study persists, and `--resume <dir>` runs
only `max_trials - len(study.trials)` further trials, so finished trials are never repeated.

A measured OOM prunes its trial (reusing `optimize.tuner.is_oom`) instead of crashing the study; any
other exception fails loudly -- but only after `hparams_manifest.json` is written, so a study killed
by one bad trial stays inspectable and resumable instead of leaving a bare `study.db`.

Pre-run infeasibility prune (finetune-hparams-infeasible-point-prune): with
`--vram-headroom-mib <n>` (make: `HPARAMS_VRAM_HEADROOM=`) -- the VRAM left beside the base model
during training on the host -- a trial whose estimated adapter TRAINING footprint exceeds the
headroom is pruned BEFORE `trainer_fn` runs, so a bounded budget never pays a full fine-tune for
a known-infeasible point. The estimate is `rank x targeted modules x layers x 2 (hidden x r)
matrices x 16 bytes/param` (bf16 weight + grad, fp32 Adam moments + master copy;
`estimated_adapter_train_mib`), with hidden size / layer count read from the model's cached HF
config (`model_arch` overrides it programmatically). Every trial row in `hparams_manifest.json`
and `trials.jsonl` carries the additive `estimated_adapter_mib`, and the prune reason names the
estimated footprint against the headroom. The estimate is deliberately coarse: it complements
the measured-OOM prune (which always stays in place), never replaces it. Without a headroom the
pre-run prune is off.

### Feeding the trainer

`trainer_defaults(data_dir, model)` reads the newest `hparams_manifest.json` for that model and
returns `{"hyperparameters": <best>, "hparams_manifest": <path>}`. It is the default trainer wiring
for `self-improve`, `finetune-campaign`, and `finetune-adapter` (which accepts `--default-hparams`
to opt out). `train_adapter` records `hparams_manifest` in `adapter_manifest.json` as pure
provenance: it never enters `adapter_digest`, because two adapters with identical hyperparameters
are the same adapter whether or not a search chose them.

Discovery only scans the default tree `$DATA_DIR/finetune-hparams/<model-slug>/<timestamp>/`. A
study written elsewhere with `--out-dir` is a one-off: it is never auto-consumed as a trainer
default.

`dataset.subset_dataset` materializes each trial's train sub-slice as a real dataset directory with
its own recomputed digest. A filtered view would inherit the parent's `dataset_digest`, and since
`adapter_digest` derives from it, two adapters trained on different data would collide on one
registry id.

Tests: `tests/llb/finetune/test_finetune_hparams.py` covers dev-slice disjointness and
determinism, both guard refusals, the no-protected-id-in-any-trial invariant, manifest writing,
the manifest surviving a failed trial, subset digests, and the trainer consuming a recorded best
config through a self-improvement round in the lightweight suite. Slow coverage keeps the seeded
full trial table, budget abort plus resume without repeated trials, OOM and infeasible-point
pruning, and effective batch sampling.

### CUDA evidence on the 12 GB RTX PRO 3000 host

An 8-trial search for `Qwen/Qwen2.5-0.5B-Instruct` over the `ua_squad_postedited_v1` tuning split
(82 verified items -> 62 train / 20 dev at `dev_fraction=0.25`, `seed=13`).

- Tuning-split base run: `objective 0.2610`, reliability `1.000`, recall@3 `0.915`, `177.7` tok/s;
  the dev slice's base objective is `0.2056`.
- Study: `.data/finetune-hparams-evidence/study/hparams_manifest.json`
  (`finetune-hparams-Qwen-Qwen2.5-0.5B-Instruct-313415c09b62-s13`); 8 complete, 0 pruned; each trial
  fine-tunes the 62 train items and scores the 20 dev items through vLLM LoRA serving in `60` to
  `99` s.

| trial | dev objective | rank | alpha | dropout | learning rate | epochs | target modules |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.3233 | 16 | 64 | 0.05 | 2.96e-05 | 3 | qv |
| 1 | 0.2917 | 8 | 16 | 0.00 | 1.18e-04 | 4 | attn_mlp |
| 7 | 0.2861 | 32 | 64 | 0.00 | 1.88e-04 | 1 | attn |
| 6 | 0.2789 | 64 | 128 | 0.00 | 1.38e-05 | 2 | qv |
| 0 | 0.2674 | 64 | 256 | 0.05 | 1.26e-05 | 4 | attn_mlp |
| 4 | 0.2583 | 4 | 8 | 0.15 | 4.71e-04 | 4 | qv |
| 3 | 0.2059 | 4 | 8 | 0.00 | 2.61e-05 | 1 | attn |
| 5 | 0.2056 | 16 | 16 | 0.10 | 1.66e-05 | 4 | qv |

The best config (trial 2) scores `0.3233` on the dev slice against the base model's `0.2056`, and the
spread across trials is non-saturated, so the search discriminates rather than tying. Rank is not
monotonic: the two rank-4 points bracket the field and the widest module preset (`attn_mlp`) does not
win, which is the whole reason to measure rather than guess.

Two caveats the numbers carry:

- The dev slice can use a seeded plain split or base-score stratification. Supplying
  `--stratify-by-base-score <run>` represents every non-empty score bucket and guarantees
  answerable items; an all-zero base run is rejected because it cannot discriminate trials.
- Trial 5 lands exactly on the base objective `0.2056`: a tuned adapter is not automatically better
  than no adapter, and the search records that honestly.

### Effective-batch-axis evidence on the 16 GB RTX 4060 Ti host

The widened-space acceptance run (2026-07-10, finetune-hparams-effective-batch-axis): a 6-trial
search for `google/gemma-3-1b-it` over the `ua_squad_postedited_v1` tuning split (82 items ->
62 train / 20 dev, `seed=13`; full-split base tuning objective `0.3050`), study
`.data/finetune-hparams/google-gemma-3-1b-it/20260710T121020*/hparams_manifest.json`, ~2 min per
trial end to end (QLoRA fine-tune + vLLM LoRA dev eval):

| trial | dev objective | geometry | eff. batch | max_length | rank | lr | preset |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | **0.3262** | 2x8 | 16 | 2048 | 16 | 2.63e-05 | attn_mlp |
| 2 | 0.3151 | 2x8 | 16 | 2048 | 4 | 2.53e-05 | attn |
| 4 | 0.2986 | 2x8 | 16 | 2048 | 64 | 1.53e-04 | qv |
| 0 | 0.2865 | 2x4 | 8 | 2048 | 64 | 2.73e-05 | attn_mlp |
| 5 | 0.2692 | 2x4 | 8 | 2048 | 8 | 3.55e-04 | attn_mlp |
| 3 | 0.2427 | 1x8 | 8 | 2048 | 64 | 9.08e-05 | attn_mlp |

What the run demonstrates: the learning-rate x effective-batch interaction is measurable, not
theoretical -- trials 0 and 1 sample a near-identical learning rate (2.7e-05 vs 2.6e-05) and the
effective-batch-16 point beats the effective-batch-8 point by **+0.040** dev objective; the three
top trials all ride the largest geometry (`2x8`). The honest caveats: the trainer-default `1x4`
geometry was never drawn in this 6-trial budget (TPE explored the wider geometries), so the
comparison to the pinned default is indirect (via `2x4`/`1x8` at effective batch 8, both of which
lose), and a 20-item dev slice carries wide uncertainty per point. The operational win stands
regardless of ranking noise: `hparams_manifest.json` records the batch geometry every
learning rate was chosen under, so the recorded best config
(`2x8`, lr 2.63e-05, rank 16, `attn_mlp`, `max_length` 2048) is self-consistent and
`trainer_defaults` feeds all of it -- geometry included -- to later rounds.

## Compressed-QAT Trainability (finetune-compat)

`src/llb/finetune/compat.py` (compressed-qat-adapter-support) answers "can this checkpoint take a
LoRA adapter on this host?" BEFORE a campaign pays for a base eval or a training run. Compressed
QAT checkpoints (`*-qat-w4a16-ct` and friends) serve well on vLLM, but PEFT can only inject LoRA
into layer types it has a dispatch for (full-precision `Linear`, bitsandbytes 4/8-bit, GPTQ, AWQ,
EETQ, HQQ) -- a `compressed-tensors` checkpoint's `CompressedLinear` layers cannot take adapters.

Two stages, both pure over injectable seams (`tests/llb/finetune/test_finetune_compat.py` runs with fake
modules and configs, no torch):

- Config introspection (`inspect_quantization` + `assess_quantization`): classifies the
  checkpoint's native `quantization_config.quant_method` against PEFT's dispatch table -- no
  weights, no CUDA. `compressed-tensors` is a deterministic not-trainable verdict with the exact
  blocker plus the documented fallback (train on the uncompressed base and serve merged/quantized,
  or take the bitsandbytes path); a PEFT-dispatched scheme names its injection strategy; an
  unrecognized scheme stays `unknown` so the heavy probe decides.
- The heavy probe (`probe_trainability`, `llb finetune-compat --model <m>`): loads the model,
  scans its ACTUAL linear module classes, selects per-architecture target modules from the modules
  that exist (`select_target_modules` grounds the choice in the model's own names -- llama-style
  `q_proj`, falcon `query_key_value`, gpt2 `c_attn`, with a most-frequent-suffix fallback --
  instead of assuming llama naming), attaches a rank-4 LoRA, and runs one forward/backward
  micro-step. Any failure becomes the recorded blocker, never a crash. Reports land under
  `$DATA_DIR/finetune-compat/<model>/<timestamp>/compat_report.json`; `--config-only` stops after
  stage 1.

Campaign integration: `run_finetune_campaign` runs a config-only compat probe (injectable
`compat_fn`; the default reads only locally-cached configs, so Ollama tags and never-downloaded
models return `unknown` without touching the network) after the memory planner and BEFORE the
base eval -- a positive not-trainable verdict skips the entry into `campaign.progress.jsonl` and
`report.md` with the exact blocker; an unknown verdict never false-skips.

CUDA evidence (2026-07-10, RTX 4060 Ti 16 GB):

- `google/gemma-4-E4B-it-qat-w4a16-ct` -> **not-trainable** at the config stage
  (`quant_method 'compressed-tensors' has no PEFT LoRA dispatch`); the skip fires before any
  weights load. `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4` hits the same verdict -- its "AWQ"
  is AWQ-inside-compressed-tensors, which the config stage classifies correctly.
- `Qwen/Qwen3-4B-FP8` -> config stage says `unknown` (`quant_method 'fp8'`), the heavy probe
  loads it and the module scan finds `FP8Linear` (no PEFT dispatch) -> **not-trainable** with
  that exact blocker -- the load-time detection path proven on a real checkpoint.
- Reports: `.data/finetune-compat/google-gemma-4-E4B-it-qat-w4a16-ct/*/compat_report.json`,
  `.data/finetune-compat/Qwen-Qwen3-4B-FP8/*/compat_report.json`.

## Adapter Registry And Lifecycle

Adapters are first-class artifacts, not loose directories. `$DATA_DIR/adapters/registry.jsonl` is an
append-only event log (`register` / `merge` / `delete`) folded into the current entry set on read, so
a partial write can never lose earlier history. The entry id IS the `adapter_digest`, so it can never
be reassigned to different weights.

Modules:

- `src/llb/finetune/registry/`: `model.py` owns `AdapterEntry`; `io.py` folds the event log;
  `register.py` performs idempotent registration and lifecycle writes; `resolve.py` handles id /
  unique prefix / label / directory lookup; `staleness.py` compares benchmark fingerprints; and
  `rows.py` renders CLI/board rows;
- `src/llb/finetune/lifecycle.py`: run-bundle citation scan, supersession, and garbage collection;
- `src/llb/finetune/serving/model.py`: immutable serve and merge contracts;
- `src/llb/finetune/serving/run.py`: serve-plan construction and launcher lifecycle;
- `src/llb/finetune/serving/merge.py`: cached merges, GGUF conversion, and Ollama Modelfiles;
- `src/llb/finetune/serving/launcher.py`: backend-specific launcher construction.

```bash
llb register-adapter --adapter-dir <dir> [--goldset <g>] [--corpus <c>] [--source-run <run>]
llb list-adapters [--json]
llb serve-adapter --adapter <id> --backend vllm|ollama|llamacpp [--smoke]
llb gc-adapters [--dry-run] [--force]
llb run-eval --adapter <id> --model <base> --backend vllm
make list-adapters ; make serve-adapter ADAPTER=<id> BACKEND=<b> ; make gc-adapters GC_DRY_RUN=1
```

Entries record the base model, dataset digest, dataset item ids and split counts, the goldset and
corpus digests observed AT TRAINING TIME, the source run, and an eval summary. Self-improvement and
campaign rounds auto-register through `register_round_adapter` after the adapter's own final eval,
so the entry carries the evidence the board later cites. Registration is best-effort: an injected
trainer that writes no `adapter_manifest.json` logs a warning instead of aborting the round. A bare
`llb finetune-adapter` does not register, so `llb register-adapter` exists to adopt a hand-trained
adapter into the registry rather than leave its board row silently dropped.

### Staleness

`staleness()` compares the recorded goldset/corpus digests against the present ones
(`durability.goldset_digest` and `corpus_governance.corpus_fingerprint`, the same functions the
durable-run journal and the stale-store check use). Verdicts are `current`, `stale`, and `unknown`;
a missing digest yields `unknown` and never `current`. Detection reports, it never retrains.

A third axis covers the RAG store (adapter-staleness-retrieval-fingerprint): an adapter is
trained on retrieved CONTEXT, so re-embedding or rechunking the same corpus invalidates its
training contexts while `corpus_fingerprint` stays unchanged. Registration records a
`retrieval_fingerprint` (embedder, chunk strategy/size/overlap, retrieval mode) read from the
store's `store_meta.json` (`register_adapter --index-dir` on the CLI; `self-improve` /
`finetune-campaign` rounds record the config's index dir automatically), and `staleness()`
compares it per knob against the store's present meta -- a rebuilt store flips the entry `stale`
with the changed knob named in the reason (for example
`retrieval embedding_model changed since training (a -> b)`). An adapter registered without an
index directory reads `unknown` on the retrieval axis (reason `retrieval fingerprint unavailable`),
never `current`.

`board/runs.py` resolves every adapter-backed bundle through the registry before it can rank:

- an unregistered adapter's row is DROPPED (a tuned number nobody can trace is not comparable);
- a registered-but-stale adapter's row is stamped `<base>+adapter-<digest> [stale]`.

`recommend.load_run_summaries` reuses `load_run_records`, so both the board and `llb recommend`
inherit the rule from one seam.

### Contamination guard through the registry

`validate_adapter_for_eval` reads training provenance from the registry when the adapter is
registered, falling back to `adapter_manifest.json` only when it is not (a freshly trained adapter
registers after its first eval). The manifest beside the weights is operator-writable, so a
hand-edited one could otherwise launder a final-split adapter past the gate. The refusal message
names the intersecting ids, the offending splits, and which provenance was consulted.

### Serving

vLLM serves the LoRA directly through the existing `--enable-lora --lora-modules` wiring, sized by
`--max-lora-rank`. That flag defaults to 16, so an adapter trained at a higher rank fails
`add_lora` at engine startup (`LoRA rank 64 is greater than max_lora_rank 16`) and vLLM exits before
serving anything. Both adapter launch paths (`executor/runner.py` for `run-eval`, `serving.py` for
`serve-adapter`) therefore read the rank off the adapter they are about to serve --
`trainer.adapter_lora_rank` prefers PEFT's own `adapter_config.json` over our manifest, since it
describes the weights actually on disk -- and `backends/vllm.served_lora_rank` rounds it up to the
nearest value vLLM accepts (`1, 8, 16, 32, 64, 128, 256, 320, 512`). An adapter of unknown rank
leaves the flag off and vLLM keeps its default.

Ollama and llama.cpp serve whole model artifacts, so `serving.py` merges the adapter into its base
weights
(PEFT `merge_and_unload`), converts to GGUF via the llama.cpp checkout's `convert_hf_to_gguf.py`, and
for Ollama registers a `llb-adapter-<short-id>` tag. The merge is expensive and one-way, so it is
cached under `$DATA_DIR/adapters/merged/<short-id>/<backend>/` behind a `merge.json` and recorded as
a registry `merge` event. Both the merge and the launcher are injectable, so CI exercises all three
backends without CUDA, llama.cpp, or a running Ollama daemon. `serve-adapter` probes the endpoint
with one generation -- an empty completion FAILS the probe (a served-but-mute endpoint is not
serving) -- and then holds it in the foreground until Ctrl-C; there is no serving daemon.

Chat-template preservation is required because llama.cpp's server applies
the `tokenizer.chat_template` GGUF metadata natively, but **Ollama ignores it** when a model is
created from a bare `FROM <gguf>` Modelfile -- the tag serves raw completions and a merged
instruct model degrades to gibberish or empty chat answers. `modelfile_text` therefore reads the
merged tokenizer's `chat_template.jinja`, detects the template family by its unambiguous marker
(ChatML
`<|im_start|>`, Gemma `<start_of_turn>`, Llama 3 `<|start_header_id|>`), and writes the
equivalent Go `TEMPLATE` plus its `PARAMETER stop` tokens into the Modelfile; an unrecognized
template stays a bare FROM with a loud warning naming the fix. Family detection, the bare-FROM
fallback, and the empty-probe failure are unit-tested with fixtures.

Pristine tokenizer files are copied from the base model because a LoRA never changes the tokenizer,
while `AutoTokenizer.save_pretrained` can be lossy for GGUF conversion: it drops the sentencepiece
`tokenizer.model` (the converter's GPT-2-style fallback then asserts on vocabularies whose added
tokens sit past `config.vocab_size`) and rewrites `tokenizer_config.json` so the control-token
markings are lost: `<start_of_turn>`/`<end_of_turn>` exported as NORMAL instead of CONTROL token
types, Ollama then never matched the template's turn markers as specials, and the merged Gemma
answered every non-trivial prompt with an immediate `<end_of_turn>` (final-split objective 0.199
vs 0.410 served properly -- while the SAME safetensors answered correctly in transformers).
`copy_base_tokenizer_assets` overwrites the resaved files with the base repo's originals
(`tokenizer.model`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`),
best-effort per file so repos without a given file (Qwen has no sentencepiece model) keep the
resaved copy that already converts fine. Unit-tested with an injected downloader.

### Garbage collection

An adapter is superseded once a newer adapter exists for the same base model, ordered by
`(created_at, log sequence)`. `created_at` has second resolution, so two fast rounds tie; the
append-log position breaks the tie exactly. Only superseded adapters are GC candidates, and GC
refuses any that a durable artifact still cites. The citation scan covers published run bundles
(`$DATA_DIR/run-eval/*/manifest.json`, matched by recorded digest or by served `adapter_path`)
AND the orchestrator journals that also link adapter directories: self-improvement
`$DATA_DIR/self-improve/*/state.json` (`rounds[].adapter_dir`) and campaign
`$DATA_DIR/finetune-campaign/*/campaign.progress.jsonl` (`entry.adapter_dir`), both resolved
through the registry's adapter-dir index the way the served-path match is. Every citation
carries its artifact kind (`run-bundle` / `self-improve-state` / `campaign-journal`) in
`GcDecision.cited_by`, the refusal reason names the citing artifact(s), and `gc_rows` exposes
the kinds in a `cited_kinds` column. `--force` overrides the citation refusal but never the
safety rule that GC only deletes directories inside `$DATA_DIR`. Deletions append a `delete`
tombstone.

### Committed fixtures

- `samples/finetune/registry/registry.jsonl`: a stale entry (with a folded `merge` event) and a
  poisoned-digest entry, both pointing at adapter dirs outside `$DATA_DIR`;
- `samples/finetune/gc-journals/`: a data-dir-shaped fixture whose campaign journal cites the
  committed stale adapter, proving a journal-only citation blocks an unforced GC;
- `samples/finetune/stale-adapter/`: recorded digests that no longer match
  `samples/goldsets/ip_regulation_uk/`;
- `samples/finetune/laundered-adapter/`: an `adapter_manifest.json` that CLAIMS a clean tuning-only
  training set while the registry records the `final`-split ids it was really trained on;
- `samples/finetune/poisoned-adapter/`: the simpler case where the manifest itself declares the
  protected split, refused even when unregistered.

`tests/llb/finetune/test_adapter_registry.py` covers registry round-trip and idempotence, the
staleness flip when the goldset digest changes, the `unknown` verdict, guard resolution through
the registry, serving smoke over a fake launcher for all three backends, merge-event recording and
merge caching, GC citation refusal plus `--force` (run-bundle, self-improve-state, and
campaign-journal citations, including the committed journal fixture), the same-second supersession
tie, the outside-`$DATA_DIR` safety rule, and board drop/stamp behavior.

Merge-serving CUDA evidence (2026-07-10, RTX 4060 Ti 16 GB, adapter-merge-serving-cuda-evidence;
the first time the real merge lane ran end to end):

- Adapter: `ea848f7e160e` (`Qwen/Qwen2.5-0.5B-Instruct`, one `self-improve` round over the
  `ua_squad_postedited_v1` tuning split, registered; campaign
  `.data/self-improve/merge-evidence-qwen05b/`).
- Both GGUF backends merged and answered the smoke probe: PEFT merge + `convert_hf_to_gguf.py`
  (f16) + launch + probe in **~15 s wall-clock per backend** for the 0.5B model, GGUF size
  **949 MB** (vs ~1 GB safetensors); converter accepted the Qwen2 architecture without complaint.
- Three-way final-split objective (n=82, same goldset/store/seed):
  base (vLLM) **0.2880** [0.204, 0.370]; vLLM LoRA row **0.3272** [0.239, 0.422]; merged tag on
  ollama **0.3119** [0.218, 0.402] -- inside the LoRA row's CI and above the base point estimate,
  so the merged artifact answers as the ADAPTER, not the base model. Run bundles:
  `.data/run-eval/20260710T075222*` (base), `...075718*` (LoRA), `...081359*` (merged, fixed
  template).
- The Ollama Modelfile carries the explicit chat template described above, while the smoke probe
  rejects an empty completion. The `finetune` extra includes both the converter's `gguf` import and
  the trainer's `bitsandbytes` dependency so failures occur during dependency validation.

Second cohort model, `google/gemma-3-1b-it` (2026-07-10, same host; adapter `db80e8440b7d` from
one `self-improve` round trained with the effective-batch search's best config, campaign
`.data/self-improve/merge-evidence-gemma-3-1b/`):

- Merge cost: ~24 s (Ollama) / ~18 s (llama.cpp) wall-clock per backend, 1.9 GB f16 GGUF. The
  converter uses the base repository's pristine tokenizer files to preserve the sentencepiece
  vocabulary and control-token types.
- Three-way final-split objective (n=82): base (vLLM) **0.3872** [0.299, 0.480]; vLLM LoRA row
  **0.4103** [0.326, 0.498]; merged tag on ollama **0.3427** [0.260, 0.428] -- inside the LoRA
  row's CI, so the merge passes the fidelity gate, with the honest caveat that the point
  estimate sits 0.068 below the LoRA row (unresolved at n=82, and partly a cross-backend
  comparison: the merged row is f16-GGUF-on-ollama while both reference rows are
  safetensors-on-vLLM). Run bundles: `.data/run-eval/20260710T122520*` (base),
  `...122821*` (LoRA), `...125503*` (merged).

CUDA evidence on the 12 GB RTX PRO 3000 host:

- Command shape: `LLB_EMBED_DEVICE=cpu llb finetune-campaign --config
  .data/quickstart-leaderboard/llb/serving/gpu-12gb/run_eval_gemma_4_12b_vllm.yaml --models
  Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct --corpus
  samples/goldsets/ua_squad_postedited_v1/corpus --rounds 1 --limit 1 --out-dir
  .data/finetune-campaign/task19-evidence-qwen-small-12gb`.
- Campaign report:
  `.data/finetune-campaign/task19-evidence-qwen-small-12gb/report.md`.
- Recommend summary:
  `.data/recommend/task19-summary.md`.
- Shared dataset digest: `5b99939c91b02500eda6fe3aa7cb27c46012928929f93def380a245b4a6711b0`.
- `Qwen/Qwen2.5-0.5B-Instruct`: base final objective `0.0000`, adapted objective `0.0000`,
  delta `0.0000`, train wall-clock `6.7800` s, adapted peak VRAM `11862` MiB.
- `Qwen/Qwen2.5-1.5B-Instruct`: base final objective `0.0000`, adapted objective `0.0000`,
  delta `0.0000`, train wall-clock `6.4219` s, adapted peak VRAM `11690` MiB.
- `llb recommend --gpu-gb 12 --no-chart` rendered the fine-tune campaign section and selected the
  0.5B base model for this smoke cohort because all one-case objectives were tied at zero and the
  base model was faster than its adapter-backed row.
- `google/gemma-4-12B-it-qat-w4a16-ct` served on the same host at `max_model_len=1024`
  (`41.8` to `42.9` tok/s, peak VRAM about `11523` MiB), but PEFT LoRA injection could not train
  the compressed-tensors QAT checkpoint because its compressed linear modules do not expose the
  normal `weight` attribute.
