# Agent Loop-Policy Recommendation

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

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

## Powered Repeat-Noop Comparison

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
