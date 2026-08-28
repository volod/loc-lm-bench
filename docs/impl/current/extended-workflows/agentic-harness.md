# Agentic Harness Comparison

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

The agentic benchmark can run the same task set through multiple harnesses while keeping the model,
tools, world state, objective checks, optional judge, context-management policy, and loop policy
fixed.

Core locations:

- `src/llb/bench/agentic/model.py`: `Harness` protocol (carries `policy`, `budget`, and
  `loop_policy`) and harness names;
- `src/llb/bench/agentic/run.py`: runner integration;
- `src/llb/cli/bench/categories/agentic.py`: the `bench-agentic` scored cell (the context-policy
  comparison lives beside it in `agentic_context.py`);
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
llb bench-agentic --model <model> --backend <backend> \
  --malformed-policy repair_once --repeated-call-policy noop --repeat-feedback uk
llb bench-agentic-compare --model <model> --context-policy observation_cap
make bench-agentic MODEL=<model> BACKEND=<backend> \
  AGENTIC_MALFORMED_POLICY=repair_once AGENTIC_REPEATED_CALL_POLICY=noop
make agentic-harness-compare MODEL=<model> BACKEND=<backend> \
  AGENTIC_CONTEXT_POLICY=full AGENTIC_HARNESSES='loop langgraph'
```

## The loop cell is a flag on the scored run

`bench-agentic-loop` sweeps three controller knobs -- step budget, malformed-call handling,
repeated-call handling (plus the repeat wording under `noop`) -- and recommends one cell. All four
are now flags on `bench-agentic` itself:

| flag | values | default | make variable |
| --- | --- | --- | --- |
| `--max-steps` | `>= 1` | `6` | `AGENTIC_MAX_STEPS` |
| `--malformed-policy` | `answer` / `repair_once` / `strict` | `answer` | `AGENTIC_MALFORMED_POLICY` |
| `--repeated-call-policy` | `allow` / `noop` | `allow` | `AGENTIC_REPEATED_CALL_POLICY` |
| `--repeat-feedback` | the eight variants in `loop_policy.py` | `current` | `AGENTIC_REPEAT_FEEDBACK` |

`--repeat-feedback` only reaches a prompt under `--repeated-call-policy noop`; under `allow` no
repeat is ever suppressed, so nothing renders it. An unknown value on any of the three exits 2 with
the accepted list, from `LoopPolicy`'s own validation rather than a second copy of the vocabulary.

Before this, a recommended cell could only be RE-SWEPT (`bench-agentic-loop` with a two-point grid),
never RUN: the sweep's own recommendation could not configure the command it was meant to configure.
The sweep still exists and still confirms a cell against its mandatory legacy baseline -- what
changed is that scoring a model under the recommended cell no longer requires a grid.

The manifest of every `bench-agentic` bundle records `malformed_call_policy`,
`repeated_call_policy`, `repeat_feedback_variant`, and `loop_policy_supported`, so a bundle states
the controller cell it ran under instead of leaving it implied by the shipped defaults.

## Protocol decision: policies transfer on the harness seam

The `Harness` protocol takes optional `policy`, `budget`, and `loop_policy` alongside
`(task, complete, catalog, max_steps)`. Product rule:

- `loop` and `langgraph` APPLY the requested policy to every step prompt (shared
  `step_prompt` / `ContextState` / overflow guard), so a measured win from
  `bench-agentic-context` is one the operator's LangGraph cell can actually run;
- `crewai` ACCEPTS the kwargs for protocol parity but does NOT apply them -- CrewAI owns its
  ReAct transcript. Episodes stamp `context_policy_supported=false`, and the comparison labels
  the policy cell as `full*` / `observation_cap*` rather than silently reporting our `full`
  assembly. Prompt sizes the framework actually sent still ride on episode telemetry.

`loop_policy` threads on the same seam and carries the same honesty flag,
`Episode.loop_policy_supported`, but its support line falls differently -- the controller decisions
are not assembled from a policy object outside the loop, they are the loop's own edges:

- `loop` implements every cell, so it always applies the requested one;
- `langgraph`'s edges hard-code the SHIPPED decisions (a malformed reply is the final prose answer,
  a repeated call is executed), which is exactly the default cell and no other. It reports
  `loop_policy_supported=true` for the default and `false` for anything else, so today's runs are
  unchanged and a non-default cell is never read off a graph that could not run it;
- `crewai` reports `false` unconditionally: the framework owns both the transcript and the
  decisions.

Every harness records per-step prompt sizes on `Episode.telemetry`. Bundles persist
`context_policy`, `context_policy_supported`, the four loop-cell fields, `loop_policy_supported`,
and `mean_max_prompt_tokens`. The harness comparison keeps the best run per
`(model, harness, context_policy)` and ranks harnesses under ONE fixed
policy (explicit `--context-policy`, else the policy with the most harness coverage), so the axis
never silently mixes framework and context management. The ranked board stays completion-only; an
appendix table reports `prompt-tok`, the requested policy, and whether it was applied.

## CUDA host evidence: the loop cell reaches the scored run (2026-08-28)

Two `make bench-agentic` runs on the RTX 4060 Ti 16 GB CUDA host, `MamayLM-Gemma-3-12B-IT-v2.0`
GGUF Q4_K_M on Ollama, the committed 4-task UA seed set (`ag-001`..`ag-004`), `--max-steps 6`,
`--context-policy full`, loop harness. The only difference between them is the loop cell: the
shipped `answer` / `allow` / `current`, and `repair_once` / `noop` / `uk`.

| cell | completion | repeated calls | suppressed as no-ops | tok/s |
| --- | --- | --- | --- | --- |
| `answer,allow,current` (shipped) | 0.250 | 5,5,5,5 | 0,0,0,0 | 17.5 |
| `repair_once,noop,uk` | 0.250 | 5,5,4,5 | 5,5,4,5 | 19.7 |

Reading: the flags reach the controller. Under `allow` every one of the 19-20 repeated calls
executed against the world; under `noop` every one was suppressed and answered with the Ukrainian
feedback string instead -- which is the behavior change the cell names, arriving on a SCORED run
rather than on a grid point. Both bundles carry their cell in the manifest
(`repeated_call_policy`, `repeat_feedback_variant`, `loop_policy_supported=true`).

Boundaries: completion 0.250 on a 4-task seed is a smoke figure with no confidence attached -- this
run demonstrates the plumbing and says NOTHING about whether the non-default cell is better. The
throughput difference (17.5 vs 19.7 tok/s) is two unpaired single runs on a shared host, not a
measurement. What would overturn it: a bundle whose recorded cell disagrees with the flags passed,
or a `noop` run whose case rows show `n_repeated_noops` of zero while `n_repeated_calls` is not.

## CUDA host evidence (2026-07-29)

`MamayLM-Gemma-3-12B-IT-v2.0` on Ollama (`--max-model-len 8192`), 4-task UA seed,
`AGENTIC_HARNESSES='loop langgraph'` (CrewAI extra not installed on this host; fake-crew CI covers
its unsupported path):

| policy | harness | completion | mean steps | mean max prompt tok | applied |
| --- | --- | --- | --- | --- | --- |
| `full` | loop | 0.250 | 6.00 | 906.2 | yes |
| `full` | langgraph | 0.250 | 6.00 | 906.2 | yes |
| `observation_cap` | loop | 0.250 | 6.00 | 906.2 | yes |
| `observation_cap` | langgraph | 0.250 | 6.00 | 906.2 | yes |

Run on the RTX PRO 3000 Blackwell 12 GiB CUDA host; four bundles, one per policy/harness cell.

Loop and LangGraph matched item-for-item on completion and prompt tokens under both policies, and
throughput separates them by less than half a percent (5.47 / 5.49 tok/s under `full`, 5.48 / 5.49
under `observation_cap`). Reading: the harness is a SEAM, not a variable -- swapping the executor
changes nothing an operator measures, which is what licenses running the cheap loop locally and
LangGraph only where its tooling is wanted. Boundaries: `observation_cap` is a no-op on this set
because every seed-task observation sits under the 800-char cap, so this run proves the transfer
seam and NOT that the cap works; completion 0.250 on a 4-task seed is a smoke figure with no
confidence attached, and `reliability` is 0.0 across all four cells. What would overturn it: a task
set with observations past the cap, where the two harnesses could truncate at different points.
