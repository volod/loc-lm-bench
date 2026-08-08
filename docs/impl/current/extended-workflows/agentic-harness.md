# Agentic Harness Comparison

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

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

## Protocol decision: policies transfer on the harness seam

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

Loop and LangGraph matched item-for-item on completion and prompt tokens under both policies
(seed-task observations sit under the 800-char cap, so `observation_cap` is a no-op on this set --
the transfer seam is what the run proves). Bundles under `.data/agentic/20260729T12*`.
