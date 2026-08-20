# Context-Policy Constant Sweeps

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

## Agent context-policy constants

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

Core locations: `src/llb/bench/context_policy/sweep.py` (grids, pairing, verdicts),
`src/llb/cli/bench/context/context_sweep.py`, CI span arithmetic in
`tests/llb/bench/context_policy/test_agentic_context_sweep.py`. Bundles land under
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

## keep_last_n on longer transcripts

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

Core locations: `src/llb/bench/context_policy/long_transcript.py` (medium-search shrink + synthetic
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
