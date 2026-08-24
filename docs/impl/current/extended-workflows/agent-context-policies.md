# Agent Context-Management Policies

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

`bench-agentic-context` ranks how the agent spends its context window for ONE fixed model over one
task set. It is the agent-side sibling of the chain-context lane below: the model, the task set, the
tool world, the success checks, and the step budget are held FIXED and only the context-management
policy varies, so every difference it reports is attributable to context handling. Every policy runs
a fresh episode through the pure `loop` harness by default; the same `policy`/`budget` kwargs
transfer onto `langgraph` via the widened `Harness` protocol (see [Agentic Harness
Comparison](agentic-harness.md#agentic-harness-comparison)), while CrewAI records the policy as
unsupported because it owns its own transcript.

Core locations:

- `src/llb/bench/agentic/context.py`: the policy vocabulary, observation trimming, transcript
  assembly, compaction, and the per-episode telemetry;
- `src/llb/backends/context_budget.py`: the per-step prompt budget (`ContextBudget`) and its
  resolution from the declared + probed usable window -- shared with the context-ablation
  document lanes ([context ablation](../rag-core/context-ablation.md)), so a loop and a lane on
  one host cannot disagree about what fits;
- `src/llb/backends/served_window.py`: per-backend probe of the window the runtime is actually
  serving (Ollama `/api/ps`, vLLM `/v1/models`, llama.cpp `/props`);
- `src/llb/bench/agentic/episode.py`: `build_agent_prompt_lines` (the policy seam) and the
  policy-aware `run_episode`;
- `src/llb/bench/context_policy/run.py`: the four-policy run + persistence;
- `src/llb/bench/context_policy/report.py`: the paired reading, the policy table, and the
  recommendation;
- `src/llb/bench/context_policy/sweep.py`: the constant-grid sweep (cap / head-share / keep_last_n)
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
  header), so a fat search hit does not re-blow every later prompt. When the summary already carries
  machine hit-count facts, a finish cue names the known `hits=` value and tells the model to call
  `finish` instead of searching again. The summary marker carries the count of steps it stands in
  for, so a folded step is never silently absent. Search observations being folded also inject their
  aggregate headers into the summary text itself (not only into the summarizer prompt), so hit
  counts do not depend on the free-text summary remembering a number. Two rules keep the policy
  honest: at most ONE compaction per step (if the compacted prompt still does not fit, the guard is
  what ends the episode, not another round of summarizing), and the summarize call's INPUT is ITSELF
  capped -- its input is the transcript that just blew the step prompt, so an uncapped summarizer is
  the one call in the loop guaranteed to overflow, and it would return a silently truncated summary
  the policy then trusts for the rest of the episode. Which bound is a policy field,
  `summary_input_cap`: the shipped `window` is the resolved prompt budget minus the summary template
  (including the elision marker the trim writes on top of its cap), so the folded transcript is
  summarized at its OWN size whenever it fits; the legacy `trigger` (`compact_share * guard`) is
  kept selectable because the published fold-step, trigger-collapse, and boundary-surface evidence
  was measured under it -- see [the summarize-input
  cap](summary-input-elision.md#the-summarize-input-cap-is-step-aligned). An empty summary is treated
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
window (no model spec, no served cap, no explicit budget, no probe) refuses nothing, the same rule
the context-ablation document lanes follow off the same resolution: an unknown model never silently
declares a prompt unusable. A refused prompt
is the one thing the loop builds that neither `complete` nor `chat` is handed, so `run_episode`
offers an optional `on_refused_prompt` observer for callers that must compare it -- inert on a run
that sends everything it builds, and used by the policy-change replay
([what a policy-constant change invalidates](policy-constant-audit.md#what-a-policy-constant-change-invalidates)).

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
(`tests/llb/bench/context_policy/test_agentic_context.py`,
`tests/llb/backends/test_served_window.py`), including the assertion that the `full` policy's prompt
is byte-identical to the pre-policy loop's, that no episode in any policy sends a prompt over the
resolved window, and that a declared window larger than a probed one is bound by the probe. The
constant-sweep lane's trim arithmetic and pin/expose verdicts are covered in
`tests/llb/bench/context_policy/test_agentic_context_sweep.py`.

CUDA host smoke (2026-07-29, MamayLM-Gemma-3-12B-IT-v2.0 on Ollama): after
`OllamaLauncher.ensure_num_ctx(8192)`, `/api/ps` reported `context_length=8192` and
`resolve_context_budget(..., probe=True)` with `--max-model-len 32768` bound to
`budget_source=served` / `served_max_model_len=8192`. A `make bench-agentic-context` pass with
`full,observation_cap` persisted those provenance fields on the agentic-context manifests.

`--base-url` routing (2026-07-29): unit test `test_drive_with_backend_routes_ollama_base_url_through_native_launcher`
in `tests/llb/backends/test_served_window.py` confirms that when `--base-url` resolves to the same
host as `ollama_host` and `--max-model-len` is set, `drive_with_backend` routes through
`OllamaLauncher` (native `/api/chat`) with the correct `num_ctx` instead of `local_complete`.

## Context-policy evidence on the 16 GB RTX 4060 Ti host

Run date 2026-07-28. `MamayLM-Gemma-3-12B-IT-v2.0` on Ollama, 24 tasks (the committed 4-task UA
seed plus 20 generated `search` tasks over the 250-document `ua_squad_postedited_v1` corpus, whose
observations run
20k-36k chars), all four policies in one ~46 min invocation, 377 model calls at 5.7 tok/s. Ollama
serves a 4096 window here regardless of the GGUF advertising 131072, so the run declared
`--max-model-len 4096` and the guard resolved a 9216-char prompt budget. One bundle per policy.

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

## Aggregate-safe trimming

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
same 24-task shape (4 seed + 10 `search-count` + 10 `search-locate` over 250 UA docs). Post-recovery
bundles (266 calls, 3.6 tok/s).

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
