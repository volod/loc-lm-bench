# The Context Budget: What A Retrieved Prompt Is Priced Against

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

A retrieved prompt is only as good as the window it arrives in. A backend that is handed more
context than it serves does not fail -- it truncates, answers from the surviving slice, and reports
nothing. Every guard in this repo that decides "does this prompt fit" reads the same arithmetic, and
this page is what that arithmetic is and what each caller does with the answer.

## The explicit budget knob

`RunConfig.context_budget` is an optional token budget that couples `top_k`, `chunk_size`, and
(for vLLM) `max_model_len`. When set, `fits_context` prunes configs whose estimated retrieved
prompt exceeds the budget, and multi-objective search samples the budget from
`{2048, 4096, 8192, 16384}` then sets `max_model_len` to that value on vLLM backends. Single-objective
`llb tune` leaves the budget unset unless the operator pins it in the run config.

## What the prune is priced against

The window arithmetic is `src/llb/backends/context_fit.py` -- one module, shared by the Optuna
prune, the eval runner, the context-ablation document lanes, and the agent-loop prompt guard, so
none of them can disagree about what fits on one host. `declared_max_context` is what the CONFIG
claims (host planner cap, roster `max_context`, `--max-model-len`, `--context-budget`);
`bound_max_context` takes the **MINIMUM of that and what the backend is probed as serving**, and
returns which side bound it. `fits_context` prices `top_k * chunk_size` against that bound.

The declared side alone is not enough, and on an Ollama host it is not even close: `num_ctx`
defaults to 4096 however large a window a GGUF advertises, so a sweep priced against the model card
keeps a trial whose prompt the backend truncates and then scores the truncated answer as that
configuration's quality. The whole sampled space is inside a declared 131072 window
(`top_k` 12 x `chunk_size` 1280 = 15,360 characters, about 5,760 tokens), so on such a host the
declared-only prune could never fire at all -- see [the measurement
below](#the-over-context-prune-was-a-no-op-on-an-unpinned-ollama-host-2026-08-24).

The served window is resolved **once per study** (`resolve_study_window`,
`src/llb/optimize/tuner_runtime.py`) rather than per trial: what a backend serves does not depend on
the RAG parameters a trial samples, while the declared side does -- `--context-budget` sampling
tightens it per trial, so a trial is always bound at least as tightly as the study-level reading.
Resolution warm-loads Ollama first (`probe_served_window`, `src/llb/backends/served_window.py`),
because `/api/ps` reports nothing until a request has loaded the model, and "unknown" is exactly the
answer that would leave the declared window bounding a run it cannot bound. The warm is sent even
when `/api/ps` already names a window, because Ollama keeps a previously loaded context until a
request asks for a different one -- a resident entry left by an earlier run answers for that run,
not for this study's `num_ctx`. `probe=False` (or an explicit `served_max_model_len`) is the
injected seam CI runs on.

Which window bound the study is recorded, never inferred: `declared_max_model_len`,
`served_max_model_len`, and `budget_source` go into `TuneResult.context_window`, into the
`context_window` block of `pareto.json`, and onto a line of `pareto.md` -- a pruned-trial count is
unreadable without it, since the same search space prunes nothing against a declared 128k window and
a large share of its trials against a served 4k one.

## The `rag` prompt is checked once, at run start

A `rag` overflow is a CONFIGURATION error, not a per-item outcome: `top_k * chunk_size` is the same
on every item, so either every prompt fits or none do, and skipping items would report a truncated
configuration as a corpus finding. `check_rag_prompt_window`
(`src/llb/executor/runner_setup.py`) therefore checks it ONCE, right after the backend starts, and
`run-eval` logs a warning naming the estimate, the bound window, and both sides of the bind. Every
run's `manifest.json` records the same three fields under `context_window`, so the window a score
was measured under is readable off the bundle.

It warns rather than refuses on purpose. `top_k * chunk_size` is an UPPER bound on a context that
short chunks, ACL filters, and reranking routinely make smaller: on the committed UA fixture a
`top_k=12` run estimates 4,224 tokens and the backend reports 2,691-2,950, so refusing on the
estimate would block runs that fit. The estimate is the right thing to WARN on and the wrong thing
to gate on.

## The over-context prune was a no-op on an unpinned Ollama host (2026-08-24)

Measured on the RTX 4060 Ti 16 GB CUDA host with Ollama, MamayLM-Gemma-3-12B-IT v2.0 GGUF Q4_K_M,
the committed UA fixture `samples/goldsets/ua_squad_postedited_v1/` (store at 800/120,
`intfloat/multilingual-e5-base`, `max_tokens=512`, 3 verified `final` items per run).

**The two windows differ by 32x.** The roster prices the model at `max_context` 131072; a warm probe
of `/api/ps` reports the backend serving **4096**, and `llb tune` logs the bind as
`prune window: 4096 tok (served; declared 131072, served 4096)`. Nothing in the sampled search space
can exceed a 131072-token window -- its widest point, `top_k` 12 x `chunk_size` 1280, is 15,360
characters or about 5,760 tokens -- so against the declared side the over-context prune could not
fire on any trial, ever. Against the served side the admissible context is
`(4096 - 512 headroom - 512 completion) * 3 = 9,216` characters, so every sampled point above that
prunes, the widest one included.

**A live sweep prunes on it.** A 12-trial single-objective `llb tune` on the same host and fixture
(`--limit 2`, seed 13, TPE) resolved the study window to `served 4096` and pruned trial 10 with
`retrieved context ~4437 tok exceeds the served window of 4096 tok` (`chunk_size` 384 x `top_k` 12 =
4,608 characters of context plus headroom and completion): 11 complete, 1 pruned of 12. Against the
declared 131072 that trial would have run, and its answer would have been scored on whatever
survived the cut. One prune in twelve understates the exposure rather than overstating it -- TPE
concentrated on the small-context region after the first few trials, so most of the sampled points
never approached the boundary.

**And the declared side was not even being read.** `llb tune` resolved its `model_spec` by matching
the manifest's top-level `source`/`name`, which no Ollama GGUF tag carries: the tag lives under a
roster entry's per-backend `sources`. The lookup returned None, the declared window resolved to
"cannot bound", and `fits_context` returned True unconditionally. Routing it through
`resolve_model_spec` / `candidate_sources` fixes the declared side; the served probe is what makes
the prune bind at all on this host.

**The truncation is real, silent, and it inverts the score.** Two `run-eval` bundles over the same
3 items, differing only in `top_k`:

| top_k | nominal context | estimate | observed `prompt_tokens` | objective |
| ---: | ---: | ---: | --- | --- |
| 12 | 9,600 chars | 4,224 tok | 2,928 / 2,691 / 2,950 | 0.25 / 0.00 / 0.67 |
| 24 | 19,200 chars | 7,424 tok | 2,051 / 2,051 / 2,051 | 0.00 / 0.00 / 0.00 |

At `top_k` 24 the backend reports the SAME prompt length on three different questions. That is not a
coincidence of three corpora slices, it is a cap: Ollama cut the prompt and never said so. Two of
the three answers became "Текст не містить інформації..." -- the model correctly reporting that the
evidence was not in the context it received, because the cut had removed it. Sending twice the
context made the measured score worse, and a sweep reading that as "top_k 24 is a bad configuration"
would have been reading a truncation artifact as a retrieval finding.

The `top_k` 12 row is the other half of the design: the estimate (4,224) says it does not fit and
the observed lengths (2,691-2,950) say it does, because retrieved chunks are shorter than
`chunk_size` and the 3 chars/token conversion is deliberately conservative. That row is exactly why
the `rag` check warns instead of refusing.

What would overturn this: an Ollama build whose `/api/ps` reports the GGUF window rather than the
served `num_ctx`, or a host with `OLLAMA_CONTEXT_LENGTH` raised to match the card -- in either case
`budget_source` reads `declared` and the two windows agree. The prune behaviour itself is pinned in
both binding directions by `tests/llb/optimize/test_tuner.py`, and the `rag` check by
`tests/llb/executor/test_rag_prompt_window.py`, so neither rests on this run.

The 3-item objective column is illustrative of the truncation, not a quality reading -- the
load-bearing number in the table is `prompt_tokens`, which the backend reports directly.
