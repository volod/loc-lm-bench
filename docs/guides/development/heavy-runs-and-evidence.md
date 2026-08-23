# Heavy Runs and Their Evidence

Read this before a real-model run on this host, while winding one down, before writing a measured
result into the delivered docs. [AGENTS.md](../../../AGENTS.md) carries the binding one-line norms;
the procedure is here so it costs nothing when you are not doing these things.

## Choosing a model for a local run

- **Local evidence models:** When running local heavy / evidence / acceptance pipelines on this
  machine (e.g. multi-objective RAG tune, joint search, auto-rag, host validation with real
  backends), pick the strongest local model that fits the host -- never use tiny smoke models
  under 7B parameters (no `llama3.2:3b`, `0.5B`, `1B`, `3B`, etc.) unless the user explicitly
  asks for a tiny model.
- **Model selection order:** 1. Prefer UA-capable instruct models already on the
  host (Ollama / vLLM / llama.cpp). 2. Prefer the largest parameter count that fits the GPU with
  RAG headroom (KV cache + embedder + optional reranker). On a ~16 GiB GPU, prefer ~12B-14B class
  over 24B+ GGUF that barely fits and thrash-swaps. 3. Floor: at least 7B parameters for any
  real-model evidence run recorded in `docs/impl/current/`. 4. CI / `make ci` fixtures and
  injected fakes stay unchanged -- this rule is for live local backend runs only.
- **Model examples (12-16 GiB class):** Prefer: MamayLM-Gemma v2.0 GGUF, `gemma4:31b`,
  `qwen3.8:27b`, when present -- the Qwen lane runs its CURRENT generation for new evidence, and
  `qwen3.6:27b` is kept only for a deliberate generation comparison against an older reading
  (`make list-model-families` says which generation is current). Avoid for evidence:
  `llama3.2:3b` and other sub-7B tags.

## Citing a measured result

Evidence must stay checkable on a machine that never held the run, so a delivered page cites
neither a `$DATA_DIR/<method>/<run-id>/` path (host-local, deleted, absent on the other GPU hosts)
nor a bare run label like `20260815T-bare-id-squad-cos060` (a lookup key that identifies nothing).
A `$DATA_DIR` TEMPLATE describing where a command writes is fine -- that documents the tool.

Write instead:

- **What ran, on what, and when.** The command, the corpus and gold set by name with item counts,
  the model and backend, the knobs the reading depends on (k, seed, resamples, served window, store
  shape), the date, and the host in recognisable terms ("RTX 4060 Ti 16 GB CUDA host").
- **Every load-bearing number, and its READING.** Deltas with intervals, the win/loss/tie ledger,
  discordant counts, n per slice -- then what they mean: direction, size in operator terms, whether
  the gate is cleared, what it licenses. `-0.081 [-0.157, -0.017]` is data; "makes multi-hop answers
  measurably worse on this model, on 21 differing items, which clears the minimum-evidence gate" is
  the citation.
- **What would overturn it,** and the boundaries -- grounding, gate relabelings, timeouts, confounds.
- **Both sides of a comparison, in words.** "byte-identical to the pre-interning sweep over the same
  24 bundles", never "(`<tag-a>` against `<tag-b>`)".

Never cite a run by describing what it WROTE: if the 150 verdicts in a `summary.json` matter, say
what they showed. A run label may TRAIL a description as a lookup key, never replace one.

**Code version is recovered from the DATE, not written into the page.** A run executes against the
working tree, so the commit holding that code does not exist yet while you write the evidence --
any hash you write names the parent commit or is a guess. Do not lean on the commit message either:
70 of this repo's 455 commits carry no ticket tag. What always works is the date the page states,
against git's own timestamps -- `git rev-list -1 --before='<date> 23:59' <branch>` is the tree the
run executed on, whatever anyone wrote in a message. So the page owes an accurate DATE, and nothing
else about code, EXCEPT where the run diverges from what landed -- measured before a fix, or
evidence predating a refactor. That divergence is the one thing no git command can show a reader.

When the run is on a host you are not on, you cannot inline numbers you cannot read: leave that
citation alone and route the conversion as a task for the host that holds it.

## Leaving the host clean after a run

- **Background shells:** list every background task this session started and confirm each has
  EXITED. Stop the ones that have not (`TaskStop`), and say so in the report if any was killed
  rather than finished. A task still running is not a finished task.
- **Do not arm a poller for work the harness tracks:** a backgrounded command notifies on exit, so a
  second shell that waits on it is pure waste. Poll only for state the harness cannot see (an
  external queue, a remote job).
- **A watcher must not match itself:** `pgrep -f` / `pgrep -a` match the FULL command line, so a
  waiter that greps for the command it is waiting on finds its own shell and loops forever. Match on
  the process name, the PID, or a pattern the watcher's own command line does not contain.
- **Host processes:** no model server, run, or tool this task started may be left holding the GPU or
  a port unless the user asked for it to stay up. Check with `nvidia-smi` after any heavy local run.
- **Temporary artifacts:** throwaway roots, scratch scripts, half-written fixtures, and debug output
  go to the session scratchpad directory or are deleted -- never left in the repo, `src/`, or
  `samples/`.
