# Summary-Input Bounds and Elision

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md). The trigger and fold-step geometry this subject
builds on is in [crossover geometry](crossover-geometry.md).

## The summarize-input cap is step-aligned

`make bench-agentic-context-compact-summary-input-cap` closes the one continuously moving cost term
left by the fold-step study. A compact episode must bound the summarize call because its input is the
transcript that just crossed the step trigger. The retired `trigger` bound moved with
`compact_share * guard`, so two guards that folded the same transcript at the same step still sent
different summarizer inputs. It also elided a transcript that would have fit the model window.

The shipped `summary_input_cap="window"` is the resolved prompt budget minus the summary template
and one worst-case elision marker. It is independent of `compact_share`; a folded transcript that
fits is summarized whole. The legacy `trigger` value remains selectable so older published designs
can pin and reproduce their original behavior. The committed comparison is
`samples/benchmarks/agentic_compact_summary_input_cap_design.json`.

The two-arm depth-10 ladder shares fold-step placement rules with the crossover study and refuses a
reference arm with no elision, a window arm that elides, different within-step folded inputs, or a
drifted fold-step boundary. `compact_fold_input_probe` settles those properties without a model.

CUDA evidence (2026-08-05, RTX 4060 Ti 16 GB): `mistral-small3.1:24b`, Ollama `num_ctx=8192`, seven
tasks per cell, and 10.63 tok/s. All eight cells completed 7/7 with one fold and zero overflows. The
aggregate is
`$DATA_DIR/agentic-compact-summary-input-cap/20260805T185837.832318Z-0f86b57558a1/manifest.json`.

| arm | cell | guard | fold step | offered | elided | compact tok | d(input tok) | side |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `trigger` | `cap-d10-step10-lo` | 20240 | 10 | 10494 | 374 | 26434.4 | -908.6 | compact |
| `trigger` | `cap-d10-step10-hi` | 22014 | 10 | 10494 | 0 | 26541.0 | -802.0 | compact |
| `trigger` | `cap-d10-step11-lo` | 22016 | 11 | 11802 | 794 | 28698.4 | +1355.4 | cap |
| `trigger` | `cap-d10-step11-hi` | 23040 | 11 | 11802 | 282 | 28878.6 | +1535.6 | cap |
| `window` | `cap-d10-step10-lo` | 20240 | 10 | 10494 | 0 | 26541.0 | -802.0 | compact |
| `window` | `cap-d10-step10-hi` | 22014 | 10 | 10494 | 0 | 26541.0 | -802.0 | compact |
| `window` | `cap-d10-step11-lo` | 22016 | 11 | 11802 | 0 | 28953.3 | +1610.3 | cap |
| `window` | `cap-d10-step11-hi` | 23040 | 11 | 11802 | 0 | 28953.3 | +1610.3 | cap |

Verdict: **the step-aligned cap is an exact step function**. The within-step residual falls from
180.1 tokens to 0.0 while the last compact-cheaper fold step stays at 10. The retired arm's elision
was free on this shape: removing up to 794 elided chars changed none of 28 paired completions. Its
shorter summarize prompts had nevertheless discounted compact by 106.6 tokens at step 10 and 180.1
at step 11.

```bash
make bench-agentic-context-compact-summary-input-cap
```

## Unavoidable elision under the shipped window bound

`make bench-agentic-context-compact-window-elision` enters the regime the cap-fitting ladder cannot:
the folded transcript is larger than the window after paying the summary template. The design
`samples/benchmarks/agentic_compact_window_elision_design.json` holds the trigger at 11200 chars and
folds the same 15402-char transcript once in both cells.

| role | guard | `compact_share` | folded input | elided |
| --- | ---: | ---: | ---: | ---: |
| transcript fits | 16200 | 0.691358 | 15402 | 0 |
| transcript elided | 14000 | 0.8 | 15402 | 2134 |

The probe predeclares those values. Live eligibility requires every episode to reproduce the input,
elision, and fold counts exactly, avoid overflow, and pass the fitting-control completion floor.

CUDA evidence (2026-08-11, RTX PRO 3000 Blackwell Laptop GPU, 12 GiB): `qwen3:14b`, four tasks,
23.22 tok/s. Both arms completed 4/4; all four exact pairs were unchanged. The 2134-char elision was
free on this typed-memory shape because the required early fact has a machine-preserved marker and
the removed middle contains padding and workflow checkpoints. The accepted aggregate is
`$DATA_DIR/agentic-compact-window-elision/20260811T102009.512985Z-38ad58e7a0dd/manifest.json`.

```bash
make bench-agentic-context-compact-window-elision
```

## Middle-critical transfer and entry-aware prototype

`make bench-agentic-context-compact-window-elision-transfer` replaces the protected early-memory
fact with ordinary answer evidence. The deterministic tasks put equal-length facts in three exact
strata of one 15776-char folded transcript. The treatment cap is 13268 chars: it retains head chars
through 7960, removes chars 7960 through 10468, and retains the tail after 10468.

| stratum | stage | answer span | head-tail treatment |
| --- | ---: | ---: | --- |
| head | 4 | 7094:7107 | retained |
| middle | 5 | 8847:8860 | elided |
| tail | 7 | 12353:12366 | retained |

Every task folds nine entries once. The fitting cell uses guard 16600 and
`compact_share=0.674698795180723`; the elided cell uses guard 14000 and `compact_share=0.8`. Both
resolve to trigger 11200 and offer the same transcript, while the treatment elides 2508 chars.
Validation independently reconstructs the exact summarizer transcript, locates every answer span,
and refuses boundary overlap, task-byte drift, trigger drift, or a nonidentical fold.

The first control pilot exposed an attention confound rather than elision: Qwen and Gemma4 each
completed 4/6 with both failures at stage 0, while Mamay-Gemma 12B completed 0/6. Moving the head
control to stage 4 kept it strictly inside the retained head but removed the absolute-first-line
confound. The rejected pilot remains at
`$DATA_DIR/agentic-compact-window-elision-transfer/20260811T110615.252883Z-b878af73e434/manifest.json`.

CUDA evidence (2026-08-11, same 12 GiB host): `qwen3:14b` at 22.24 tok/s and `gemma4:e4b` at
41.71 tok/s. Both qualified by completing every fitting-control task and reproduced the exact
geometry in every live episode.

| family | stratum | fitting control | head-tail elision | paired delta |
| --- | --- | ---: | ---: | ---: |
| Qwen | head | 2/2 | 2/2 | 0.0 |
| Qwen | middle | 2/2 | 0/2 | -1.0 |
| Qwen | tail | 2/2 | 2/2 | 0.0 |
| Gemma4 | head | 2/2 | 2/2 | 0.0 |
| Gemma4 | middle | 2/2 | 0/2 | -1.0 |
| Gemma4 | tail | 2/2 | 2/2 | 0.0 |

Verdict: **middle-critical elision costs completion across both qualified families**. Every retained
head/tail pair is unchanged, while the fitting control wins all four middle pairs and loses none.
This is the positional control the typed-memory task could not provide.

That result opens the predeclared prototype gate. The evidence-only `per_entry_head` strategy shares
the same summary-input budget across entries and retains the leading facts in each one. Against the
same elided guard, task bytes, and task ids, it recovers middle completion from 0/2 to 2/2 on both
families without changing any head/tail outcome. Both strategies send exactly 13992 summarizer
prompt chars per case, so the recovery is entry placement rather than extra context. The prototype
does not change the shipped `head_tail` default.

The accepted aggregate is
`$DATA_DIR/agentic-compact-window-elision-transfer/20260811T112518.130381Z-069978ce07e2/manifest.json`;
its aggregate measured throughput is 31.72 tok/s.

```bash
make bench-agentic-context-compact-window-elision-transfer
```

## Implementation map

| What | Where |
| --- | --- |
| Shipped bound, exact transcript renderer, and evidence-only entry-aware trim | `src/llb/bench/agentic/context_summary.py` |
| Generic deterministic task probe | `src/llb/bench/memory/boundary/probe.py` |
| Trigger-matched base runner and live byte eligibility | `src/llb/bench/memory/window_elision/run.py` |
| Head/middle/tail tasks and independent span placement | `src/llb/bench/memory/window_elision/tasks.py` |
| Transfer design and model-free gates | `src/llb/bench/memory/window_elision/transfer_design.py` |
| Two-family runner and conditional prototype | `src/llb/bench/memory/window_elision/transfer.py` |
| Per-stratum, transfer, and prototype readings | `src/llb/bench/memory/window_elision/transfer_reading.py` |
| Persistence and command | `src/llb/bench/memory/window_elision/transfer_report.py`, `src/llb/cli/bench/memory/window_elision_transfer.py` |
| Deterministic contracts | `tests/llb/bench/memory/test_agentic_memory_window_elision.py`, `tests/llb/bench/memory/test_agentic_memory_window_elision_transfer.py` |
