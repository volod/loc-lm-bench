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
tasks per cell, and 10.63 tok/s. All eight cells completed 7/7 with one fold and zero overflows.

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
the removed middle contains padding and workflow checkpoints. Reading: eliding 2,134 of 15,402
folded chars (13.9%) cost NOTHING measurable here -- but only because the typed marker survives
elision by construction, so this is evidence about the memory TYPE, not about elision in general.
The positional control the shape cannot give is the transfer run below. Boundary: four tasks and
4/4 completion in both arms is a ceiling, so the run can show no loss and could not have shown a
small one. Lookup key: run `agent-context-policy-summary-elision-under-window-bound`, run id
`38ad58e7a0dd`.

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
confound. That rejected pilot is retained as evidence of the refusal, not of elision (lookup key:
run `agent-context-policy-middle-critical-window-elision-transfer`, run id `b878af73e434`).

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

That result opens the predeclared prototype gate. The `per_entry_head` strategy shares the same
summary-input budget across entries and retains the leading facts in each one. Against the same
elided guard, task bytes, and task ids, it recovers middle completion from 0/2 to 2/2 on both
families without changing any head/tail outcome. Both strategies send exactly 13992 summarizer
prompt chars per case, so the recovery is entry placement rather than extra context. The prototype
did not change the shipped `head_tail` default; what it became, and what the audit says a default
change would cost, is [the policy choice below](#entry-aware-summary-folding-as-a-policy-choice).

What would overturn the verdict: a family that keeps middle completion under `head_tail`, or a
middle stratum whose answer span survives the cut by accident -- validation refuses boundary
overlap precisely so the second cannot happen silently. The accepted aggregate measured 31.72 tok/s
across both families. Lookup key: run
`agent-context-policy-middle-critical-window-elision-transfer`, run id `069978ce07e2`.

```bash
make bench-agentic-context-compact-window-elision-transfer
```

## Entry-aware summary folding as a policy choice

`make bench-agentic-context-summary-trim-adoption` promotes the prototype above into a public
context-policy field and decides whether it should replace the shipped fold. **The answer measured
here is: ship it as a supported option, keep `head_tail` as the default.** It costs nothing and
loses nothing on any workload, it recovers every middle-critical case the shipped trim could not
finish, and a default change would retire no published cell -- but one of the two families cannot
put its whole declared middle stratum into the folding regime, so the recovery rests on 6 of 8
declared middle cases rather than on all of them. That last gap is now a POWER limit and nothing
else: the arms run task-adjacent in a balanced order, so it can no longer be read as an artifact of
which arm went second.

`ContextPolicy.summary_trim_strategy` is now a validated choice like `summary_input_cap`
(`head_tail`, the shipped default, and `per_entry_head`), pinned in
`samples/benchmarks/agentic_context_policy_pins.json`, audited by the policy-change replay, and
enumerated in the coupling table -- so the field can no longer move without the pin gate naming
what it retires.

**The field is readable only where a fold ELIDES.** Both strategies return the offered transcript
untouched while it fits the summarize-input cap, so where nothing is cut they render byte-identical
prompts. That single property answers every pair the field is in: the six new couplings report
`no_geometry` or `independent`, and none of them opens a separating band, because a partner field
that turns an un-elided fold into an elided one has already moved the summarize prompt by itself
(`summary_trim_strategy x compact_share` and `x summary_input_cap` state that as measured elision
counts rather than as an assertion).

The study runs five workloads, each predeclaring its fold count, offered transcript, elision, and
the summarize prompt bytes BOTH strategies spend -- all measured with an oracle controller and no
model, so a workload that stops producing its regime fails the design gate rather than the reading:

| workload | tasks | guard | share | folds | offered | elided | `head_tail` chars | `per_entry_head` chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| typed memory | 4 | 14000 | 0.8 | 1 | 15402 | 2134 | 13992 | 13992 |
| aggregate search | 3 | 8000 | 0.8 | 1 | 8732 | 1464 | 7992 | 7992 |
| repeated fold | 2 | 7000 | 0.8 | 2 | 10373 | 659 | 11131 | **9967** |
| crossover control | 2 | 20240 | 0.5 | 1 | 10494 | 0 | 11188 | 11188 |
| middle-critical | 12 | 14000 | 0.8 | 1 | 15776 | 2508 | 13992 | 13992 |

The crossover row is the published fold-step cell `fold-d10-step10-lo` at its own guard, share,
depth and padding; its offered 10494 chars is the same number the step-aligned cap table above
reports. It elides nothing, so it is a CONTROL that must stay byte-identical rather than evidence.
The repeated-fold row is the only geometry where the two strategies spend different byte totals,
and the entry-aware trim spends 1164 FEWER chars per episode: it hands each entry a fixed share of
the budget and does not redistribute what a short entry leaves over, so it can undershoot the cap
that `head_tail` fills exactly. Adoption therefore never costs summary bytes on any measured
geometry.

Pairing is exact and per case. The two arms are byte-identical up to and including the transcript
the first fold offers the summarizer, so a case pairs when both arms fold the same offered bytes;
anything else is reported as an unpaired case instead of being averaged into a delta. That is what
makes the aggregate-search workload readable at all -- its walk is not forced by a token chain, so a
live model may search a different number of terms, and only the per-case pairing survives that.

An unpaired case has two kinds, and the study treats them differently because only one of them can
be about the trim. **A case where an arm never FOLDS ran no trim in that arm at all** -- a replay
shows the two arms building byte-identical prompts up to the fold, so it diverged upstream of the
strategy -- and it is excluded from the delta, counted, and named. The exclusion cannot be
correlated with the arm, and the case stays in the completion RATE, so an arm cannot buy a better
rate by ending episodes early. **A case
where BOTH arms fold and still offer different bytes** is downstream of a trim that did run, is not
separable from the treatment, and refuses the whole workload reading. Excluding a case also costs
the evidence something, so a run whose middle stratum drops below its declared size in usable pairs
is reported as UNDER-POWERED rather than as a result in either direction.

### Arm order is balanced, not fixed

Both arms of ONE task run back to back, and which arm opens a task alternates with the task index;
the rotation carries across workloads and its phase flips per family, so the single leftover first
position an odd task count leaves over cancels over the run. Under the fixed arm blocks this study
first ran -- every episode of `head_tail`, then every episode of `per_entry_head` -- "ran second"
and "ran under the candidate trim" were the same column, and an episode that left the folding
regime in the second arm alone could not be attributed to either. The schedule is declared in the
design (`arm_order`) and validated, and the executed order is persisted per episode, so the balance
is auditable rather than asserted.

The check the schedule buys is read on the FOLDING channel, not on completion. Whether an episode
reaches its first fold is decided before the arms can diverge -- they build byte-identical prompts
up to and including the transcript that fold offers -- so a gap there cannot be the treatment and
is the serving stack by elimination. Completion is the opposite: it is the treatment's own outcome,
and it moves with position whenever an arm's wins fall unevenly across the two slots, so reading a
position effect off it would report the recovery itself as a scheduling artifact.

**Which other lanes inherit the seam.** The schedule lives in
`llb.bench.context_policy.interleave`, not in this study, because every paired agentic comparison
on this host drives one stateful endpoint. Three lanes still run fixed arm blocks, and they are not
equally exposed. `compact_vs_cap` walks the observation-cap policy and then the compact policy over
one task set and pairs them per case with nothing gating the second arm, so it is the one that can
adopt the balanced schedule as-is. The repeated-fold completion lane runs its two mechanism arms as
blocks WITHIN each cell, which is interleavable, but its cell ladder is ordered by a control gate
that stops the run when the one-fold control fails. The window-elision base runner is the same
shape one level up: its elided arm runs only if the transcript-fitting control passed. For those
two, arm order is a sequencing DECISION rather than an unexamined default, so removing it needs a
design that keeps the gate, not this helper -- and until one exists, a dropout in their second
block carries the same ambiguity this study just removed from its own.

### The measured comparison

CUDA evidence (2026-08-29, RTX 4060 Ti 16 GB, balanced arm order): `qwen3:14b` at 19.46-20.77
tok/s and `gemma4:e4b` at 48.69-52.34 tok/s, Ollama `num_ctx=8192`, 23 tasks per arm per family, 92
episodes, run twice. Both families qualified by completing the elision-free crossover control 2/2
with zero overflows.

| family | workload | pairs | skipped | ea wins | ht wins | d(model-input chars) | d(summary chars) | d(folds) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen | typed memory | 4 | 0 | 0 | 0 | +225 | 0 | 0 |
| Qwen | aggregate search | 3 | 0 | 0 | 0 | -2 | 0 | 0 |
| Qwen | repeated fold | 2 | 0 | 0 | 0 | -2653 | **-2029** | 0 |
| Qwen | crossover control | 2 | 0 | 0 | 0 | **0** | 0 | 0 |
| Qwen | middle-critical | 12 | 0 | **5** | 0 | +634 | 0 | 0 |
| Gemma4 | typed memory | 4 | 0 | 0 | 0 | -183 | 0 | 0 |
| Gemma4 | aggregate search | 3 | 0 | 0 | 0 | -13221 | 0 | 0 |
| Gemma4 | repeated fold | 2 | 0 | 0 | 0 | -1754 | **-1828** | 0 |
| Gemma4 | crossover control | 2 | 0 | 0 | 0 | +78 | 0 | 0 |
| Gemma4 | middle-critical | 12 | 0 | **2** | 0 | -63 | 0 | 0 |

Verdict: **the entry-aware fold costs nothing and recovers middle-critical completion, but the
evidence does not carry a default change.** `head_tail` wins ZERO paired cases anywhere -- across
two families, five workloads and 46 paired cases -- so there is no regression to trade against. The
recovery reproduces per stratum:

| family | head | middle | tail |
| --- | --- | --- | --- |
| Qwen | 4/4 -> 4/4 | **0/4 -> 4/4** | 3/4 -> 4/4 |
| Gemma4 | 4/4 -> 4/4 | **0/2 -> 2/2** (2 of 4 declared never fold) | 4/4 -> 4/4 |

Read the numbers in operator terms. Every middle case the shipped trim could not finish, the
entry-aware trim finished -- 4 of 4 on Qwen, 2 of 2 usable on Gemma4 -- while head and tail never
moved except upward (Qwen's tail gained one case). Summary prompt bytes are EXACTLY equal on every
single-fold workload and 1828-2029 chars cheaper on the repeatedly folding one, so the recovery is
bought with entry placement rather than with window.

**The order is balanced and the dropout stayed.** Both families opened their tasks with each arm
within one task of evenly (12/11 and 11/12 of 23), and the pre-divergence channel is flat in both:
Qwen reached the fold in 23 of 23 first-position and 23 of 23 second-position episodes, Gemma4 in
21 of 23 either way. Position moves nothing about entering the regime under test, and Qwen's 5
paired losses are all `head_tail` losses spread across BOTH slots -- it loses two of them running
first and three running second -- so the recovery is a property of the arm, not of the schedule.

The crossover control is what makes the `d(model-input chars)` column readable: it elides nothing,
so both arms render byte-identical summarize prompts there. Under fixed arm blocks it still moved
+76 and +298 chars; running the two arms back to back on the same task shrinks that to **0 on Qwen
and +78 on Gemma4**, because the pair now sees nearly the same request history. That is the run's
noise floor -- a served endpoint returns different continuations for identical prompts depending on
the requests before them -- and every non-repeated-fold row in the table sits inside or near it,
with only the repeated-fold savings an order of magnitude clear. Gemma4's aggregate-search -13221
is not a trim effect either: it is one episode taking a shorter walk.

**Why this is an option and not a default.** Gemma4 still puts only 2 of its 4 declared middle
cases into the folding regime -- and under the balanced schedule the shortfall is no longer
attributable to the arms. Both dead cases now end the token chain at step 7 in BOTH arms and in
BOTH positions: `window-elision-m-001-d10` fails with `per_entry_head` opening it and `head_tail`
second, `window-elision-m-002-d10` fails with `head_tail` opening it and `per_entry_head` second.
Under the fixed order, one of those two dropped in the entry-aware arm alone, which is exactly the
reading the confound made unusable; with order balanced, it drops in both. Widening the stratum
from two to four cases per stratum did not fix it either -- the shortfall scaled with the set -- so
what remains is a per-family property of how far Gemma4 walks these two tasks, not an artifact of
which arm ran when. The recovery is therefore established on the 6 of 8 declared middle cases that
enter the regime, which is enough to OFFER the strategy and not enough to move a default every
later run inherits.

What would overturn this, and what would settle it: a family that loses a paired case under
`per_entry_head` overturns the safety claim; a fully powered middle stratum on both families --
which now needs Gemma4 to walk `m-001` and `m-002` far enough to fold, not a schedule change -- is
what a default change is waiting on. Lookup key: run
`agent-context-policy-entry-aware-summary-fold-adoption`, two consecutive balanced-order runs on
this host reproduced every table entry above exactly, including which cases dropped out and in
which position.

### What a default change would cost

The study is not allowed to recommend a default change on its own measurements, so it runs the
model-free policy-change audit under the PINNED policy and takes the answer as a gate. Under the
shipped `window` bound, moving `summary_trim_strategy` from `head_tail` to `per_entry_head` is
prompt-invariant on all 27 published cells and retires nothing; no registered published value
declares the field either.

The same audit read against the designs' own `held_fixed` -- which still records the retired
`trigger` bound -- invalidates 4 cells (`surface-d10-g23000`, `fold-d10-step10-lo`,
`fold-d10-step11-lo`, `fold-d10-step11-hi`). Those are exactly the cells whose trigger-bound cap
elided (374, 794 and 282 chars in the step-aligned table above), which is the same mechanism read
from a direction that owes it nothing: the field bites where, and only where, something is cut.

```bash
make bench-agentic-context-summary-trim-adoption
# the model-free half alone -- geometry plus the audit, no GPU:
make bench-agentic-context-summary-trim-adoption AGENT_CONTEXT_SUMMARY_TRIM_ADOPTION_AUDIT_ONLY=1
```

## Implementation map

| What | Where |
| --- | --- |
| Shipped bound, exact transcript renderer, and both trim strategies | `src/llb/bench/agentic/context_summary.py` |
| The trim strategy as a validated policy field | `src/llb/bench/agentic/context_policy.py`, `samples/benchmarks/agentic_context_policy_pins.json` |
| Why no partner constant can separate the compound audit reading on it | `src/llb/bench/policy_change/interaction/trim.py` |
| Generic deterministic task probe | `src/llb/bench/memory/boundary/probe.py` |
| Trigger-matched base runner and live byte eligibility | `src/llb/bench/memory/window_elision/run.py` |
| Head/middle/tail tasks and independent span placement | `src/llb/bench/memory/window_elision/tasks.py` |
| Transfer design and model-free gates | `src/llb/bench/memory/window_elision/transfer_design.py` |
| Two-family runner and conditional prototype | `src/llb/bench/memory/window_elision/transfer.py` |
| Per-stratum, transfer, and prototype readings | `src/llb/bench/memory/window_elision/transfer_reading.py` |
| Persistence and command | `src/llb/bench/memory/window_elision/transfer_report.py`, `src/llb/cli/bench/memory/window_elision_transfer.py` |
| Adoption workloads, their oracles, and the aggregate-search task family | `src/llb/bench/summary_trim/workloads.py`, `src/llb/bench/summary_trim/tasks.py` |
| Balanced arm schedule shared by any multi-arm policy lane | `src/llb/bench/context_policy/interleave.py` |
| Adoption design gate, run, readings, verdict, and persistence | `src/llb/bench/summary_trim/design.py`, `run.py`, `reading.py`, `adoption.py`, `analysis.py`, `report.py` |
| Adoption command | `src/llb/cli/bench/context/summary_trim_adoption.py` |
| Deterministic contracts | `tests/llb/bench/memory/test_agentic_memory_window_elision.py`, `tests/llb/bench/memory/test_agentic_memory_window_elision_transfer.py`, `tests/llb/bench/summary_trim/test_agentic_summary_trim_adoption.py`, `tests/llb/bench/context_policy/test_agentic_arm_interleave.py` |
