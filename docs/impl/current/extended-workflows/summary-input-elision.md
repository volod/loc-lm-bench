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
here is: the evidence now carries a default change.** It costs nothing and loses nothing on any
workload, it recovers every middle-critical case the shipped trim could not finish, a default
change would retire no published cell, and the power limit that used to hold the verdict at
"option" is gone -- both families now read all four of their declared middle cases, not 6 of 8, and
every one of them recovers. The study's verdict is `adopt_entry_aware_as_the_shipped_default`.

It does not itself move the default. `changes_shipped_default` stays false and
`ContextPolicy.summary_trim_strategy` still ships `head_tail`, because moving a policy default is a
product change that runs through the pin gate and the published-value scope, not something a
measurement run performs on its own authority.

What closed the power gap was the WORKLOAD, not the guard.
[The per-family fit](#the-guard-is-fitted-per-family-and-the-fold-lands-inside-the-walk) had
already established that no guard could reach the two cases Gemma4 walked short of, because the
middle-critical set grew its transcript too slowly for any usable guard to fold before step 10. The
set now pads three times as fast, folds at step 7, and every declared case of both families reaches
it.

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
| middle-critical | 12 | 9500 | 0.8 | 1 | 16764 | **7996** | 9492 | 9492 |

The crossover row is the published fold-step cell `fold-d10-step10-lo` at its own guard, share,
depth and padding; its offered 10494 chars is the same number the step-aligned cap table above
reports. It elides nothing, so it is a CONTROL that must stay byte-identical rather than evidence.
The repeated-fold row is the only geometry where the two strategies spend different byte totals,
and the entry-aware trim spends 1164 FEWER chars per episode: it hands each entry a fixed share of
the budget and does not redistribute what a short entry leaves over, so it can undershoot the cap
that `head_tail` fills exactly. Adoption therefore never costs summary bytes on any measured
geometry.

The middle-critical row cuts nearly half of what it folds: 7996 of 16764 offered chars, 48% of the
transcript, against 6-17% on the other three eliding workloads. That is
a DECLARED position in the elision regime rather than a drift -- the padding was raised to 3200
chars precisely so the transcript outgrows the window bound after five entries -- so the recovery
this workload measures is a recovery from a much deeper cut than the typed-memory or
aggregate-search rows measure, and the numbers are not interchangeable with theirs.

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

CUDA evidence (2026-08-29, RTX 4060 Ti 16 GB, balanced arm order): `qwen3:14b` at 23.42-23.49
tok/s and `gemma4:e4b` at 55.67-55.72 tok/s, Ollama `num_ctx=8192`, 23 tasks per arm per family, 92
paired episodes plus 12 walk-control episodes per family, run twice. Both families qualified by
completing the elision-free crossover control 2/2 with zero overflows.

| family | workload | pairs | skipped | ea wins | ht wins | d(model-input chars) | d(summary chars) | d(folds) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen | typed memory | 4 | 0 | 0 | 0 | +225 | 0 | 0 |
| Qwen | aggregate search | 3 | 0 | 0 | 0 | -2 | 0 | 0 |
| Qwen | repeated fold | 2 | 0 | 0 | 0 | -2653 | **-2029** | 0 |
| Qwen | crossover control | 2 | 0 | 0 | 0 | **0** | 0 | 0 |
| Qwen | middle-critical | 12 | 0 | **4** | 0 | +5615 | 0 | 0 |
| Gemma4 | typed memory | 4 | 0 | 0 | 0 | -183 | 0 | 0 |
| Gemma4 | aggregate search | 3 | 0 | 0 | 0 | -13221 | 0 | 0 |
| Gemma4 | repeated fold | 2 | 0 | 0 | 0 | -1754 | **-1828** | 0 |
| Gemma4 | crossover control | 2 | 0 | 0 | 0 | +78 | 0 | 0 |
| Gemma4 | middle-critical | 12 | 0 | **4** | 0 | -6590 | 0 | 0 |

Verdict: **the entry-aware fold costs nothing, recovers every declared middle-critical case, and
the evidence now carries a default change.** `head_tail` wins ZERO paired cases anywhere -- across
two families, five workloads and 46 paired cases -- so there is no regression to trade against, and
nothing is skipped: all 46 cases pair in both families. The recovery reproduces per stratum, and
this time on the full declared set:

| family | head | middle | tail |
| --- | --- | --- | --- |
| Qwen | 4/4 -> 4/4 | **0/4 -> 4/4** | 4/4 -> 4/4 |
| Gemma4 | 4/4 -> 4/4 | **0/4 -> 4/4** | 4/4 -> 4/4 |

Read the numbers in operator terms. The shipped trim finished NONE of the eight declared
middle-critical cases across the two families; the entry-aware trim finished all eight, while head
and tail stayed at 4/4 in both. Summary prompt bytes are EXACTLY equal on every single-fold
workload and 1828-2029 chars cheaper on the repeatedly folding one, so the recovery is bought with
entry placement rather than with window.

**Why the middle-critical byte column is large, and why that is not a cost.** The crossover control
is what makes the column readable: it elides nothing, so both arms render byte-identical summarize
prompts there, and it moved 0 chars on Qwen and +78 on Gemma4 -- what a served endpoint returns for
identical prompts under slightly different request histories. That is the run's noise floor, and
the typed-memory and Qwen aggregate-search rows sit inside it. Three rows do not, for three
different reasons. The repeated-fold row's -1754 and -2653 travel with its real summary-byte saving
below. Gemma4's aggregate-search -13221 is one episode taking a shorter walk, on the one workload
whose walk is not forced by a token chain. And the middle-critical row's +5615 and -6590 are the
recovery itself showing up in the byte column: on four cases per family the two arms now END
DIFFERENTLY, so their continuations after the fold are genuinely different episodes. The sign is a per-family
property of what a failing episode does -- Qwen's `head_tail` failures give up sooner than the
recovered episode runs, Gemma4's run longer -- which is exactly why model-input chars are reported
beside the cost gate and not inside it. Gating on them would price a recovered case as a cost.
Summary prompt chars, which the gate does read, are unchanged at 0 on every eliding single-fold
workload.

**The order is balanced and nothing dropped out.** Both families opened their tasks with each arm
within one task of evenly (12/11 and 11/12 of 23), and the pre-divergence channel is flat and FULL
in both: Qwen and Gemma4 each reached the fold in 23 of 23 first-position and 23 of 23
second-position episodes. Under the slower-growing shape Gemma4 managed only 21 of 23 either way;
the two episodes it used to lose were the ones that ended before any fold, and at a fold on step 7
they no longer do. Completion is 21 of 23 in each position for both families -- the two
non-completions per slot are `head_tail`'s middle-critical failures, split evenly across the slots,
so the recovery is a property of the arm and not of the schedule.

What would overturn this: a family that loses a paired case under `per_entry_head` overturns the
safety claim; a middle stratum whose answer span survives the cut by accident would overturn the
recovery, and validation refuses boundary overlap precisely so that cannot happen silently. What
this does NOT establish is the shape-independence of the recovery -- it is measured at one point in
the elision regime (48% of the folded transcript cut) on one fact-stage triple, and the shallower
typed-memory and aggregate-search cuts are where it is measured to cost nothing, not where it is
measured to recover. Lookup key: run
`agent-context-policy-entry-aware-summary-fold-adoption`, run ids `72d6e04094ce` and
`568179767d8f`; two consecutive balanced-order runs on this host reproduced every table entry above
exactly -- every delta, every stratum count, and both guard fits.

### The guard is fitted per family, and the fold lands inside the walk

One shared character guard is not one shared regime. The middle stratum is readable only where an
episode REACHES its fold -- both arms build byte-identical prompts up to the transcript the fold
offers -- so a family whose walk ends before the trigger is crossed ran no trim at all, and the
constant was silently measuring per-family walk length. The middle-critical workload's guard is
therefore fitted per family from that family's own measured walk, the way the repeated-fold ladder
fits its guard from the family's measured fold length; the band arithmetic both fits use (declared
guard wins any tie, centre of the widest run otherwise) is one module.

The measurement is a WALK CONTROL: the same twelve tasks at a guard the model-free probe never
folds at (20000 chars), which makes its prompts byte-identical to what every candidate guard builds
before its own fold. It runs first, per family, and is not an arm of anything -- nothing pairs
against it.

**The band is filtered by the declared REGIME before it is scored.** A lower guard folds earlier,
but the summarize-input bound is the same window, so it also folds a shorter transcript against a
smaller cap and moves the span the cap elides. Each of the 45 candidates in the declared band
(4000-15000 chars, step 250) is walked with an oracle and either usable or refused by name:

| candidate guards | folds | fold step | offered | elided | usable |
| --- | ---: | ---: | ---: | ---: | --- |
| 4000 | 2 | 2 | 4557 | 84 | no -- it folds more than once |
| 4250-4750 | 1 | 2 | 3352 | 0 | no -- the fold fits the bound and elides nothing |
| 5000-8250 | 2-5 | 3-6 | 20178-30420 | 7642-15201 | no -- it folds more than once |
| 8500-9250 | 1 | 6 | 13411 | 4893-5643 | no -- the fold stops before the tail fact's stage |
| **9500-10250** | **1** | **7** | **16764** | **7246-7996** | **yes** (9500 is the declared guard) |
| 10500-14750 | 1 | 8-11 | 20117-30176 | 9349-16908 | no -- an answer fact leaves its declared stratum |
| 15000 | 0 | -- | 0 | 0 | no -- the trigger is never crossed, so nothing folds |

Four of 45 candidates survive, and they all fold at step 7. FOUR different properties refuse the
rest, and the table shows where each bites. At the bottom the guard is small enough that the
episode folds repeatedly, or small enough that the one fold it does fits the bound and cuts
nothing. Just below the usable run the fold is a single fold that does elide -- but it stops after
four entries, and the tail fact sits at stage 4, so the fold happens before the fact exists to be
placed at all. Above the usable run the fold contains every fact and the trim boundaries move under
them instead: the retained tail is 0.4 of a cap that grows with the guard, so at six folded entries
the tail fact is no longer in the tail. The band was drawn wide enough to contain every guard that
folds at all -- from one that folds twice to the first that never crosses the trigger -- precisely
so all four are measured rather than assumed outside it.

**The two placement refusals are now reported apart, and at this shape it matters.** They were one
reason before, because on a slower-growing transcript only the boundary-movement one ever bounded
the band -- so the floor was always named "an answer fact leaves its declared stratum" and nobody
had to ask which way. Here the floor is set by the other one:
`an_answer_fact_is_not_inside_the_folded_transcript_yet` says the fold stops before the stage that
plants the fact, so nothing was placed anywhere, while
`an_answer_fact_leaves_its_declared_elision_stratum` says the fold does contain the fact and the
trim boundaries moved out from under it. Both are facts about the shape rather than the family, but
they point at different repairs -- more stages before the fold against a different stage inside it
-- and a floor that cannot say which is a floor nobody can act on.

**The band offers one fold step, so the fit cannot trade.** Every usable guard folds at step 7, so
`select_guard`'s tie rule returns the declared guard for any family and the fit can only confirm it
or report that the walk did not reach it. That is a property of the workload's padding rather than
of the fit: the retained tail is about one entry wide at this fold (0.4 of an 8768-char cap against
3353-char entries), so a guard one step later has already pushed the tail fact out of it. The
reading states the reach (`band_fold_steps`) rather than leaving it to be inferred from the guard
list, because a one-step band is what makes a shortfall unfixable by any guard.

CUDA evidence (2026-08-29, RTX 4060 Ti 16 GB): `qwen3:14b` at 23.42-23.49 tok/s and `gemma4:e4b` at
55.67-55.72 tok/s, Ollama `num_ctx=8192`, 12 walk-control episodes plus 46 paired episodes per
family, run twice. The two runs agree on every entry below and on which four guards the band can
use.

| family | walk control | measured walk | fitted guard | fold step | cases reaching the fold |
| --- | ---: | --- | ---: | ---: | ---: |
| Qwen | 12/12 | 11 on every case | 9500 (unchanged) | 7 | 12/12 |
| Gemma4 | 12/12 | 11 on every case | 9500 (unchanged) | 7 | 12/12 |

Verdict: **the fit moves no guard, and this time that is because the declared one already reaches
every case.** Both families complete the walk control 12 of 12 with zero folds and walk all 12
tasks to the full 11 steps, in both runs; every declared case of both families therefore crosses a
trigger that now sits at step 7 instead of step 11. The short walks are gone TWICE OVER, and the
two reasons are worth keeping apart because only the second is guaranteed by construction. Gemma4's
early finish did not reproduce at the re-staged tasks -- these are different tasks with different
codes and stages, and the walk control measures that it walks them whole rather than explaining
why. What the shape does guarantee is the other half: a walk that ended at step 7, as those two
cases did, now reaches a fold that happens AT step 7, so the same behavior would no longer cost the
stratum a case.

That is the same fit reporting a different answer because the WORKLOAD moved under it, which is
what the fit was built to make visible. Held at the slower-growing shape the fit had exhausted its
band and said so: no usable guard folded before step 10, Gemma4 ended two cases at step 7, and the
recovery rested on 6 of 8 declared middle cases. The obstacle was never the guard -- observations
are capped at 800 chars while that shape's folded transcript grew 1753 chars a step, so the
transcript could not outgrow the window bound before step 8 and nothing earlier elided anything to
be critical about. Padding to 3200 chars a step moves the whole ladder: the transcript outgrows the
bound after five entries, and the fold lands inside a walk every family was already taking.

What would overturn the fit itself: a usable guard below step 7, which would mean the placement
check had let a fact out of its stratum, or a family whose walk control does not complete -- which
would put the shortfall back and, with a one-step band, put it beyond any guard's reach.

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

**The gate is now open, and the default has not moved through it.** Every condition the ladder
asks for is met -- no regression, a whole middle stratum in both families, no extra summary bytes,
a balanced order, and an audit that retires nothing -- so the study returns
`adopt_entry_aware_as_the_shipped_default`. What it does NOT do is perform the change:
`changes_shipped_default` is false in the design and in the persisted analysis, and
`ContextPolicy.summary_trim_strategy` still ships `head_tail`. A measurement lane recommending a
product default is not the same act as moving it, which runs through the pin gate and the
published-value scope; until that lands, an operator gets the recovery by setting the field.

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
| Balanced arm schedule shared by every paired agentic study | `src/llb/bench/context_policy/interleave.py` |
| Guard-band arithmetic shared by every per-family guard fit | `src/llb/bench/context_policy/guard_band.py` |
| Which guards still measure the middle-critical regime, and why the rest do not | `src/llb/bench/summary_trim/guard_regime.py` |
| The per-family fit, its walk control, and the design gate on the band | `src/llb/bench/summary_trim/guard_fit.py` |
| The step each fold lands on, carried from the episode to the probe | `src/llb/bench/agentic/context.py`, `src/llb/bench/memory/boundary/probe.py` |
| Adoption workloads, their oracles, and the aggregate-search task family | `src/llb/bench/summary_trim/workloads.py`, `src/llb/bench/summary_trim/tasks.py` |
| Balanced arm schedule shared by any multi-arm policy lane | `src/llb/bench/context_policy/interleave.py` |
| Adoption design gate, run, readings, verdict, and persistence | `src/llb/bench/summary_trim/design.py`, `run.py`, `reading.py`, `adoption.py`, `analysis.py`, `report.py` |
| Adoption command | `src/llb/cli/bench/context/summary_trim_adoption.py` |
| Deterministic contracts | `tests/llb/bench/memory/test_agentic_memory_window_elision.py`, `tests/llb/bench/memory/test_agentic_memory_window_elision_transfer.py`, `tests/llb/bench/summary_trim/test_agentic_summary_trim_adoption.py`, `tests/llb/bench/context_policy/test_agentic_arm_interleave.py` |
