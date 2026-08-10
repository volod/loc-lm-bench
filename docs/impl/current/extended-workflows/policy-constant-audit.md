# Policy-Constant Change Audit

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

## What a policy-constant change invalidates

`make bench-agentic-policy-change-audit` generalizes the mechanism above from ONE bound to any agent
context-policy constant. A context policy is a pure function of the deterministic tool world, so
fixing the geometry and the controller fixes the exact sequence of prompts an episode sends before
any model runs. The audit therefore replays every published cell under both values of the changed
field with an oracle controller, records every prompt each replay sends -- controller prompts and
summarize calls alike, by recording through the injected `complete`, which is the seam they all pass
through -- and compares the sequences byte for byte.

Two properties make a replay a statement about a real run. The summarize call is answered with a
FIXED summary so the replay is deterministic, which can only hide downstream divergence, never
invent it: identical summarize prompts mean a temperature-0 model returns the same summary, so the
later controller prompts are identical too, and "all prompts identical under the replay" implies
"all prompts identical under the served model". And BOTH arms of a cell are replayed
(`observation_cap` and `compact`), because a published number is a compact-minus-cap delta and a
change that moves either arm moves it.

A replay records the prompt the guard REFUSED as well as the ones it sent, and compares it the same
way -- byte for byte. Recording through `complete` sees only what reached a model, and the loop's
last act before an overflow is to build a prompt, price it, and end the episode WITHOUT sending it
(`budget.fits` in `run_episode`). Two arms that overflow at the same step therefore send
byte-identical prefixes whatever the refused prompts held, so before this the audit reported
`prompt_invariant` for a change that moved the very prompt which ended the run. A cell whose arms
all sent identical bytes and diverged only on the refusal is named as such in the re-run scope
(`the prompt the guard refused, never sent`), at the model call neither replay made -- one past the
last recorded one.

The refused text needs its own seam, because no other one can reach it. `run_episode` takes an
`on_refused_prompt` observer and hands it exactly the prompt it refused, at both of the loop's
refusal sites (the step prompt and the repair prompt built from it); it never fires on a run that
sends everything it builds, and it cannot change what the loop does. Comparing the refusal by SIZE
would not have been enough, and the reason is one of the audited fields: `observation_head_share`
re-splits a trimmed observation head-and-tail at a fixed cap, which CI measures as moving bytes and
never a prompt length. At depth 6 behind a 3500-char guard, 0.6 and 0.5 both send `[3000]` and are
both refused a 3904-char prompt -- every size the audit could compare is equal, and only the refused
prompt's own bytes separate them. That cell reads `prompts_change` with the scope line
`same size, different bytes`, so nobody reads the equal char counts in the run bundle as an equal
prompt. The priced size is kept beside the text because the two are not redundant: under a
controller channel the guard prices a serialized chat transcript, so the size includes a
serialization the prompt text does not show. The audit replays through `complete`, where they agree
exactly, and CI asserts that agreement.

CI now asserts what the cap-fitting studies previously only assumed: under the BASELINE policy --
the configuration the published numbers were measured under -- no published cell ends on a refused
prompt in either arm. That is why every recorded verdict in the table below is unchanged by the
refusal recording: with no baseline refusal anywhere, each answer is still decided by sent prompts
alone. A CANDIDATE value is a different matter, and one published cell shows it. Widening
`observation_cap_chars` 800 -> 1600 does not merely move `surface-d10-g14000`'s `observation_cap`
arm: that arm peaks at 11926 chars under the pinned 800 and is refused a 14621-char prompt at step 9
under 1600, on all seven tasks. The cell already read `prompts_change` from its sent prompts (the
trim reaches the prompt at model call 2), so no verdict moves -- what is new is that the re-run
scope can say `observation_cap no longer fits this guard under the candidate`, which is the
difference between re-measuring the cell and needing a new guard to measure it at all.

One case needs its own verdict rather than a comparison. A cell that declares the audited field
ITSELF -- the trigger collapse sweeps `compact_share` cell by cell -- is not describable by the
change: replaying it at another value measures a different cell, not the published one. Those report
`cell_pins_the_field` and are excluded from the counts. A value inherited from `held_fixed` is not
the same thing; that is the study's inherited setting, and whether its number holds at another value
is exactly the counterfactual the audit answers.

A CHANGE IS A SET OF FIELDS, not one field. A commit that re-pins `observation_cap_chars` and
`compact_keep_recent` together moved both, so both sides of the comparison are whole policies:
`PolicyChange` carries every moved field, the baseline arm replays the full baseline policy and the
candidate arm the full candidate policy, and the audit answers with ONE verdict and one re-run scope.
Auditing each field on its own would instead compare "baseline cap + candidate keep" against
"candidate cap + candidate keep" -- neither of which is what the published cells were measured
under, and neither of which is what the new build ships -- so its first-divergent step can name a
model call that neither build ever sends. CI proves the difference at the byte level: the compound
candidate arm's prompt digest is the digest of an episode replayed under the whole candidate policy,
and it differs from the digest the single-field audit compares against.

A compound change meets the study-axis rule per field. A cell that declares SOME of the moved fields
keeps its own value for those and is audited on the rest (reported as `not_applicable_fields` on the
row and counted as `n_partially_applicable`); only a cell that declares ALL of them reports
`cell_pins_the_field`. So a `compact_share` + `summary_input_cap` change reads the eight trigger-
collapse cells through the bound half of the change rather than dropping them, which the per-field
audit could not do.

Core locations are `src/llb/bench/agentic_policy_change_replay.py` (`ReplayedEpisode`, the digest
over sent prompts plus the refused one, and the per-arm comparison, which takes two whole settings
maps), the `on_refused_prompt` observer in `src/llb/bench/agentic/episode.py`,
`src/llb/bench/agentic_policy_change_audit.py`
(`PolicyChange`, the auditable fields, the per-study cell geometry, and the verdict),
`src/llb/bench/agentic_policy_change_audit_report.py`,
`src/llb/cli/bench/category_agentic_policy_change_audit.py`, and
`tests/llb/bench/test_agentic_policy_change_audit.py`. The summarize-bound audit
(`src/llb/bench/agentic_memory_cap_audit.py`) is now ONE use of this mechanism rather than a second
one: it supplies the elision diagnostic that explains the verdict, and CI asserts the two agree cell
for cell.

```bash
make bench-agentic-policy-change-audit \
  POLICY_FIELD=observation_cap_chars POLICY_BASELINE=800 POLICY_CANDIDATE=1600
# a compound change: space-separated lists, read field by field, audited as ONE change
make bench-agentic-policy-change-audit \
  POLICY_FIELD="observation_cap_chars compact_keep_recent" \
  POLICY_BASELINE="800 1" POLICY_CANDIDATE="1600 2"
```

Every auditable field against the 22 published cells of the three cap-fitting studies (2026-08-05,
no GPU, about 0.7 s per field; audits land under `$DATA_DIR/agentic-policy-change-audit/<run>/`):

| field(s) | change | invariant | invalidated | not applicable |
| --- | --- | ---: | ---: | ---: |
| `observation_cap_chars` | 800 -> 400 | 0 | **22** | 0 |
| `observation_cap_chars` | 800 -> 1600 | 0 | **22** | 0 |
| `observation_head_share` | 0.6 -> 0.5 | 0 | **22** | 0 |
| `keep_last_n` | 3 -> 1 | **22** | **0** | 0 |
| `compact_share` | 0.5 -> 0.45 | 2 | 12 | 8 |
| `compact_keep_recent` | 1 -> 2 | 0 | **22** | 0 |
| `summary_input_cap` | trigger -> window | **18** | 4 | 0 |
| `observation_cap_chars` + `compact_keep_recent` | 800 -> 1600, 1 -> 2 | 0 | **22** | 0 |
| `compact_share` + `summary_input_cap` | 0.5 -> 0.45, window -> trigger | 10 | 12 | 0 (8 partial) |

Two readings an operator can act on. **`keep_last_n` is free**: the constant sweep EXPOSES keep=1 as
cheaper on prompt tokens, and this says taking that up costs no published compact evidence at all,
because no cap-fitting cell runs the `keep_last_n` policy. **The observation-trim constants are
not**: `observation_cap_chars` and `observation_head_share` change both arms of every cell from
model call 2 -- the first prompt that carries a trimmed observation -- so re-pinning either one
retires all three studies at once.

The `compact_share` row also reproduces the fold-step mechanism from a direction that owes it
nothing. Of the 14 applicable cells, the two that survive 0.5 -> 0.45 are `fold-d6-step6-hi` and
`fold-d6-step7-hi` -- the HIGH guard in each fold step, where a smaller share still lands the trigger
inside the same step's interval and folds the identical transcript. At the low guards the trigger
drops into the previous step and everything downstream changes. A byte-level prompt comparison that
knows nothing about fold steps rediscovers exactly where they are.

The two compound rows are the mechanism's own check on itself. `compact_share` + `summary_input_cap`
is the interesting one: read field by field it reports 12 invalidated plus 8 cells the share half
cannot describe, and read as one change it reports the same 12 -- but now those 8 collapse cells are
ANSWERED (they keep their own share and are audited on the bound), and the whole verdict is computed
between the two policies that actually existed. On the evidence committed today the compound scope
and the per-field union name the same cells at the same first-divergent steps, so what the change
buys HERE is a guarantee rather than a correction. The case where the two answers separate needs a
geometry in which two constants interact, which is the next section.

## The compound guarantee has a geometry that tests it

On the 22 published cells the compound reading and the per-field union agree, so nothing there
proves the compound replay is doing anything the collapsed implementation could not: delete
`PolicyChange`, audit one field at a time, and every test above stays green. The separation needs
an INTERACTION -- two constants where one decides what the other MEANS -- and `compact_share` x
`summary_input_cap` is exactly that pair. Under the `trigger` bound the summarize call's input cap
IS `compact_share * max_prompt_chars` (`summary_input_cap_chars` in
`src/llb/bench/agentic/episode.py`); under the `window` bound it does not depend on the share at
all. So the share moves the bound's own value, and a guard whose folded transcript sits BETWEEN the
two shares' triggers reads three different ways:

- move the share alone (bound stays `window`): both shares fold at the same step and the window
  bound elides nothing, so every prompt is identical -- invariant;
- move the bound alone (share stays at the baseline): the baseline trigger still clears the offered
  transcript, so nothing is elided -- invariant;
- move BOTH, which is what the commit did: the candidate trigger falls below the offered transcript,
  the summarizer is shown a head-and-tail elision of it, and every prompt from the fold on differs.

`samples/benchmarks/agentic_policy_change_interaction_design.json` commits that geometry for the
change `compact_share 0.5 -> 0.48` with `summary_input_cap window -> trigger`, at depth 10 over the
memory-dependent tasks. It publishes NO number and is deliberately absent from the audited study
registry, so no future constant change is ever asked to re-run it. Each cell PREDECLARES the three
numbers the separation rests on -- the fold step (which must be the same at both shares), the
transcript the fold offers the summarizer, and each share's trigger -- and CI checks them with an
oracle controller and no model, so a drift names the claim that moved rather than only the digest:

| cell | guard | fold step | offered | trigger at 0.5 / 0.48 | compound | per-field union |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `interaction-d10-g21500` | 21500 | 10 | 10494 | 10750 / 10320 | **invalidated** @10 | invariant |
| `interaction-d10-g23700` | 23700 | 11 | 11802 | 11850 / 11376 | **invalidated** @11 | invariant |
| `control-d10-g21900` | 21900 | 10 | 10494 | 10950 / 10512 | invariant | invariant |
| `control-d10-g23000` | 23000 | 11 | 11802 | 11500 / 11040 | invalidated @11 | invalidated @11 |

The two `interaction` cells are the counterexample: `trigger at 0.48 < offered <= trigger at 0.5`,
so the compound reading reports the cell invalidated at the fold step and the per-field union
reports that NEITHER field changes anything. The compound one is the true statement -- CI also
replays the whole baseline policy against the whole candidate policy there and the prompt digests
differ -- so a per-field audit would have cleared a commit that really did move the transcript. The
two `control` cells sit just outside the band on either side and are where the two readings agree,
one invariant and one invalidated, so the fixture tests the interaction rather than the change.

The band is narrow by construction -- the two triggers must land inside ONE fold step's interval
while straddling the offered transcript -- so an unrelated edit to a prompt template or the memory
task world shifts every prompt size and walks the committed guards out of it. That band is not
searched for, it is SOLVED: three conditions, each a half-open guard interval, intersected per fold
step. The three are the same three for ANY pair of moved fields -- each field audited alone must
change no prompt, and the two together must change one -- and what differs per pair is the
arithmetic they turn into. For this pair all three are intervals the fold-step ladder and the
boundary probe already give between them -- the guard intervals are ladder arithmetic over one
measured prompt sequence, the offered transcript is what the probe's episode walk reports:

| condition | why | interval |
| --- | --- | --- |
| both shares fold at the same step | else the share alone reads as changed | `fold_step_guard_interval` per share, intersected |
| the baseline share elides nothing | else the bound alone reads as changed | `guard >= smallest_guard_reaching(offered, 0.5)` |
| the candidate share elides | else the compound reading has nothing to report | `guard < smallest_guard_reaching(offered, 0.48)` |

At depth 10 that solves to guards `[21084, 21863)` at fold step 10 and `[23604, 23852)` at fold
step 11 -- the two committed cells sit inside them, the two controls outside. The bands are exact
rather than indicative: CI replays the audit at `low - 1`, `low`, `high - 1`, and `high` and asserts
the separation starts and stops there. An empty answer is an answer too: at depth 8 the fold offers
the summarizer too little to overtake any trigger, and the solver reports no band rather than an
unusable interval. So a drifted task world fails with the guards to commit instead of with a
geometry mismatch:

```text
[band] depth 10, compact_share 0.5 -> 0.48, summary_input_cap 'window' -> 'trigger'
  fold step 10: guards [21084, 21863) separate (offered 10494, folds at triggers [10120, 11008))
  fold step 11: guards [23604, 23852) separate (offered 11802, folds at triggers [11008, 11926))
```

The elision inequality is about ONE offered transcript. The probe therefore keeps a per-fold
breakdown (`summary_fold_input_chars` on `ContextTelemetry`, surfaced by `compact_fold_input_probe`)
beside the summed `summary_input_chars`, and the solver states the inequality against the first fold
whose candidate share would flip elision relative to the baseline -- the compaction at the fold step
being solved. A deep episode under a small guard still folds more than once; those later folds are
checked against the same both-shares interval and, on the shipped geometry, never open a band of
their own, which the empty-band detail records (`N later fold(s) never separate`). The committed
single-fold bands above are unchanged by that generalization: "no separating band at this depth"
now means no band exists rather than none the solver could read past the first fold.

Only the VERDICT direction separates for this pair. A scan of depths 6 / 10 / 14 across guards 6000
to 34000 found no geometry where both readings report an invalidated cell but name different
first-divergent steps, so the committed cells are the whole separating set the pair offers. The
solver was checked against that scan at depths 8 / 12 / 14 -- every guard it excludes fails to
separate and every guard it includes separates.

Core locations are `src/llb/bench/agentic_policy_change_interaction_fixture.py` (the fixture
contract, and the no-model probe of the predeclared geometry),
`src/llb/bench/agentic_policy_change_interaction.py` (the two readings, and the separation verdict),
`src/llb/bench/agentic_policy_change_interaction_band.py` (the band solver and its report),
`src/llb/bench/agentic_policy_change_interaction_terms.py` (the interval vocabulary a condition is
stated in), `..._conditions.py` (what each pair demands of a guard, including the per-fold elision
read) and `..._cap.py` (the
observation cap's own case, which is the one that has to tell a prompt the episode SENDS from a
prompt the loop merely builds), and the four test modules that ARE the CI assertion --
`tests/llb/bench/test_agentic_policy_change_interaction.py` for the separation, `..._band.py` for
the band (including that a multi-fold step answers rather than refusing), `..._couplings.py` for the
enumeration below, and `..._cap.py` for the discarded-prompt
arithmetic. All run inside `make ci`, together in about two seconds, with no target of their own.

```bash
make ci                       # the separation assertion; a collapsed audit fails here
.venv/bin/python -m pytest tests/llb/bench/test_agentic_policy_change_interaction.py \
  tests/llb/bench/test_agentic_policy_change_interaction_band.py \
  tests/llb/bench/test_agentic_policy_change_interaction_couplings.py \
  tests/llb/bench/test_agentic_policy_change_interaction_cap.py
```

## One pair separates, and the other fourteen are answered

A guarantee resting on ONE pair through ONE coupling retires with the next refactor: the `trigger`
bound is already not the shipped default, and retiring it takes the only counterexample with it. So
every pair of the six `AUDITABLE_FIELDS` -- `C(6, 2) = 15` of them -- states its mechanism and its
own three conditions, and the solver answers each. Two of the three contradict each other outright
for every pair except one, which is why the band arithmetic can answer them at all depths rather
than only where it was asked:

| pair | how one decides what the other means | answer |
| --- | --- | --- |
| `compact_share` x `summary_input_cap` | the bound's own value IS `compact_share * guard` under `trigger` | **band**, solved per fold step |
| `observation_cap_chars` x `compact_share` | the cap decides every prompt size, and the size is what crosses the trigger | no geometry |
| `observation_cap_chars` x `summary_input_cap` | the cap decides which step folds, and the fold decides what the bound must fit | no geometry |
| `observation_cap_chars` x `compact_keep_recent` | the cap decides which step folds, the keep what it leaves live | no geometry |
| `observation_cap_chars` x `observation_head_share` | the cap decides whether the head share means anything at all | no geometry |
| `compact_share` x `compact_keep_recent` | the share picks the fold step, the keep how much survives behind it | no geometry |
| `compact_keep_recent` x `summary_input_cap` | the keep sizes the offered transcript, the bound decides whether it is elided | no geometry |
| `observation_head_share` x share / keep_recent / bound | nothing: the head share moves no prompt LENGTH | independent |
| `keep_last_n` x cap / head / share / keep_recent / bound | nothing: `keep_last_n` parameterizes neither audited arm | independent |

Each `no geometry` answer is one contradiction, stated as a condition rather than as an empty list:

| the field that blocks | the two conditions that cannot both hold |
| --- | --- |
| `compact_keep_recent` | silent alone needs at most `min(keeps)` live entries at the fold (`compact_state` folds the WHOLE transcript when the keep would leave nothing to fold); contributing to the compound needs more than that |
| `observation_cap_chars` | silent alone needs every prompt it moves to be one no arm shows; contributing needs it to move a prompt size at all. The two cross even in the corner where a fold discards the moved prompt -- see [the cap's own case](#the-caps-silence-is-about-the-prompts-the-loop-builds) |
| `observation_head_share` | `trim_observation` keeps `head + tail = cap_chars`, so the head share moves no length, no fold step, no trigger crossing and no overflow; the bytes it does move are shown at the same steps under either partner value |
| `keep_last_n` | neither `observation_cap` nor `compact` reads it, so no value of it moves a prompt in either replayed arm |

The solver reports a blocked step the way it reports a solved one, which is what makes the negative
answer readable rather than merely empty. It asks only about the steps an episode can actually fold
at rather than every step a trigger can SELECT: step 1 is reachable on the shipped geometry -- a
small enough guard trips on the first prompt -- but its prompt is built from zero entries, so
`compact_state` finds nothing older to summarize and returns unchanged. Conditions stated there are
about a fold that cannot happen, and since the report carries each condition once, at its first
blocked step, that vacuous row would LEAD the answer with its least informative line. Dropping it
moves no solved band: the two committed bands are unchanged, and the step it drops states a
condition no guard satisfies anyway. The ladder itself is `foldable_fold_steps` in
`agentic_memory_fold_step_ladder`, beside the `reachable_fold_steps` it narrows, and the fold-step
placement rules are placed against the same one
([the crossover is a fold step](crossover-geometry.md#the-crossover-is-a-fold-step-not-a-char-guard)).

```text
[band] depth 10, compact_share 0.5 -> 0.48, compact_keep_recent 1 -> 2
  no fold step separates the two readings at this depth
  compact_share x compact_keep_recent [no_geometry]: the share decides which step folds, and the
    keep decides how much of the transcript that fold leaves live behind it
  fold step 2: the_moved_keep_folds_a_different_span impossible: the compound must fold something
    the partner field alone does not, which needs more than 1 live entries; step 2 has 1
  fold step 3: the_keep_audited_alone_is_silent impossible: the fold must hand the summarizer the
    same span under both keeps, which needs at most 1 live entries; step 3 has 2
```

An inequality can be wrong about the loop it describes, so the same question is asked by REPLAY too:
`agentic_policy_change_interaction_scan` walks a grid of geometries, reads every cell both ways, and
reports where they actually disagree. Arithmetic and replay agree -- inside a solved band and
nowhere else. The scan also sweeps the moved VALUES (`FIELD_CANDIDATE_GRID` per field, with the
`FIELD_MOVES` neighbour as the first entry): every pair is asked at every candidate combo, not only
at one neighbour, and the result records which pairs stay silent across that grid. The share
direction is why that matters -- the band that opens for `compact_share` 0.5 -> 0.48 vanishes for
0.5 -> 0.55, because only a lower share can elide where the baseline did not -- so a silent answer
backed only at 0.48 would be a property of one chosen value. Evidence (2026-08-06, no GPU, ~3 min):
9630 cells at depth 10 over guards 2000 to 34000 in steps of 100, every pair scanned twice (the
shipped value against a plausible neighbour, then against a second alternative so an answer cannot
be a property of one chosen value). Exactly 10 cells separate, all of them `compact_share` x
`summary_input_cap` at guards 21100-21800 and 23700-23800 -- the two solved bands. The `slow`-marked
test replays the compact form of that grid (depths 6 / 10 / 14, guards 2000 to 34000 in steps of
1000 plus the two committed guards) across the candidate-value grid and asserts both halves: the
other fourteen pairs stay silent at every candidate combo, and every hit for the separating pair
falls inside a solved band for the change that produced it (share 0.48 only; 0.55 contributes none).
`make ci` runs the same value sweep at the fixture depth and the two committed guards.

## The cap's silence is about the prompts the loop builds

A replay calls a field silent when no prompt the episode SENDS moves, and the loop builds one prompt
it never sends: `step_prompt` assembles the step prompt, finds it over the compaction trigger, folds,
and rebuilds. Everything downstream of that fold is cap-independent -- the summarize call is
assembled from RAW observations (`summarize_entries`), and so are the aggregate and memory facts
folded into the summary -- so a fold that drops every entry the two caps disagree about leaves the
post-fold prompt identical under both. `observation_cap_chars` can therefore move a size the compact
arm never shows, which is a case its `no geometry` answer has to ANSWER rather than assume away.

That corner is not hypothetical. At depth 10 the shipped geometry has 37 guards (6000-7800) whose
fold drops the whole transcript -- step 2, one live entry, so `compact_state` takes its
fold-everything fallback -- and at every one of them a cap move of 800 -> 1600 audited alone reports
`changed_arms = ['observation_cap']`: the compact arm really is blind to the cap there, and the only
thing that reports it is the OTHER audited arm.

So the cap states three cases instead of two (`agentic_policy_change_interaction_cap`):

| what the two caps move | silent audited alone | reaching the compound |
| --- | --- | --- |
| no prompt at any step, sent or discarded | always | impossible -- nothing to move a fold step with |
| a prompt some arm shows | impossible -- the union reports the cap alone | (already blocked) |
| only prompts the fold discards | `guard < smallest`, because the `observation_cap` arm has no fold to hide it behind and either sends the prompt or overflows on it | needs a trigger inside `[smallest, largest)`, so `guard >= smallest_guard_reaching(smallest, share)` |

The third row is the corner, and it closes: a compact share is at most 1, so a guard whose trigger
REACHES a prompt is never smaller than that prompt, and the two bounds cross at every fold step. The
answer is now a proof over every prompt the loop builds rather than over the ones it sends. The
`or overflows on it` half of that row is no longer only an argument, either: the replay records the
prompt the guard refused and compares it byte for byte
([what a policy-constant change invalidates](#what-a-policy-constant-change-invalidates)), so an
`observation_cap` arm that ends on a differently trimmed overflow is read rather than reasoned about.
Which
entries a cap moved is read off the FIRST DIFFERENCE of the two prompt sequences the solver already
computes -- a step prompt is a fixed scaffold plus one line per live entry -- so the third case
costs no extra replay, and it catches a pair of entries that move in opposite directions where a
comparison of prompt sizes would not. The shipped task world pads every observation alike, so a cap
that re-trims the first one re-trims all ten: the corner's branch is exercised on stated sequences,
and the fold-everything count it turns on is checked against `compact_state` itself.

The blocked step reads as a derivation rather than as an empty list, and it names WHICH moved entry
outlives the fold:

```text
fold step 2: nothing separates (caps 800 -> 1600 move step 2's prompt (3904 vs 4333 chars), and
  the fold here discards entries 1-1) -- the_cap_audited_alone_is_silent impossible: a differently
  trimmed observation the fold leaves live is shown in a prompt the episode sends; the fold here
  discards entries 1-1 and entry 2 (9 of 10 moved entries) survives it
```

Core locations are `src/llb/bench/agentic_policy_change_interaction_couplings.py` (the enumeration,
its mechanisms, the one concrete move per field, and the per-field candidate grid the value sweep
asks with),
`src/llb/bench/agentic_policy_change_interaction_conditions.py` (the per-pair conditions) and
`..._cap.py` (the cap's three cases and the per-entry arithmetic),
`src/llb/bench/agentic_policy_change_interaction_scan.py` (the replay scan, the candidate-value
sweep, and its refusal to scan a baseline the per-field arm would not replay), and
`tests/llb/bench/test_agentic_policy_change_interaction_couplings.py`, which is the assertion: every
pair enumerated, one pair separating, the two independence claims (`keep_last_n` inert, the head
share length-preserving) measured on real prompts rather than asserted, and the fourteen other pairs
silent across the candidate-value grid.
`tests/llb/bench/test_agentic_policy_change_interaction_cap.py` holds the cap's three cases: the
shipped geometry's blocked steps stay blocked for the stated reason, the corner's branch is read on
stated sequences, and the fold-everything count is measured against `compact_state`.

```bash
make ci                       # the enumeration; a new policy constant fails here unpaired
.venv/bin/python -m pytest tests/llb/bench/test_agentic_policy_change_interaction_couplings.py \
  -m slow                     # the wide replay scan, ~30 s, excluded from `make ci`
```

## The audit runs in CI, on the act that creates the problem

The audit above answers the question only when someone asks it, and the person editing a constant in
`src/llb/bench/agentic/context_policy.py` is precisely the person who does not know the question
exists.
`samples/benchmarks/agentic_context_policy_pins.json` closes that loop: it PINS each shipped
`ContextPolicy` constant to the value the published evidence stands on, and CI compares the pins with
the live dataclass defaults on every run. A drifted field is audited on the spot and the failure
message is the re-run scope -- every invalidated cell by id, depth, guard, changed arms, and the model
call where the change first bites -- plus the doc sections that publish those numbers. The scope also
walks every registered published value whose operation declares the moved field. A
`compact_share` drift therefore names both portable trigger-ratio bands (depths 6 and 10) through
their `trigger_over_own_cap_peak` operation, even though the trigger-collapse cells themselves pin
share as their study axis and are excluded from the cell replay.

Constants that drift TOGETHER are audited together, as the one change the commit made: the baseline
arm replays the full pinned policy, the candidate arm the full shipped policy, and the failure
message carries one scope under a `2 constants moved together and are audited as ONE change` heading
that lists every move (`- observation_cap_chars: pinned 800 -> shipped 1600`) plus each constant's
own `pinned because` note. Auditing each drifted constant separately would have compared "pinned cap
+ shipped keep" against "shipped cap + shipped keep": two configurations no published cell was
measured under, reported as two re-run scopes for one act. The gate also feeds the full pinned map
into the replay for fields the change does NOT move: without that, a `restated` pin on a held field
(`observation_cap_chars` or `observation_head_share`) would leave the design's stale `held_fixed`
value on the baseline arm -- the same class of bug the compound audit closed, one level down. A
hand-run CLI audit that has no pins keeps the design / dataclass-default fallback; CI always has the
pins, so the baseline arm is the policy the published numbers were measured under for every field.

The gate fails on ANY drift, including a drift the audit clears. The pin is the record of what the
evidence was measured under, so a change that invalidates nothing costs one fixture line to restate
only when no registered arithmetic depends on it; the message says `restating the pin is free` in
that case. A cell-free change with affected arithmetic instead lists each published statement, its
operation, and the moved dependency, then requires those values to be re-derived and restated. What
is refused is the silent case: a constant moving while the docs keep quoting numbers measured under
its old value. A clean build replays nothing, and a drifted one costs under a second per field.

Each pin also declares how it relates to the committed designs -- `agree` (every design's
`held_fixed` states the pinned value), `restated` (a design states another value and the pin
supersedes it, which is where `summary_input_cap` sits: the designs record the retired `trigger`
bound and the crossover restatement moved the published numbers to `window`), or `unstated` (no
design states the field, as for `keep_last_n` and `compact_keep_recent`) -- and CI verifies that claim
against the designs themselves, so a pin cannot quietly disagree with the studies it names. CI also
asserts that the pinned set is exactly `ContextPolicy`'s constants, so a NEW shipped constant is
pinned here or the build is red, and that every doc anchor the fixture names still resolves.

Core locations are `src/llb/bench/agentic_policy_pin_gate.py` (the fixture reader and the drift
check, which passes the full pinned policy into the replay for untouched fields),
`src/llb/bench/agentic_policy_pin_gate_report.py` (the failure message, which renders its
re-run scope through the audit's own reporter),
`src/llb/bench/agentic_published_value_operation_scope.py` (the registered-value half of that
scope), `src/llb/bench/agentic_published_value_operation_policy.py` (the perturbation check that
makes each operation declaration trustworthy), the shared study registry `AUDITED_DESIGN_PATHS` in
`src/llb/bench/agentic_policy_change_audit.py` (one registry, so the CLI audit and the gate can never
walk different evidence), `src/llb/bench/agentic_policy_change_replay.py` (`_policy`, which prefers
pins over design `held_fixed` for fields the change does not move), and
`tests/llb/bench/test_agentic_policy_pin_gate.py`, which is the gate itself -- it runs inside
`make ci`, with no target of its own.

```bash
make ci                       # the gate; a drifted constant fails here with the re-run scope
.venv/bin/python -m pytest tests/llb/bench/test_agentic_policy_pin_gate.py   # just the gate
```
