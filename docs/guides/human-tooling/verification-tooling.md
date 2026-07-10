# Data-verification tooling -- how to use it

This is the operator manual for the three `verify-*` commands that turn the human verification gate human
sample-acceptance gate into a reproducible workflow. The *why* (the gate, the human-ground-truth
guarantee, the acceptance-sampling framing, the papers) lives in the human-in-the-loop manual's
[Eval-data verification](human-in-the-loop-evaluation.md#eval-data-verification----human-sample-acceptance-of-ai-drafted-data)
section; this page is the *how*. It is the verification-side twin of the
[calibration-tooling manual](calibration-tooling.md).

Verification answers one question: **can the AI-drafted, frontier-cross-checked data be trusted to
score models?** The pipeline already drafts items and a *second* frontier model cross-checks each
one for grounding + support. human verification gate is the irreducibly-human stage: you verify
a **stratified sample** and accept it, and only then may those items flip to `verified=true`
and score real models. A clean sample accepts the bundle; a dirty one sends the drafts back.
Nothing scores a real model until its data clears this gate.

## At a glance

```text
draft bundle (verified=false)
  -> 0. validate structure        make validate-goldset    [gate: spans/ids/splits ok]
  -> 1. draw stratified sample    make verify-sample       [offline, deterministic]
  -> 2. review each item [HUMAN]  make verify-review       [four checks per item]
  -> 3. accept + emit ledger      make verify-accept       [gate: reject rate <= tolerance]
  -> 4. flip via the ledger       ingest_squad --verified-goldset .../accepted/goldset.jsonl
```

Step-by-step for a real-corpus bundle (details in
[Case 1](#case-1-a-draft-from-a-real-corpus-the-common-case)):

```bash
make validate-goldset GOLDSET=<bundle>/goldset.jsonl CORPUS=<bundle>/corpus
make verify-sample  BUNDLE=<bundle> VERIFY_N=30
make verify-review  VERIFY_WS=<bundle>/verify_sample.csv     # the human step
make verify-accept  BUNDLE=<bundle> VERIFY_WS=<bundle>/verify_sample.csv VERIFY_TOLERANCE=0.05
```

Only step 2 is human-paced; everything else is a command. The gates: the structural validate
must pass before you read anything, and `verify-accept` reports the per-stratum + overall reject
rate against tolerance -- only individually-accepted items enter the accepted ledger, and only
ledger items may become `verified=true`.

## The commands

| Command | What it does | Needs |
| --- | --- | --- |
| `make verify-sample` | Draws a **stratified sample** from a draft bundle and writes a verification worksheet + a `sample_manifest.json` (size + strata). `VERIFY_ANNOTATORS=<k>` writes the same sample as `k` per-reviewer worksheets. | nothing (offline; reads the bundle) |
| `make verify-review` | **Interactive verifier**: walk the sample item by item, run the four checks against the corpus, accept/reject. | nothing (offline; CSV only) |
| `make verify-adjudicate` | Multi-annotator only: writes the **agreement report** (Cohen's/Fleiss' kappa) and draws disagreements into `adjudication.csv`. | nothing (offline; CSV only) |
| `make verify-accept` | Computes the reject rate vs tolerance under the chosen policy (`VERIFY_ACCEPT_POLICY=global\|per-stratum\|weighted`) and emits the **accepted-ledger** bundle. | nothing (offline; CSV only) |

None of these needs a GPU or an endpoint -- the drafting and the second-frontier cross-check
already ran. Sampling, reviewing, and accepting are pure file operations: you can verify on a
laptop, in several sittings.

## The draft bundle

A draft bundle is a self-contained directory the pipeline writes under
`$DATA_DIR/prepare-goldset/<ts>/` (from `prepare-goldset-draft`) or a `prepare-synthetic-corpus`
output. The tool reads:

- the gold file -- `goldset.jsonl` (real-corpus drafts) **or** `planted_labels.jsonl` (the
  synthetic planter); either is accepted;
- `corpus/` -- the grounding docs, so the worksheet's spans resolve offline;
- `provenance.json` (optional) -- carries the bundle-level `synthetic` flag (the planter sets it);
  a synthetic bundle gets the extra **planted-labels** check, a real one does not;
- `*.cross_check.json` (optional) -- the second-frontier verdict per item, surfaced as read-only
  `cc_*` context (hidden by default; see below).

Run the structural gate first -- it confirms every span resolves to its labeled text, ids are
unique, and splits are disjoint, before you read anything:

```
make validate-goldset GOLDSET=<bundle>/goldset.jsonl CORPUS=<bundle>/corpus
```

## The worksheet

The verification worksheet is a single CSV, by default `<bundle>/verify_sample.csv` (override with
`VERIFY_WS=`). It lives **inside the bundle** -- which is already under `$DATA_DIR` (gitignored) --
so there is no permanent/temporary routing to think about, unlike the calibration worksheet.

The worksheet **is** the session state -- every interactive edit rewrites only the human columns,
merged into the file atomically, so resume and crash-safety are free. Columns:

| Column | Filled by | Meaning |
| --- | --- | --- |
| `item_id`, `provenance`, `split`, `source_doc_id` | `verify-sample` | the gold item this row verifies |
| `synthetic` | `verify-sample` | bundle-level flag (from `provenance.json`); gates the planted check |
| `stratum` | `verify-sample` | the sampling cell (`provenance \| split \| source_doc_id`) |
| `question`, `reference_answer` | `verify-sample` | the drafted item you verify |
| `span_doc_id`, `span_text` | `verify-sample` | the cited span the answer is grounded in |
| `context` | `verify-sample` | the span shown inside its surrounding corpus window (`>>>span<<<`) -- read this to confirm grounding without leaving the tool |
| `retrieval_rank` | `verify-sample` | the item's needle retrieval rank against the full-corpus index (from `needle_items.jsonl` / `item_provenance.jsonl`); blank = retrieval miss or not annotated |
| `page_citation` | `verify-sample` | `<source.pdf> p.N[-M]` for the cited span (from the PDF `*.citations.json` sidecar); blank for non-PDF docs |
| `cc_grounded`, `cc_non_circular`, `cc_supported`, `cc_answerable`, `cc_note` | `verify-sample` | the second-frontier verdict -- **hidden in the reviewer by default** |
| `chk_grounded`, `chk_answerable`, `chk_reference`, `chk_planted` | **you** | the four per-item checks: `pass` / `fail` / "" (planted is "" / N/A for real items) |
| `decision` | **you** (`y` / `x`) | `accept` / `reject` / "" |
| `reject_code` | **you** (`x` / `x <code>`) | coded rejection reason (`ungrounded`, `circular`, `wrong_reference`, `label_mismatch`, `bad_question`, `other`); bare `x` infers it from the first failed check |
| `edited_answer` | **you** (`e` command) | accept-with-edit reference answer; stored only after it re-grounds to a verbatim corpus span |
| `human_note` | **you** (`note` command) | optional free-text note (record the reason for any `fail`) |
| `human_status` | the reviewer | `pending` / `decided` (resume keys on an empty `decision`) |
| `reviewer_id` | `verify-sample` | multi-annotator worksheets only: which reviewer this sheet belongs to (`r1`..`rk`, or `adjudicator`); blank on single-reviewer worksheets |

The four checks (the same ones the second frontier ran -- you are confirming it did not
*systematically* err):

1. **`chk_grounded`** -- the cited offsets really support the answer/label (read the `context`).
2. **`chk_answerable`** -- the item is answerable and non-circular (the question does not leak its
   own answer).
3. **`chk_reference`** -- the reference answer is correct.
4. **`chk_planted`** -- *synthetic only*: the planted labels match what the doc actually says.

---

## Case 1: a draft from a real corpus (the common case)

A bundle from the ontology-assisted drafting pipeline (`prepare-goldset-draft` over a corpus)
lands `verified=false` with its gold file as `goldset.jsonl`. This is the normal path.

### 1. Draw the stratified sample

```
make verify-sample BUNDLE=$DATA_DIR/prepare-goldset/<ts> VERIFY_N=30
```

This allocates `VERIFY_N` across the strata (provenance x split x source-doc) -- proportional to
each stratum's size with a floor of one per cell, so a stratum can never go unsampled -- and writes
`<bundle>/verify_sample.csv` plus a `sample_manifest.json` documenting the bundle, seed, requested
vs actual size, population, and the per-stratum counts. **Stratifying matters**: an error
concentrated in one source doc cannot hide behind a clean overall rate when every cell is
represented. `VERIFY_SEED=` makes the draw deterministic (reproducible samples).

> Size the sample for the error rate you will tolerate -- a few dozen across strata is typical for
> an acceptance check. If `VERIFY_N` >= the bundle size, every item is sampled.

### 2. Verify each item with the interactive reviewer

```
make verify-review VERIFY_WS=$DATA_DIR/prepare-goldset/<ts>/verify_sample.csv
```

This opens the reviewer and resumes at the first undecided item. The card layout mirrors the
external-RAG review session (`make score-external-rag`): a `=====` banner, `== field:` labels, a
blank line before `== question:` so consecutive cards are visually delimited, and the shared
`o`/`w` edit keys. Each card shows the question, the
reference, and the cited span **inside its corpus window**, so you confirm grounding in place. The
second-frontier `cc_*` verdict is **hidden by default**: if you see it first you will unconsciously
defer to it, and the gate would then measure whether you echoed the cross-check instead of whether
an *independent* human agrees -- which is the whole point. So verify before you peek.

For each item: read the `context`, run the four checks (mark each `pass` or `fail`), then `accept`
or `reject` -- the decision advances to the next item.

#### Reviewer commands

| Key | Action |
| --- | --- |
| `g` / `a` / `r` / `p` | mark grounded / answerable / reference / planted **pass** |
| `G` / `A` / `R` / `P` | mark the same check **fail** (uppercase) |
| `y` | accept this item and advance |
| `x` | reject and advance (the reject code is inferred from the first failed check) |
| `x <code>` | reject with an explicit code: `ungrounded`, `circular`, `wrong_reference`, `label_mismatch`, `bad_question`, `other` |
| `e` / `w` | accept-with-edit: type a corrected reference answer; it is **re-grounded immediately** against the bundle corpus and refused unless it is a verbatim span |
| `o` / `note` | edit `human_note` (prompts for one line; empty clears it) |
| `n` / Enter | next item (no change) |
| `b` / arrows | previous item (go back to change a mark) |
| `j <N>` | jump to item N (1-based) |
| `u` | jump to the next undecided item |
| `c` | clear this item's marks + decision |
| `?` / `h` | help + the check legend |
| `q` | save and quit |

(The planted check is refused on a real, non-synthetic item -- it is N/A there. An accept over an
edited answer that no longer matches a verbatim span is blocked until you re-ground it with `e`.)

Each decision prints a pace line (items decided this session, items/hour, ETA for the remainder),
and every sitting appends its measured throughput to `verify_session_stats.json` beside the
worksheet.

Useful options:

```
make verify-review VERIFY_WS=… VERIFY_ORDER=confidence  # least-confident items first
make verify-review VERIFY_WS=… START=20           # begin at item 20
make verify-review VERIFY_WS=… SHOW_CROSSCHECK=1   # reveal cc_* (POST-HOC review only -- it anchors)
make verify-review VERIFY_WS=… CLEAR=1             # wipe all human columns and start fresh (gated)
```

Ctrl-C, EOF, and `q` all save and quit; the last edit is already on disk (write-through), so you
never lose work. Re-running `make verify-review` continues where you left off.

Need a bigger sample after starting? Enlarge it **additively**:

```
make verify-sample BUNDLE=… VERIFY_N=60 VERIFY_MERGE=1
```

This appends only item ids the worksheet does not already hold -- decided rows are preserved
byte-for-byte and never re-shown, and re-running the merge is a no-op.

### 3. Accept (or send it back) and flip via the ledger

```
make verify-accept BUNDLE=$DATA_DIR/prepare-goldset/<ts> \
    VERIFY_WS=$DATA_DIR/prepare-goldset/<ts>/verify_sample.csv
```

This prints the acceptance report -- the per-stratum and overall **reject rate** against
`VERIFY_TOLERANCE` (default `0.05`) -- and writes the **accepted-ledger** bundle at
`<bundle>/accepted/`: the items you accepted, with `verified=true`, plus their copied corpus docs,
so the ledger is self-contained. It also flags any item that has a failed check but no decision
(`undecided_with_failures`) so nothing slips through unreviewed. Accept-with-edit answers are
re-grounded against the bundle corpus here as well (an un-groundable edit aborts instead of
certifying), and a `rejection_reasons.json` summary of your coded rejections lands beside the
ledger for draft-pipeline feedback.

The policy is **report + accept the clean items**: a stratum or overall rate above tolerance prints
a `FAIL` warning, but the individually-accepted items are still emitted -- *you* decide whether the
defect rate means the bundle goes back to the pipeline. Re-draft and re-verify if it does.

Then flip the accepted items into your scored gold set **through the ledger** -- never by
hand-editing the boolean. `verify-accept` prints the exact command:

```
python -m llb.prep.ingest_squad ... --verified-goldset <bundle>/accepted/goldset.jsonl
```

The ingester re-adopts those ids by **replacement** (canonical content + grounded spans), which is
what stops a reused id from certifying changed content. See
[goldset-from-scratch](../data-prep/goldset-from-scratch.md) for the ledger mechanics.

### Done when
A documented stratified sample passes the four checks and the accepted items are flipped to
`verified=true` through the ledger. Only then may they enter a scored run.

---

## Case 2: a synthetic planted bundle

A bundle from `prepare-synthetic-corpus` names its gold file `planted_labels.jsonl` and records
`"synthetic": true` in `provenance.json`. The tool detects both automatically -- the workflow is
identical to Case 1, with one addition: every card now carries the **planted** check, because the
labels were *authored*, not drawn from a real corpus. Verify that each planted label actually
matches what the synthetic doc says (`p` = pass, `P` = fail), then accept/reject as usual.

```
make verify-sample  BUNDLE=$DATA_DIR/prepare-goldset/<synthetic-ts> VERIFY_N=30
make verify-review  VERIFY_WS=$DATA_DIR/prepare-goldset/<synthetic-ts>/verify_sample.csv
make verify-accept  BUNDLE=$DATA_DIR/prepare-goldset/<synthetic-ts> \
    VERIFY_WS=$DATA_DIR/prepare-goldset/<synthetic-ts>/verify_sample.csv
```

Because the synthetic flag is a bundle-level fact, a bundle is uniformly real or synthetic -- the
planted check applies to all of its items or none.

---

## Case 3: per-category, pull-based verification

human verification gate is **not** a one-shot upfront task -- it runs in parallel with the build
and is pull-based: as each category's bundle is drafted + cross-checked and **stabilizes**,
verify *that* bundle, flip its accepted items, and real-model scoring unblocks for it.
The objective category boards never wait on this -- only real-model HEADLINE scoring does.
Practical notes:

- Verify a bundle only once its drafting is stable; do not re-verify seeds a newer bundle
  supersedes.
- Each bundle gets its own `verify_sample.csv` (it lives inside the bundle dir), so several
  categories' verifications can be in flight independently.
- Keep the `sample_manifest.json` -- it is your record of what you sampled and the strata you
  covered (the "document the sample" discipline from *Datasheets for Datasets*).
- When a verified category bundle is ready for the category suite composite headline,
  rerun that category with `--data-verified --verification-ref <bundle>/sample_manifest.json`,
  then follow the [composite-headline close-out](../benchmarking/composite-headline.md).

### Verification references in category runs

The `bench-* --data-verified` commands do not trust a boolean alone. They validate the
`--verification-ref` before model calls and before a verified manifest can be persisted. Accepted
forms are:

- a reviewed `verify_sample.csv` whose rows are all decided and whose reject rate is within
  tolerance;
- a `sample_manifest.json` whose `worksheet` points to such a reviewed worksheet;
- an accepted-ledger directory or `accepted/goldset.jsonl` whose items are all `verified=true`.

If the artifact is missing or invalid, the command fails with the path, kind, reason, detailed
statistics, and the commands needed to repair the verification data. Worksheet diagnostics include
sample size, decided/accepted/rejected/undecided counts, undecided failed checks, reject rate,
tolerance, and failing strata. Accepted-ledger diagnostics include total, verified, unverified, and
sample unverified ids.

Use the printed next steps directly:

```
make verify-review VERIFY_WS=<bundle>/verify_sample.csv
make verify-accept BUNDLE=<bundle> VERIFY_WS=<bundle>/verify_sample.csv
```

Then rerun the category command with one of:

```
--data-verified --verification-ref <bundle>/verify_sample.csv
--data-verified --verification-ref <bundle>/sample_manifest.json
--data-verified --verification-ref <bundle>/accepted
```

---

## Case 4: multi-annotator review and adjudication

When one reviewer's judgment is not enough (a high-stakes goldset, or you want measured
inter-annotator agreement), run the same gate with several annotators:

```bash
make verify-sample     BUNDLE=<bundle> VERIFY_N=30 VERIFY_ANNOTATORS=2
make verify-review     VERIFY_WS=<bundle>/verify_sample.r1.csv    # reviewer 1
make verify-review     VERIFY_WS=<bundle>/verify_sample.r2.csv    # reviewer 2, independently
make verify-adjudicate BUNDLE=<bundle>
make verify-review     VERIFY_WS=<bundle>/adjudication.csv        # decide the disagreements
make verify-accept     BUNDLE=<bundle> VERIFY_ACCEPT_POLICY=per-stratum
```

How it behaves:

- **One sample, k sheets.** All reviewers verify the SAME stratified sample (agreement is only
  defined over shared ratings); each sheet is stamped with its `reviewer_id`.
- **Agreement report.** `verify-adjudicate` writes `agreement.json` beside the worksheets:
  observed agreement, Cohen's kappa (2 reviewers) or Fleiss' kappa (3+), per-reviewer tallies,
  and the disagreement item ids. Rough reading: kappa above ~0.6 is substantial agreement;
  near 0 means the reviewers agree no more than chance -- recalibrate on the four checks before
  trusting the sample.
- **Adjudication.** Disagreements (differing decisions, or unanimous accepts whose edited
  answers differ) land in `adjudication.csv` with human columns blank and every prior verdict
  in a read-only `prior_decisions` column on the card. Decide independently first, then compare.
  Re-running `verify-adjudicate` never loses adjudicator decisions already made.
- **Consensus acceptance.** `verify-accept` (pointed at the ordinary base
  `VERIFY_WS=<bundle>/verify_sample.csv`; the manifest routes it) scores the consensus:
  unanimous decisions stand, adjudicated decisions override, everything else counts as
  undecided and blocks acceptance. Only the consensus feeds the accepted ledger.
- **Acceptance policies.** `VERIFY_ACCEPT_POLICY=global` (default) keeps the single overall
  tolerance; `per-stratum` requires EVERY stratum within its own tolerance (override cells with
  `VERIFY_STRATUM_TOLERANCES="frontier-drafted|final|doc.md=0.1"`); `weighted` compares a
  confidence-weighted reject rate that penalizes rejects on rows the automated signals
  (cross-check + retrieval rank) rated confident. Policies apply to single-reviewer worksheets
  too.

A multi-reviewer bundle's `sample_manifest.json` deliberately cannot serve as a
`--verification-ref` (it has no single `worksheet`); use the accepted ledger after
`verify-accept` passes.

---

## In this repo

- `src/llb/goldset/verify.py` -- stratification, deterministic sampling, the acceptance arithmetic
  (global / per-stratum / confidence-weighted policies), worksheet I/O, and accepted-ledger
  emission; the `sample` / `review` / `adjudicate` / `accept` subcommands.
- `src/llb/goldset/verify_multi.py` -- the multi-annotator lane: per-reviewer worksheets,
  Cohen's/Fleiss' kappa agreement report, the adjudication worksheet, and consensus resolution.
- `src/llb/goldset/verify_session.py` -- the interactive reviewer (`run_session` + the pure
  `parse_command` / `format_card` / `first_undecided_index` pieces).
- `src/llb/goldset/validate.py` -- the structural gate (`make validate-goldset`).
- `src/llb/prep/verified_ledger.py` -- the adoption-by-replacement mechanism behind the flip.
- `tests/llb/goldset/test_goldset_verify.py` -- the strata/sampling/acceptance math, the accepted-ledger
  round-trip through the ledger, and the scripted session loop (no model/endpoint/GPU needed).
- `tests/llb/goldset/test_verify_adjudication.py` -- hand-computed kappa fixtures, the adjudication draw,
  consensus resolution, and each acceptance policy.

The *why* and the papers are in the
[Eval-data verification](human-in-the-loop-evaluation.md#eval-data-verification----human-sample-acceptance-of-ai-drafted-data)
section of the human-in-the-loop manual; see also the
[data-prep guide](../data-prep/data-prep.md) and the
[calibration-tooling manual](calibration-tooling.md) (its verification-side twin).
