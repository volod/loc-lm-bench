# Data-verification tooling -- how to use it

This is the operator manual for the three `verify-*` commands that turn the MH.5 human
sample-acceptance gate into a reproducible workflow. The *why* (the gate, the human-ground-truth
guarantee, the acceptance-sampling framing, the papers) lives in the human-in-the-loop manual's
[Eval-data verification](human-in-the-loop-evaluation.md#eval-data-verification----human-sample-acceptance-of-ai-drafted-data)
section; this page is the *how*. It is the verification-side twin of the
[calibration-tooling manual](calibration-tooling.md).

Verification answers one question: **can the AI-drafted, frontier-cross-checked data be trusted to
score models?** The pipeline already drafts items and a *second* frontier model cross-checks each
one for grounding + support. MH.5 is the irreducibly-human stage: you verify a **stratified sample**
and accept it, and only then may those items flip to `verified=true` and score real models. A clean
sample accepts the bundle; a dirty one sends the drafts back. Nothing scores a real model until its
data clears this gate.

## The three commands

| Command | What it does | Needs |
| --- | --- | --- |
| `make verify-sample` | Draws a **stratified sample** from a draft bundle and writes a verification worksheet + a `sample_manifest.json` (size + strata). | nothing (offline; reads the bundle) |
| `make verify-review` | **Interactive verifier**: walk the sample item by item, run the four checks against the corpus, accept/reject. | nothing (offline; CSV only) |
| `make verify-accept` | Computes the per-stratum + overall reject rate vs tolerance and emits the **accepted-ledger** bundle. | nothing (offline; CSV only) |

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
| `cc_grounded`, `cc_non_circular`, `cc_supported`, `cc_answerable`, `cc_note` | `verify-sample` | the second-frontier verdict -- **hidden in the reviewer by default** |
| `chk_grounded`, `chk_answerable`, `chk_reference`, `chk_planted` | **you** | the four per-item checks: `pass` / `fail` / "" (planted is "" / N/A for real items) |
| `decision` | **you** (`y` / `x`) | `accept` / `reject` / "" |
| `human_note` | **you** (`note` command) | optional free-text note (record the reason for any `fail`) |
| `human_status` | the reviewer | `pending` / `decided` (resume keys on an empty `decision`) |

The four checks (the same ones the second frontier ran -- you are confirming it did not
*systematically* err):

1. **`chk_grounded`** -- the cited offsets really support the answer/label (read the `context`).
2. **`chk_answerable`** -- the item is answerable and non-circular (the question does not leak its
   own answer).
3. **`chk_reference`** -- the reference answer is correct.
4. **`chk_planted`** -- *synthetic only*: the planted labels match what the doc actually says.

---

## Case 1: a draft from a real corpus (the common case)

A bundle from the M4.4 pipeline (`prepare-goldset-draft` over a corpus) lands `verified=false` with
its gold file as `goldset.jsonl`. This is the normal path.

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

This opens the reviewer and resumes at the first undecided item. Each card shows the question, the
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
| `x` | reject this item and advance |
| `note` | edit `human_note` (prompts for one line; empty clears it) |
| `n` / Enter | next item (no change) |
| `b` / arrows | previous item (go back to change a mark) |
| `j <N>` | jump to item N (1-based) |
| `u` | jump to the next undecided item |
| `c` | clear this item's marks + decision |
| `?` / `h` | help + the check legend |
| `q` | save and quit |

(The planted check is refused on a real, non-synthetic item -- it is N/A there.)

Useful options:

```
make verify-review VERIFY_WS=… START=20           # begin at item 20
make verify-review VERIFY_WS=… SHOW_CROSSCHECK=1   # reveal cc_* (POST-HOC review only -- it anchors)
make verify-review VERIFY_WS=… CLEAR=1             # wipe all human columns and start fresh (gated)
```

Ctrl-C, EOF, and `q` all save and quit; the last edit is already on disk (write-through), so you
never lose work. Re-running `make verify-review` continues where you left off.

### 3. Accept (or send it back) and flip via the ledger

```
make verify-accept BUNDLE=$DATA_DIR/prepare-goldset/<ts> \
    VERIFY_WS=$DATA_DIR/prepare-goldset/<ts>/verify_sample.csv
```

This prints the acceptance report -- the per-stratum and overall **reject rate** against
`VERIFY_TOLERANCE` (default `0.05`) -- and writes the **accepted-ledger** bundle at
`<bundle>/accepted/`: the items you accepted, with `verified=true`, plus their copied corpus docs,
so the ledger is self-contained. It also flags any item that has a failed check but no decision
(`undecided_with_failures`) so nothing slips through unreviewed.

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
[goldset-from-scratch](goldset-from-scratch.md) for the ledger mechanics.

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

MH.5 is **not** a one-shot upfront task -- it runs in parallel with the build and is pull-based:
as each category's bundle is drafted + cross-checked and **stabilizes**, verify *that* bundle, flip
its accepted items, and real-model scoring unblocks for it. The objective category boards never wait
on this -- only real-model HEADLINE scoring does. Practical notes:

- Verify a bundle only once its drafting is stable; do not re-verify seeds a newer bundle
  supersedes.
- Each bundle gets its own `verify_sample.csv` (it lives inside the bundle dir), so several
  categories' verifications can be in flight independently.
- Keep the `sample_manifest.json` -- it is your record of what you sampled and the strata you
  covered (the "document the sample" discipline from *Datasheets for Datasets*).

---

## In this repo

- `src/llb/goldset/verify.py` -- stratification, deterministic sampling, the acceptance arithmetic,
  worksheet I/O, and accepted-ledger emission; the `sample` / `review` / `accept` subcommands.
- `src/llb/goldset/verify_session.py` -- the interactive reviewer (`run_session` + the pure
  `parse_command` / `format_card` / `first_undecided_index` pieces).
- `src/llb/goldset/validate.py` -- the structural gate (`make validate-goldset`).
- `src/llb/prep/verified_ledger.py` -- the adoption-by-replacement mechanism behind the flip.
- `tests/test_goldset_verify.py` -- the strata/sampling/acceptance math, the accepted-ledger
  round-trip through the ledger, and the scripted session loop (no model/endpoint/GPU needed).

The *why* and the papers are in the
[Eval-data verification](human-in-the-loop-evaluation.md#eval-data-verification----human-sample-acceptance-of-ai-drafted-data)
section of the human-in-the-loop manual; see also the [data-prep guide](data-prep.md) and the
[calibration-tooling manual](calibration-tooling.md) (its verification-side twin).
