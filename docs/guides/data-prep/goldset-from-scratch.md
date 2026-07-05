# Create a gold set (end-to-end workflow)

This is the spine for building a new gold set and getting it to the point where it can **score
real models** -- including the human-assisted steps. The benchmark separates stable public
*development* data from user-specific private *evaluation* data: `make demo-eval` uses the
committed public fixture and never regenerates it, and any runtime import or AI draft stays
`verified=false` (cannot score a model) until a human accepts it.

## At a glance

The full pipeline, one `make` shortcut per stage; stages marked `[HUMAN]` need your judgment:

```text
1. create -> 2. validate -> 3. cross-check -> 4. sample-verify [HUMAN] ->
5. flip via ledger -> 6. calibrate the judge [HUMAN, judged runs only] -> 7. score
```

Step-by-step, with the gate each stage must clear (each links to its detailed section below):

1. [Create the items](#1-create-the-items): committed fixture, manual skeleton, or
   `make ingest-uk-squad GOLDSET_MODE=draft CORPUS=<dir>`. Gate: a gold set + sibling `corpus/`.
2. [Validate](#2-validate-structural-gate): `make validate-goldset GOLDSET=... CORPUS=...`.
   Gate: every span resolves, ids unique, splits disjoint.
3. [Cross-check](#3-cross-check-second-frontier-data-gate):
   `make cross-check-goldset BUNDLE=... CROSS_CHECK_MODEL=<second model>`. Gate: a SECOND model
   re-confirms grounding; passing does NOT set `verified=true`.
4. [Human verification gate](#4-human-verification-gate----human-sample-verify-then-flip-via-the-ledger)
   `[HUMAN]`: `make verify-sample` -> `make verify-review` -> `make verify-accept`.
   Gate: reject rate within `VERIFY_TOLERANCE`; only accepted items may flip.
5. [Flip via the ledger](#44-flip-via-the-ledger-never-hand-edit-the-boolean):
   `python -m llb.prep.ingest_squad ... --verified-goldset <bundle>/accepted/goldset.jsonl`.
   Never hand-edit the `verified` boolean.
6. [Calibrate the judge](#5-if-the-run-is-judged-calibrate-the-judge) `[HUMAN, judged runs
   only]`: `make calibration-run` -> `calibration-rate` -> `calibration-score`.
   Gate: Spearman `rho >= 0.6` admits the judge; otherwise objective scoring ranks alone.
7. [Score](#6-score): `make build-index` -> `make run-eval`.

Stages 3-4 are the **human-assisted** part. The detailed human how-to lives in two operator
manuals this guide links into: [verification tooling](../human-tooling/verification-tooling.md)
(the human verification gate gate) and [calibration tooling](../human-tooling/calibration-tooling.md)
(the judge gate). Everything is offline except the draft/cross-check endpoints and
the eventual model run.

---

## 1. Create the items

Pick the workflow that matches where the data comes from. All of them land a gold set + a sibling
`corpus/`; only the source differs.

### 1a. Stable public development fixture (no creation needed)

The committed `samples/goldsets/ua_squad_postedited_v1/` (250 items, all `verified=true`,
`provenance=public-reused`) ships ready to score. Its README records the upstream revision,
selection rule, verification basis, attribution, and license. Just use it:

```
make validate-goldset
make build-index
make run-eval MODEL=llama3.2:3b LIMIT=20
```

It is already human-verified (upstream post-editing), so it skips stages 3-5 -- go straight to
scoring. Regenerate it only to exercise ingestion:
`make ingest-uk-squad GOLDSET_MODE=development GOLDSET_N=250` (pinned revision/split; its 250 items
re-adopt from the reviewed ledger and all come out verified).

### 1b. Manual skeleton (you author the items)

```
make ingest-uk-squad GOLDSET_MODE=skeleton          # editable template under $DATA_DIR/goldset-skeleton/<ts>/
# edit squad_goldset.json: your contexts + QA pairs
make ingest-squad SQUAD_JSON=<path-to-edited-squad-json>
```

The imported canonical JSONL is `verified=false`. Follow the [authoring rules](#authoring-rules)
below, then take it through stages 2-4.

### 1c. Assisted corpus drafting (ontology-assisted, ontology-assisted drafting)

`GOLDSET_MODE=draft` runs the ontology-assisted pipeline over a corpus: inventory docs, extract
entities + evidence-backed relations, induce an ontology candidate, sample for coverage, and draft
unverified QA -- through one configured endpoint.

```
make ingest-uk-squad GOLDSET_MODE=draft CORPUS=<corpus-dir> DRAFT_MODEL=<tag>
```

The endpoint is LOCAL by default (an OpenAI-compatible server such as Ollama; no corpus leaves the
box); opt into a frontier endpoint with `DRAFT_ENDPOINT=frontier` (egress; needs a provider key).
CLI form: `llb prepare-goldset-draft --corpus-root <dir> --model <id> [--endpoint local|frontier]
[--base-url <url>] [--max-items N]`. It writes a self-contained bundle under
`$DATA_DIR/prepare-goldset/<ts>/`: `goldset.jsonl` (every item `verified=false`,
`provenance=ontology-drafted`, spans exact), a verbatim `corpus/`, the induced `ontology.json`,
per-doc `extraction.jsonl`, and `provenance.json` (endpoint, prompt fingerprints, per-doc hashes,
stage counts, cost). A synthetic-planted bundle (`llb prepare-synthetic-corpus`) is the same shape
with `planted_labels.jsonl` + `provenance.json` carrying `synthetic: true`.

For the rest of this guide, `BUNDLE=$DATA_DIR/prepare-goldset/<ts>` is the drafted bundle.

---

## 2. Validate (structural gate)

Before reading anything, confirm every span resolves to its labeled text, ids are unique, and
splits are disjoint:

```
make validate-goldset GOLDSET=$BUNDLE/goldset.jsonl CORPUS=$BUNDLE/corpus
```

Fully offline. Structural validity is necessary but not sufficient -- it says nothing about factual
correctness; that is what stages 3-4 establish.

---

## 3. Cross-check (second-frontier data gate)

A SECOND, independent frontier model (different from the drafter) re-confirms each item's grounding
+ support + answerability, layered on cheap deterministic pre-checks. This filters the obvious
failures so the human sample only has to confirm the residual.

```
make cross-check-goldset BUNDLE=$BUNDLE CROSS_CHECK_MODEL=<second-frontier id>
```

This writes `$BUNDLE/goldset.cross_check.json` (per-item verdicts + pass count). **Passing does NOT
set `verified=true`** -- it only marks which drafted items are eligible and produces the report the
human samples in stage 4. (`CROSS_CHECK_MODEL` must differ from the drafter -- a model grading its
own drafts is circular.)

---

## 4. human verification gate -- human sample-verify, then flip via the ledger

The irreducibly-human stage: verify a **stratified sample** and accept it; only accepted items flip
to `verified=true`. A clean sample accepts the bundle; a dirty one sends the drafts back. Three
`make` shortcuts (full how-to + the per-item check legend: the
[verification-tooling manual](../human-tooling/verification-tooling.md)):

```
make verify-sample  BUNDLE=$BUNDLE VERIFY_N=30            # stratified sample -> verify_sample.csv
make verify-review  VERIFY_WS=$BUNDLE/verify_sample.csv   # interactive: four checks per item
make verify-accept  BUNDLE=$BUNDLE VERIFY_WS=$BUNDLE/verify_sample.csv VERIFY_TOLERANCE=0.05
```

`verify-review` shows each cited span inside its corpus window and hides the cross-check verdict by
default (so it cannot anchor you -- `SHOW_CROSSCHECK=1` to reveal post-hoc). The four checks:
grounded span / non-circular + answerable / correct reference / planted labels match (synthetic
only). `verify-accept` prints the per-stratum + overall reject rate vs `VERIFY_TOLERANCE` and writes
an accepted-ledger at `$BUNDLE/accepted/` (accepted items, `verified=true`, + their copied corpus).

### 4.4. Flip via the ledger (never hand-edit the boolean)

`verify-accept` prints the exact command. Adopt the accepted items into your scored gold set by
**replacement** through the ingester, so a reused id can never certify changed content:

```
python -m llb.prep.ingest_squad --squad-json <source> --verified-goldset $BUNDLE/accepted/goldset.jsonl
```

human verification gate is **per-bundle and pull-based**: run it on each bundle as its
drafting stabilizes, not once upfront. Keep the `sample_manifest.json` -- it is your record
of what you sampled and the strata you covered.

---

## 5. (If the run is judged) calibrate the judge

Skip this for objective-only scoring. If a board uses the LLM-as-judge (QA faithfulness,
summarization, agentic trajectory quality, free-form text-analysis), calibrate the judge against
human ratings on the new set's `calibration` split first -- the judge enters the ranking blend only
when its Spearman `rho >= 0.6`. Full walkthrough: the
[calibration-tooling manual](../human-tooling/calibration-tooling.md).

```
make calibration-run   GOLDSET=<verified-goldset>.jsonl CAL_NAME=<name>  # answers + ungated judge
make calibration-rate  CAL_NAME=<name>   # interactive: human ratings (judge column hidden)
make calibration-score CAL_NAME=<name>   # rho + bootstrap CI + trust decision
```

Calibration needs `verified=true` calibration items (it scores only verified rows), so it comes
after stage 4. The committed fixture is already calibrated out of the box.

---

## 6. Score

With the gold set verified (and the judge calibrated if judged), it scores models like any other:

```
make build-index   GOLDSET=<verified-goldset>.jsonl CORPUS=<corpus>
make run-eval      MODEL=<tag> GOLDSET=<verified-goldset>.jsonl [JUDGE_RHO=<rho> enables the judge]
```

---

## Authoring rules

For every item:

1. Use a stable unique id and one clear Ukrainian question.
2. Ensure the answer is supported by the supplied context.
3. Copy the answer verbatim and record its zero-based character offset.
4. Avoid ambiguous questions, duplicate facts, and clues that expose the answer trivially.
5. Preserve calibration / tuning / final split isolation after canonical import.
6. Record reviewer, decision, timestamp, and notes in a sidecar review log (the `sample_manifest.json`
   + the reviewed `verify_sample.csv` are that log for the human verification gate stage).

Schema (one JSON object per line): `id, lang, question, reference_answer, source_doc_id,
source_spans[{doc_id, char_start, char_end, text}], provenance, verified, split`. Labels are
SOURCE-SPAN (char offsets, not chunk ids). Only `verified: true` items score models.

## See also

- [Verification tooling](../human-tooling/verification-tooling.md) -- the human verification gate `verify-*`
  operator manual (stages 3-4).
- [Calibration tooling](../human-tooling/calibration-tooling.md) -- the judge `calibration-*`
  operator manual (stage 5).
- [Data prep](data-prep.md) -- the create-stage commands in brief.
- [Human-in-the-loop evaluation](../human-tooling/human-in-the-loop-evaluation.md) -- the *why*
  behind the human gates (acceptance sampling, the ground-truth guarantee, the papers).
