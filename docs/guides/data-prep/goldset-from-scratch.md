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
box). A frontier route needs a provider key, `DRAFT_ENDPOINT=frontier`, an interactive consent
prompt that names the corpus and destination, and a call or spend cap:

```bash
make prepare-goldset-draft \
  DRAFT_CORPUS=<corpus-dir> \
  DRAFT_ENDPOINT=frontier \
  DRAFT_FRONTIER_MODEL=<litellm-model-id> \
  DRAFT_MAX_USD=<usd-cap> \
  DRAFT_MAX_CALLS=<call-cap>
```

Use `DRAFT_FRONTIER_STAGE=extraction|drafting|both`; either mixed route also needs
`DRAFT_LOCAL_MODEL=<local-model>`. The CLI equivalents are `--frontier-stage`, `--local-model`,
`--max-usd`, and `--max-calls`. It writes a self-contained bundle under
`$DATA_DIR/prepare-goldset/<ts>/`: `goldset.jsonl` (every item `verified=false`,
`provenance=ontology-drafted`, spans exact), a verbatim `corpus/`, the induced `ontology.json`,
per-doc `extraction.jsonl`, and `provenance.json` (endpoint, prompt fingerprints, per-doc hashes,
stage counts, per-call cost/latency telemetry). Budget exhaustion leaves an inspectable aborted
provenance record and extraction journal rather than deleting partial work.

For an exact shared-seed local/frontier comparison, run:

```bash
make draft-compare \
  DRAFT_COMPARE_CORPUS=<corpus-dir> \
  DRAFT_COMPARE_SEEDS=<n> \
  DRAFT_COMPARE_LOCAL_MODEL=<local-model> \
  DRAFT_COMPARE_FRONTIER_MODEL=<litellm-model-id> \
  DRAFT_COMPARE_MAX_USD=<usd-cap>
```

The comparison root under `$DATA_DIR/draft-compare/<ts>/` contains both bundles, both verification
worksheets, and `comparison.json` with shared seed fingerprints, parse rate, kept yield, gate
results, and yield/accept-rate rankings. Accept rates remain pending until a human reviews the
worksheets. After review, update only the report (no provider calls or spend) with:

```bash
make draft-compare-report \
  DRAFT_COMPARE_OUT_DIR=<comparison-root> \
  DRAFT_COMPARE_LOCAL_VERIFICATION=<reviewed-local-csv> \
  DRAFT_COMPARE_FRONTIER_VERIFICATION=<reviewed-frontier-csv>
```

### Finish the bounded Ukrainian local comparison

`make local-ua-draft-probe` uses the committed synthetic two-document fixture at
`samples/text_analysis_bundle_uk/corpus`. It detects the GPU tier through the same hardware path as
benchmark serving, selects a Qwen baseline and Gemma probe that fit individually, and requires both
Ollama tags to exist before starting.

| GPU tier | Qwen baseline | Gemma probe | context |
| --- | --- | --- | ---: |
| 12 GiB | `qwen3:8b` | `gemma4:e2b` | 8192 |
| 16 GiB | `qwen3:14b` | `gemma4:e4b` | 8192 |
| 24 GiB | `qwen3:30b` | `gemma4:26b` | 8192 |
| 32 GiB | `qwen3:30b` | `gemma4:31b` | 16384 |

The runner unloads all resident Ollama models before starting, runs Qwen extraction and drafting,
strictly unloads Qwen, runs Gemma on the exact shared seed objects, and unloads Gemma before exit.
An unload timeout aborts the workflow instead of allowing the models to overlap in VRAM.

For any generated artifact, use one abstract root consistently:

```bash
source scripts/shared/common.sh
llb_load_env
export COMPARISON_ROOT=<comparison-root>
make local-ua-draft-analyze LOCAL_DRAFT_COMPARE_OUT_DIR="$COMPARISON_ROOT"
make local-ua-draft-complete LOCAL_DRAFT_COMPARE_OUT_DIR="$COMPARISON_ROOT"
```

Use `y` to accept, `x` to reject, `h` for help, and `q` to save and pause. Re-run the review command
to resume. Finalization reads both completed worksheets and updates `comparison.json` without any
model call.

For a new CUDA host, install the two tags named by any missing-model error and run:

```bash
make local-ua-draft-probe LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root>
make local-ua-draft-complete LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root>
```

Use the analytics command for a compact table, a machine-readable normalized view, or a gate check:

```bash
make local-ua-draft-analyze LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root>
make local-ua-draft-analyze LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root> \
  COMPARE_ANALYZE_JSON=1
make local-ua-draft-analyze LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root> \
  COMPARE_REQUIRE_GATES=1
```

The table reports model order, unload policy, shared seeds, parsed drafts, kept yield, calibration,
calls, drafting latency, live worksheet progress, human accept rate, and probe-minus-baseline
deltas. Preserve a failing artifact and its output; do not change human decisions to force a pass.

A synthetic-planted bundle (`llb prepare-synthetic-corpus`) is the same shape
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

### 6b. Score an already-answered external RAG log

Use this when the RAG system under analysis is outside this codebase and has already answered the
gold questions. The input is one JSON object per line with the normal gold fields plus an answer
field such as `llm_answer` or `predicted_answer`; optional `llm_sources` are shown in the
interactive card and flattened into the final CSV. This is an external-system diagnostic, not a
certified local benchmark leaderboard. If the rows are still `verified=false`, treat the result as
an estimate until the human review fields are filled.

```
make score-external-rag \
  EXTERNAL_RAG_ANSWERS=<answered-jsonl>
```

The command opens an interactive human scoring session. Each row shows the question,
`reference_answer`, gold source text, raw `llm_answer`, scored answer text, first returned
`llm_sources`, and `llm_error`. Use:

```
a        accept, score=1
p        partial, score=0.5
r        reject, score=0
s <0..1> explicit score
o        edit human_notes
w        edit human_corrected_answer
n/b/u/j  navigate
q        save and quit
```

Intermediate state is written back into the same JSONL as `human_score_0_1`,
`human_decision`, `human_notes`, `human_corrected_answer`, and `human_status`. Re-run the same
command to resume at the first unscored row. To restart scoring, use:

```
make score-external-rag \
  EXTERNAL_RAG_ANSWERS=<answered-jsonl> \
  EXTERNAL_RAG_CLEAR=1
```

Only after every row has a human score and decision does the command write final artifacts. By
default they are:

```
<answered-jsonl-stem>.csv
<answered-jsonl-stem>.report.md
```

Use explicit paths or non-standard field names when needed:

```
make score-external-rag \
  EXTERNAL_RAG_ANSWERS=<answered-jsonl> \
  EXTERNAL_RAG_CSV=<out.csv> \
  EXTERNAL_RAG_REPORT=<report.md> \
  EXTERNAL_RAG_ANSWER_FIELD=predicted_answer \
  EXTERNAL_RAG_SOURCES_FIELD=sources
```

The scorer strips a trailing `Source:` footer before objective scoring but keeps the
raw answer in the CSV. If your external API can return corpus `doc_id`, `char_start`, and
`char_end` for sources, include them in `llm_sources`; otherwise source-span recall cannot be
computed for the external system.

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
