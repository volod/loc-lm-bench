# Judge-calibration tooling -- how to use it

This is the operator manual for the three `calibration-*` commands that turn the judge
calibration gate judge gate into a reproducible workflow. The *why* (the gate, the bias
it guards against, the papers) lives in the human-in-the-loop manual's
[Judge calibration](human-in-the-loop-evaluation.md#judge-calibration----validating-llm-as-judge-against-human-ratings)
section; this page is the *how*.

Calibration answers one question: **does the local judge rank Ukrainian answers the same way a
human does?** You produce two columns over the same answers -- the judge's `judge_rating` and
your `human_rating` -- and measure their Spearman rank correlation. `rho >= 0.6` admits the
judge into the ranking blend; below it the judge stays a demoted diagnostic and objective
correctness ranks alone. The decision travels in the run manifest.

## At a glance

```text
verified goldset with a calibration split
  -> 1. pre-fill worksheet      make calibration-run    [needs candidate + judge endpoints]
  -> 2. rate each item [HUMAN]  make calibration-rate   [judge column hidden -- rate first]
  -> 3. score the agreement     make calibration-score  [gate: Spearman rho >= 0.6]
```

Step-by-step on the committed goldset (details in [Case 1](#case-1-the-canonical-committed-goldset-default)):

```bash
make calibration-run      # candidate answers + ungated judge ratings -> worksheet
make calibration-rate     # the human step: 1-5 ratings, judge hidden
make calibration-score    # rho + bootstrap CI + trust decision
```

Only step 2 is human-paced, and only step 1 needs a GPU/endpoint. The gate: `rho >= 0.6` admits
the judge into the ranking blend; a demotion below the gate is the mechanism working, not a bug.
For your own goldset add `GOLDSET=<path> CAL_NAME=<name>` ([Case 2](#case-2-a-new-goldset-you-built));
an unverified corpus draft must clear the verification gate first
([Case 3](#case-3-a-draft-generated-from-a-text-corpus)).

## The three commands

| Command | What it does | Needs |
| --- | --- | --- |
| `make calibration-run` | Runs a candidate over the `calibration` split and writes a **pre-filled worksheet** (`model_answer` + ungated `judge_rating`). | candidate + judge endpoints |
| `make calibration-rate` | **Interactive rater**: walk the worksheet item by item and fill the human columns. | nothing (offline; CSV only) |
| `make calibration-score` | Computes `rho` + bootstrap CI + the mechanical trust decision. | nothing (offline; CSV only) |

Only `calibration-run` needs a GPU/endpoint. Rating and scoring are pure CSV operations -- you
can rate on a laptop, on a plane, in several sittings.

## The worksheet

The worksheet is a single CSV at `CAL_WS` (default `calibration/ua_squad_postedited_v1.csv`), kept
in one of **two roots**, chosen automatically by `CAL_NAME`:

- **permanent** (committed -> survives a clone): the tracked root `calibration/` dir, for names
  listed in the Makefile's `CAL_PERMANENT` (the committed goldset by default);
- **temporary** (gitignored): `$DATA_DIR/llb/calibration/`, where every other `CAL_NAME`
  (generated / in-progress sets) auto-routes.

So:

- committed canonical goldset -> `make calibration-run` -> `calibration/ua_squad_postedited_v1.csv`
- skeleton goldset -> `make calibration-run CAL_NAME=skeleton …` -> `$DATA_DIR/llb/calibration/skeleton.csv`
- text-corpus goldset -> `make calibration-run CAL_NAME=<corpus> …` -> `$DATA_DIR/llb/calibration/<corpus>.csv`

To persist a generated set, copy it into `calibration/` and add its name to `CAL_PERMANENT`. This
two-root split avoids a brittle `.gitignore` exception -- the whole `calibration/` dir is committed,
generated sets live elsewhere. See [`calibration/README.md`](../../../calibration/README.md).

The worksheet **is** the session state -- every interactive edit rewrites only the human columns,
merged into the file atomically, so resume and crash-safety are free. Columns:

| Column | Filled by | Meaning |
| --- | --- | --- |
| `item_id`, `split` | `calibration-run` | the gold item this row scores |
| `provenance` | `calibration-run` | the item's source (`public-reused`, `human-authored`, `frontier-drafted`, ...); lets you see, per card, whether you are rating reused vs AI-drafted data |
| `question`, `reference_answer` | `calibration-run` | the inputs you rate against |
| `model_answer` | `calibration-run` | the candidate's answer (what you are rating) |
| `human_answer` | **you** (`a` command) | your own reference answer for the item |
| `human_rating` | **you** (a number) | your 1-5 rating of `model_answer` |
| `human_note` | **you** (`note` command) | optional free-text note |
| `human_status` | the rater | `pending` / `rated` (a refinement; resume keys on an empty `human_rating`) |
| `judge_rating` | `calibration-run` | the judge's [0,1] score -- **hidden in the rater by default** |

---

## Case 1: the canonical committed goldset (default)

`samples/goldsets/ua_squad_postedited_v1` ships with the repo and is what `GOLDSET` points at by
default. Its `calibration` split is 86 `verified=true` post-edited SQuAD-uk items
(`provenance=public-reused`) -- the canonical set the gate is calibrated on out of the box. You
do not need to build anything; just run the three steps.

### 1. Stand up a judge endpoint and pre-fill the worksheet

The defaults target a local **Ollama** judge (`gemma3:27b` on `:11434`), and the embedder is pinned
to CPU (`LLB_EMBED_DEVICE=cpu`) so the GPU stays free for the judge -- so on the committed goldset
this is just:

```
make calibration-run            # Ollama gemma3:27b judge, llama3.2:3b candidate (defaults)
```

To use a **vLLM** judge instead (e.g. the 16 GB QAT 12B served on `:8000`), override the knobs --
on 16 GB a 12B vLLM judge usually cannot co-reside with a vLLM candidate, so keep the candidate on
Ollama or serve the judge on another host:

```
make calibration-run \
    MODEL=llama3.2:3b BACKEND=ollama \
    JUDGE_MODEL=google/gemma-4-12B-it-qat-w4a16-ct \
    JUDGE_BASE_URL=http://127.0.0.1:8000/v1
```

Either way this runs the candidate over the 86 calibration items and writes `CAL_WS` with
`model_answer` and an **ungated** `judge_rating` (the threshold is irrelevant here -- calibration
measures agreement, not trust). If the judge endpoint is unreachable, `judge_rating` is left blank
and a warning is logged; fix the endpoint and re-run before rating.

### 2. Rate independently with the interactive rater

```
make calibration-rate
```

This opens the rater on `CAL_WS` and resumes at the first unrated item. The judge's rating is
**hidden by default**: if you see it first you will unconsciously copy it, and the correlation
would then measure whether you echoed the judge instead of whether the judge matches an
*independent* human -- which is the whole point of the exercise. So rate before you peek. For each
item: read the question, reference, and the candidate's `model_answer`; author your own answer
(`a`); then give a 1-5 rating.

#### Rater commands

| Key | Action |
| --- | --- |
| `1`-`5` | set the rating and advance to the next item |
| `a` | author/edit `human_answer` (prompts for one line; empty clears it) |
| `note` | edit `human_note` (prompts for one line; empty clears it) |
| `n` / Enter | next item (no change) |
| `p` / `b` | previous item (go back to change an answer) |
| `j <N>` | jump to item N (1-based) |
| `u` | jump to the next unrated item |
| `c` | clear this item's rating |
| `?` / `h` | help + the rating anchors |
| `q` | save and quit |

Rating anchors (1-5 Likert; Spearman is rank-based, so this maps cleanly onto the judge's
[0,1] scale): **1** = wrong / unfaithful, **2** = mostly wrong, **3** = partially correct,
**4** = mostly correct, **5** = fully correct + faithful.

How to rate well (from the manual, condensed): span the full range; **deliberately include
fluent-but-wrong answers** (the failure mode the judge is most likely to miss); rate against the
reference, not your prior knowledge; apply a consistent rubric.

Useful options:

```
make calibration-rate START=20      # begin at item 20
make calibration-rate SHOW_JUDGE=1  # reveal judge_rating (POST-HOC review only -- it anchors)
make calibration-rate CLEAR=1       # wipe all human columns and start fresh (confirmation-gated)
```

Ctrl-C, EOF, and `q` all save and quit; the last edit is already on disk (write-through), so you
never lose work. Re-running `make calibration-rate` continues where you left off.

> **Re-running `calibration-run` is safe.** It MERGES your existing human columns into the
> freshly pre-filled worksheet by `item_id` -- your ratings survive a re-run with the same
> deterministic candidate. If a regenerated `model_answer` actually CHANGED (you pointed it at a
> different candidate), the stale rating for that row is cleared with a warning (it no longer
> applies to the shown answer); your authored `human_answer` is kept either way.

### 3. Score it

```
make calibration-score
```

(`RATINGS` defaults to `CAL_WS`.) This prints `rho`, the bootstrap CI, and the mechanical
decision: `rho >= 0.6` admits the judge, otherwise it stays demoted. A small local judge may not
clear the gate for Ukrainian, and a 12B is borderline -- a demotion is the gate working as
designed, not a bug.

---

## Case 2: a new goldset you built

To calibrate against your own set instead of the committed fixture, point `GOLDSET` at a JSONL
that has a `calibration` split with `verified=true` items. (`run-eval` scores **only**
`verified=true` items -- an unverified set produces zero calibration rows.) See
[goldset-from-scratch](../data-prep/goldset-from-scratch.md) for building and splitting one.

Set `CAL_NAME` so the worksheet auto-routes to its own gitignored file under
`$DATA_DIR/llb/calibration/` (copy it into `calibration/` + add to `CAL_PERMANENT` to persist):

```
make calibration-run  GOLDSET=path/to/your/goldset.jsonl CAL_NAME=my_goldset   # default Ollama judge
make calibration-rate  CAL_NAME=my_goldset
make calibration-score CAL_NAME=my_goldset
```

The judge defaults to the Ollama `gemma3:27b` endpoint; add `JUDGE_MODEL=… JUDGE_BASE_URL=…` to
point at a vLLM judge instead (as in Case 1). The rater and scorer are identical to Case 1 -- only
the source of the items changes. The `provenance` column on each card tells you what kind of item
you are rating (e.g. `human-authored` vs `public-reused`), worth watching when a set mixes sources.

---

## Case 3: a draft generated from a text corpus

A draft produced by the ontology-assisted drafting pipeline (`prepare-goldset-draft` over a corpus) lands
`verified=false`, because its reference answers are AI-drafted and not yet human-checked. Rating
a candidate against an **unverified** reference would inject that noise straight into the
calibration, so the draft cannot be calibrated as-is -- `run-eval` will refuse it (no verified
items). Promote a calibration subset to verified first, exactly as for any scored data:

1. **Draft from the corpus** (ontology-assisted drafting):

   ```
   make ingest-uk-squad GOLDSET_MODE=draft CORPUS=path/to/corpus
   # writes a bundle under $DATA_DIR/prepare-goldset/<ts>/ (goldset.jsonl + corpus/)
   ```

2. **Second-frontier cross-check** (a different model re-confirms grounding/support):

   ```
   llb cross-check-goldset --goldset <bundle>/goldset.jsonl \
       --corpus <bundle>/corpus --model <second-frontier-id>
   ```

3. **Human sample-verify (human verification gate)** -- structural validate, draw a stratified
   sample over a calibration subset, check each item, then flip accepted items to `verified=true`
   through the **ledger** (never hand-edit the boolean). Full procedure:
   [Eval-data verification](human-in-the-loop-evaluation.md#eval-data-verification----human-sample-acceptance-of-ai-drafted-data).

   ```
   make validate-goldset GOLDSET=<bundle>/goldset.jsonl CORPUS=<bundle>/corpus
   python -m llb.prep.ingest_squad ... --verified-goldset <accepted-ledger>
   ```

4. **Calibrate against the verified ledger** -- now it has `verified=true` calibration items, so
   it behaves like Case 2 (give it its own `CAL_NAME`; the default Ollama judge applies, or
   override with `JUDGE_MODEL=… JUDGE_BASE_URL=…` for vLLM):

   ```
   make calibration-run  GOLDSET=<verified-ledger>.jsonl CAL_NAME=<corpus>
   make calibration-rate  CAL_NAME=<corpus>
   make calibration-score CAL_NAME=<corpus>
   ```

> **Quick judge sanity on an unverified draft is intentionally NOT supported here** (it would
> need the worksheet to bypass the `verified` filter, and a headline calibration must never run
> on unverified references). For a fast "is the judge wired and sane?" check that does not touch
> the gate, use the recorded fixed cases instead: `make judge-experiment` (see
> [judge-experiments](judge-experiments.md)).

---

## In this repo

- `src/llb/judge/calibration.py` -- stats (Spearman + bootstrap CI + trust decision) and the
  worksheet I/O (schema, atomic load/save, merge-on-regenerate); the `worksheet` / `score` /
  `rate` subcommands.
- `src/llb/judge/rate/` -- the interactive rater (`run_session` + the pure
  `parse_command` / `format_card` / `first_unrated_index` pieces).
- `tests/llb/judge/test_calibration.py`, `tests/llb/judge/test_rate.py` -- the stats, the
  worksheet round-trip + merge, and the scripted session loop (no model/endpoint/GPU needed).
