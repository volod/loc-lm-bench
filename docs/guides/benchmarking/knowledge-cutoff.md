# Measure a Local Model's Knowledge Cutoff

Use this benchmark to estimate the latest month for which a local model reliably knows
unpredictable public events. It is a command-driven diagnostic, not an interactive application,
and it works with the same Ollama, vLLM, llama.cpp, canonical artifact, and MLflow paths as the
rest of loc-lm-bench.

## Run the benchmark

The standard workflow loads the public event set from Hugging Face, resolves `main` to an exact
commit, evaluates the full time range, fits the decay curve with a seeded Optuna study, and writes
the report bundle:

```sh
make bench-knowledge-cutoff MODEL=<model> BACKEND=ollama
```

For an already running local OpenAI-compatible endpoint:

```sh
make bench-knowledge-cutoff MODEL=<model> BACKEND=vllm \
  KNOWLEDGE_CUTOFF_BASE_URL=http://127.0.0.1:8000/v1
```

Pin a dataset commit when a run must be exactly repeatable without resolving a moving branch:

```sh
make bench-knowledge-cutoff MODEL=<model> BACKEND=ollama \
  KNOWLEDGE_CUTOFF_REVISION=<40-character-hf-commit>
```

An operator-provided JSONL file bypasses Hugging Face and is content-addressed in the manifest:

```sh
make bench-knowledge-cutoff MODEL=<model> BACKEND=ollama \
  KNOWLEDGE_CUTOFF_EVENTS=<events-jsonl>
```

`KNOWLEDGE_CUTOFF_LIMIT=<n>` is only for connectivity smokes. The selector spreads the cap across
the complete event horizon, but a limited run can be too sparse for the fit and must not be used as
a model claim. `KNOWLEDGE_CUTOFF_TRIALS` and `KNOWLEDGE_CUTOFF_SEED` bound and reproduce the Optuna
fit; the defaults are 200 and 42.

## What is measured

The pipeline uses the external project only as a dataset and methodology reference. Its scoring
and application architecture are not copied.

1. The loader validates every dated event and records the exact dataset revision.
2. Only real events tagged low or medium predictability enter the decay fit. High-predictability
   events can be guessed from old trends and are excluded.
3. Each four-choice question is relabeled with a stable per-event permutation. This removes the
   upstream answer-position distribution as a shortcut while keeping reruns identical.
4. The prompt contains neither the current date nor a statement that the model is taking a recency
   test. Temperature-zero local inference asks for one answer letter.
5. Monthly correct, incorrect, and unparseable/abstain counts remain visible in the report.
6. A seeded Optuna study fits a monotone logistic decay with a fixed 0.25 four-choice chance floor,
   a learned early-period ceiling, midpoint, and scale. The primary effective cutoff is the fitted
   midpoint between that ceiling and chance.
7. Living-person and fake-event rows do not enter the curve. They report over-prediction and
   confabulation diagnostics beside it.

The report also includes a raw `last_above` threshold month and the first month after which all
remaining observations stay below the threshold. These are audit aids; the Optuna decay midpoint
is the primary estimate because real knowledge usually thins gradually instead of ending at a
perfect cliff.

## Artifacts and MLflow

Each run writes:

```text
$DATA_DIR/knowledge-cutoff/<run_timestamp>/
|-- manifest.json
|-- scores.jsonl
|-- report.json
`-- report.md
```

`scores.jsonl` is the event-level evidence, including the model's raw answer, balanced choice
order, expected/selected letters, eligibility, and objective score. `report.json` is the stable
machine-readable summary; `report.md` is the operator view. The manifest records model/backend,
dataset identity and revision, license context, Optuna controls and best fit, aggregate accuracy,
parse reliability, and throughput. The shared MLflow mirror logs both canonical and report
artifacts after the local bundle exists.

## Calibrate English against Ukrainian

The bilingual workflow translates the exact revision-pinned English questions and choices, keeps
their source answer identities and deterministic choice permutations aligned, and requires a
complete review worksheet before either lane can support a claim. Drafting is local and resumable;
it does not make a translation accepted.

First pin the upstream commit and draft the translation bundle with a local model:

```sh
make knowledge-cutoff-ua-draft \
  KNOWLEDGE_CUTOFF_REVISION=<40-character-hf-commit> \
  KNOWLEDGE_CUTOFF_UA_TRANSLATOR_MODEL=<local-model> \
  KNOWLEDGE_CUTOFF_UA_TRANSLATOR_BACKEND=vllm \
  KNOWLEDGE_CUTOFF_MAX_MODEL_LEN=1024
```

The draft gate rejects non-Ukrainian output, changed numeric clues, malformed JSON, missing choices,
and duplicate choices. It retries one invalid model response and saves each valid row immediately,
so repeating the target resumes rather than retranslating valid drafts.

Review every row in the shared terminal session:

```sh
make knowledge-cutoff-ua-review \
  KNOWLEDGE_CUTOFF_UA_BUNDLE=<translation-bundle>
```

The worksheet shows both questions and both ordered choice lists. Lowercase `g`, `a`, `r`, and `p`
mark factual equivalence, Ukrainian fluency, source-answer preservation, and absence of added
temporal clues as passing; uppercase marks failure. On this translation profile, `y` records the
aggregate acceptance and fills any unchecked criteria as passing; it refuses acceptance if a check
is explicitly failed. Use `x` to exclude a row, `o` to record a revision note, and `q` to save and
resume later.

To repair rejected model wording without hand-editing the stateful worksheet, put replacements in a
JSONL file. Each row contains `item_id`, `question_uk`, and the four-element `choices_uk` array, then
apply it through the same automatic gates:

```sh
make knowledge-cutoff-ua-revise \
  KNOWLEDGE_CUTOFF_UA_BUNDLE=<translation-bundle> \
  KNOWLEDGE_CUTOFF_UA_REVISIONS=<revisions-jsonl>
```

Repeat the draft target afterward. A complete draft skips model startup and refreshes the
worksheet; unchanged rows retain their decisions, while revised rows return to pending review.

Check mechanical alignment and review progress at any point; this exits nonzero until the bundle is
ready to freeze:

```sh
make knowledge-cutoff-ua-validate \
  KNOWLEDGE_CUTOFF_UA_BUNDLE=<translation-bundle>
```

After every row is accepted or excluded, freeze the accepted lanes with a bilingual reviewer
sign-off:

```sh
make knowledge-cutoff-ua-freeze \
  KNOWLEDGE_CUTOFF_UA_BUNDLE=<translation-bundle> \
  KNOWLEDGE_CUTOFF_UA_REVIEWER=<reviewer-name-or-id>
```

The freeze fails on an undecided row, an accepted row without all four passing checks, changed
source identity, an empty accepted set, or an English/Ukrainian answer-key mismatch. It writes
aligned `events.en.reviewed.jsonl`, `events.uk.reviewed.jsonl`, the accepted worksheet snapshot,
and `review_summary.json` inside the translation bundle.

Run both lanes through one local backend lifecycle and emit the paired report:

```sh
make bench-knowledge-cutoff-bilingual \
  MODEL=<model> BACKEND=vllm \
  KNOWLEDGE_CUTOFF_UA_BUNDLE=<translation-bundle>
```

The output under `$DATA_DIR/knowledge-cutoff-bilingual/<run_timestamp>/` contains both event-level
lanes, both cutoff reports, per-month Ukrainian-minus-English accuracy deltas, and a seeded paired
95% bootstrap interval. The source-choice mapping is checked again from the scored rows. Rejected
translations never enter either lane.

For a new bundle, the interactive umbrella target runs draft, review, freeze, and paired scoring in
order:

```sh
make knowledge-cutoff-bilingual \
  MODEL=<model> BACKEND=vllm \
  KNOWLEDGE_CUTOFF_REVISION=<40-character-hf-commit> \
  KNOWLEDGE_CUTOFF_UA_TRANSLATOR_MODEL=<local-model> \
  KNOWLEDGE_CUTOFF_UA_REVIEWER=<reviewer-name-or-id>
```

Read the result as an estimate of effective recall on this event distribution. It is not proof of
the model's training-data boundary. Benchmark contamination, later fine-tuning, quantization,
event selection, sparse months, forced-choice guessing, and English comprehension for a
language-specialized model can all move the curve. Compare models only on the same resolved dataset
revision and fit settings, and inspect controls plus monthly evidence before publishing a cutoff
claim.

## Inspiration, copyright, and dataset attribution

The benchmark idea and public dataset are inspired by
[Apoorv Saxena's `knowledge-cutoff` project](https://github.com/apoorvumang/knowledge-cutoff) and
its [Hugging Face dataset](https://huggingface.co/datasets/apoorvumang/knowledge-cutoff-benchmark).
The upstream dataset card identifies the dataset as CC BY 4.0. Downloaded event data and derived
redistributions remain subject to those terms; preserve creator attribution and source links.

Attribution notice: "Knowledge Cutoff Benchmark" by Apoorv Saxena, licensed CC BY 4.0. Source:
[Hugging Face dataset](https://huggingface.co/datasets/apoorvumang/knowledge-cutoff-benchmark).

No upstream application source was copied into loc-lm-bench. This implementation uses the
project's own local-backend lifecycle, position balancing, Optuna model, canonical run bundle,
report generator, and MLflow mirror. The dataset license does not replace the repository's source
code license.
