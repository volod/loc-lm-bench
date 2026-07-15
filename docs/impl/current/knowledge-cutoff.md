# Local Model Knowledge-Cutoff Benchmark

## Delivered behavior

`llb bench-knowledge-cutoff` and `make bench-knowledge-cutoff` run a non-interactive effective
knowledge-cutoff benchmark against local Ollama, vLLM, llama.cpp, or an already running local
OpenAI-compatible endpoint. The implementation lives in
`src/llb/bench/knowledge_cutoff/`, with Typer wiring in
`src/llb/cli/bench/knowledge_cutoff.py` and Make orchestration in
`make/eval/knowledge-cutoff.mk`.

The loader accepts an operator JSONL file or the `events`/`train` configuration of
`apoorvumang/knowledge-cutoff-benchmark`. Moving Hugging Face revisions are resolved to an exact
commit before loading; a supplied 40-character commit is used directly. Local files are recorded
by SHA-256. Dataset imports are lazy behind the `cutoff` extra.

The benchmark uses project-native methodology:

- stable per-event answer permutation removes source answer-position bias;
- prompts disclose neither the current date nor that this is a recency test;
- low/medium-predictability real events alone feed the monthly curve;
- deterministic letter parsing produces correct/incorrect/abstain evidence;
- a seeded Optuna study fits a monotone logistic curve with a fixed four-choice chance floor and
  learned ceiling, cutoff midpoint, and scale;
- living-person and fake-event rows stay outside the fit and expose over-prediction/confabulation;
- raw threshold landmarks remain in the report beside the primary Optuna estimate.

Canonical output is
`$DATA_DIR/knowledge-cutoff/<run_timestamp>/{manifest.json,scores.jsonl,report.json,report.md}`.
Reports join the manifest and scores in the same atomic staging transaction before the shared
MLflow mirror runs, and `tracking/mlflow.py` includes `report.*` among canonical mirrored artifacts
for any run that has them.

## Ukrainian bilingual calibration workflow

The source-aligned Ukrainian workflow lives in `translation.py`, `translation_review.py`,
`paired.py`, and `paired_report.py` beside the baseline implementation, with commands in
`src/llb/cli/bench/knowledge_cutoff_ua.py`. The Make entrypoints are
`knowledge-cutoff-ua-draft`, `knowledge-cutoff-ua-review`, `knowledge-cutoff-ua-revise`,
`knowledge-cutoff-ua-confirm-accepted`, `knowledge-cutoff-ua-validate`,
`knowledge-cutoff-ua-freeze`,
`bench-knowledge-cutoff-bilingual`, and the interactive umbrella target
`knowledge-cutoff-bilingual`.

Local drafting is revision-bound, per-row resumable, and syntactically separate from acceptance.
Every combined question-and-choice record must be Ukrainian-dominant and contain exactly four
unique choices; all numeric clues must match the English source exactly. The specialized shared
review card presents both language versions and assigns its four checks to factual equivalence,
fluency, answer preservation, and absence of added temporal clues.

For translation rows, aggregate `y` acceptance records all still-unchecked criteria as passing and
blocks acceptance when any criterion explicitly fails. The confirm-accepted command migrates
earlier aggregate accept decisions by recording the same implied passes without changing rejects.

Freezing requires every worksheet row to be accepted or excluded, all four checks on accepted rows
to pass, and a non-empty bilingual reviewer sign-off. It writes aligned reviewed English and
Ukrainian event files plus an immutable worksheet snapshot and review summary. The runner verifies
ids, answer keys, expected display letters, and source-choice permutations again, then evaluates
both lanes through one backend lifecycle. Canonical output under
`$DATA_DIR/knowledge-cutoff-bilingual/<run_timestamp>/` includes a combined event ledger, both
language summaries and decay fits, per-month Ukrainian-minus-English accuracy deltas, and a seeded
2,000-resample paired 95% bootstrap interval.

## Validation

The focused fake-completion suite in
`tests/llb/bench/knowledge_cutoff/test_knowledge_cutoff.py` covers validation, local and injected
Hugging Face loading, exact revision provenance, prompt balancing/date blindness, parser variants,
curve/control aggregation, seeded fitting, smoke sampling, CLI registration, and a persisted
no-network/no-GPU run.

`tests/llb/bench/knowledge_cutoff/test_bilingual_cutoff.py` covers invalid translations, resume
without duplicate model calls, specialized review guidance, undecided and incomplete-acceptance
gate failures, aligned freeze/load, choice-mapping drift, paired bootstrap persistence, and all
seven CLI registrations. The real local drafting smoke used the exact upstream revision below and a
quantized Gemma 4 12B translator; its one reviewed-format row passed the language/numeric/schema
gate at 36.8 generated tokens/s. The smoke artifact remains runtime data rather than accepted
translation evidence.

The complete local draft for that revision contains all 330 source-aligned rows under
`$DATA_DIR/knowledge-cutoff-ua/70ac8333a6fdd742f73f85a02a303aafba84617e/`. It was generated
with `google/gemma-4-12B-it-qat-w4a16-ct`; the final 167-call resume averaged 41.5 generated
tokens/s. Automated scans and a complete agent pass over the translated questions repaired 76
objective defects through `agent_revisions.jsonl`, then regenerated the worksheet and reran schema,
Ukrainian-dominance, numeric-clue, source-hash, and translation-hash validation. The worksheet now
records human aggregate acceptance of all 330 rows, with all four implied criteria persisted as
passing and no exclusions or incomplete accepts. Reviewer `vola` signed and froze the aligned
English/Ukrainian lanes for the exact upstream revision.

## Reviewed bilingual calibration result

The paired vLLM run used `google/gemma-4-12B-it-qat-w4a16-ct` with a 1,024-token context and wrote
`$DATA_DIR/knowledge-cutoff-bilingual/20260714T141514.889394Z-fcbedec92785/`. It completed 660
calls and 1,543 completion tokens at 18.7 tokens/s. Of the 330 reviewed events, 280 low/medium
predictability real-event pairs entered the cutoff curve.

- English effective cutoff: `2024-12`; eligible accuracy 0.354; parse rate 0.958.
- Ukrainian effective cutoff: `2025-01`; eligible accuracy 0.468; parse rate 0.994.
- Ukrainian-minus-English paired accuracy delta: 0.114, with a seeded 2,000-resample 95% bootstrap
  interval of [0.061, 0.168].
- Both languages scored 1.000 living-person accuracy, 1.000 fake-event rejection, and 0.000
  fake-event confabulation across 10 living-person and 8 fake-event controls.

On this reviewed event set, the model performed better in Ukrainian rather than exhibiting a
Ukrainian-comprehension penalty; the fitted effective cutoff moved one month later. The paired
interval excludes zero, but this remains a benchmark-distribution result rather than proof of a
training-data boundary or a causal language advantage. The canonical `report.json`, `report.md`,
combined event ledger, manifests, and both language summaries remain in the run directory above.

The live Hugging Face check on 2026-07-13 loaded 330 events spanning 2024-01 through 2026-06 and
resolved the dataset to commit `70ac8333a6fdd742f73f85a02a303aafba84617e`.

## Attribution boundary

The idea and dataset source were inspired by Apoorv Saxena's
[`knowledge-cutoff`](https://github.com/apoorvumang/knowledge-cutoff) project. Its Hugging Face
dataset card marks the data CC BY 4.0. No upstream application source was copied; the local
backend, Optuna, persistence, reporting, and CLI implementation follow this repository's
architecture. See the [operator guide](../../guides/benchmarking/knowledge-cutoff.md) for the full
workflow and attribution notice.
