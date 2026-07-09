# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every task line in this file must describe work that remains. Current behavior,
operator workflows, durable evidence, and design decisions live in [`current.md`](current.md) and
the topic files under [`current/`](current/). The product spec lives in
[`docs/design/spec.md`](../design/spec.md).

## Forward Tasks

The any-corpus autopipeline that turns a mixed `txt`/`md`/`pdf` directory into a validated RAG
index plus a resumable, unverified draft bundle is now shipped (`llb ingest-corpus`,
`make quickstart-corpus`, `prepare-goldset-draft --resume`; see [data prep](current/data-prep.md)).
The tasks below build the rest of the corpus-to-recommendation spine on that foundation.

The forward work is split into two sections by **who must act to complete it**:

- **[Agent Implementation Tasks](#agent-implementation-tasks)** land to `make ci` green with
  committed fixtures, injected fakes, and deterministic harnesses. A few carry a heavy real-model
  run for durable evidence; those runs are deterministic and execute on the CUDA host without any
  human judgment, so they stay in this section.
- **[Human-Assisted Tasks](#human-assisted-tasks)** cannot reach their stated acceptance without a
  human in the loop: the deliverable *is* human judgment (verification-gate reviews, drafting
  oversight, measured reviewer throughput) or requires a human authorization (egress consent +
  API spend). An agent can still build the supporting code and unit tests; the marked **human
  step** is what gates completion.

Task numbers are stable ids and never change; every task carries an explicit `Dependencies` line,
and the recommended build order within each section follows those lines. Two dependencies cross the
section boundary and are called out because they are **blocked by human work**:

- **Agent task 8 (`context-policy-bench`) is BLOCKED BY human task 7 (`chain-goldset-generation`).**
  Task 8 scores a *verified* chain fixture, and only the human review gate in task 7 can produce
  one. Task 8's code (context-assembly + fake-endpoint tests) can be written earlier, but its
  acceptance run cannot pass until task 7's human-accepted chains exist.
- **Agent task 11 (`verification-gate-adjudication`) depends on the CODE of human task 3
  (`verify-cli-throughput`).** Both extend `src/llb/goldset/verify.py` and `verify_session.py`;
  land task 3's code first to avoid a merge conflict. Task 3's *human throughput evidence* does
  **not** block task 11 -- only the shared code surface does.

For remaining tasks that depend on retrieval behavior, use the current RAG baseline documented in
[RAG core](current/rag-core.md) and the mixed-corpus ingestion baseline documented in
[data prep](current/data-prep.md).

The fine-tuning cluster (18-22) extends the spine one step past recommendation: from naming the
best base model to naming the best *adapted* model for the operator's corpus, with the whole loop
drivable by an agent end to end. Task 18 builds the single-model loop (contamination-guarded
export, injectable trainer seam, round report) and owns the invariants every later task reuses;
19 runs that loop across a multi-model roster with feasibility-aware scheduling and ranks models
by measured tunability; 20 gives adapters a registry and lifecycle so every tuned board number
stays traceable and servable; 21 adds budgeted per-model hyperparameter search that never leaves
the tuning split; 22 (optional) distills the roster's best local teacher into smaller students --
all local, no egress. Ordering inside the cluster: 18 first, then 19; 20 lands beside 19; 21 and
22 after 19.

## Agent Implementation Tasks

These land to `make ci` green with fixtures, fakes, and deterministic harnesses. The
Ukrainian-RAG-quality foundations are shipped: the measured embedder ranking
(`llb compare-embeddings`; see [RAG core](current/rag-core.md) retrieval store) that replaces the
assumed default embedder with evidence, the page/section join
(`src/llb/rag/page_metadata.py`) that links every chunk back to its origin file, and the hybrid
dense+BM25 retrieval with the chunk-metadata filter seam (see [RAG core](current/rag-core.md)
hybrid retrieval); the query-side processing lane (`--query-prep`) that builds on the measured
embedder result is shipped (see [RAG core](current/rag-core.md) query-side processing).
The external multi-service drafting lane is also shipped end to end -- both the `curate-drafts`
merge/dedup/filter step and the grounded-JSONL `import-external-draft` lane for full-document needle
realism (see [data prep](current/data-prep.md) grounded-JSONL import).
The miss analysis (`llb analyze-misses` + probe mode + the recommend misses section) is also
shipped; see [evaluation rigor](current/rigor-board-judge.md) miss-analysis section.
Recommended sequence: 11 after task 3's code, 18 anytime
(its miss-targeted export consumes the shipped miss analysis's miss clusters when an analysis
exists; the export/guard/trainer code stands
alone), 19-22 after 18 (the fine-tuning cluster reuses 18's trainer seam and contamination
guard; 20 beside 19, 21 and 22 after 19), and 8 last (blocked by human task 7). The
durable-eval-runner (retry + `cases.progress.jsonl` journal +
`--resume` + bounded backend relaunch + `manifest.durability` counters) is now shipped; see
[RAG core](current/rag-core.md) durability section.

### 8. context-policy-bench

- Dependencies: **BLOCKED BY human task 7 (`chain-goldset-generation`)** -- the acceptance run
  needs the verified `chains.jsonl` committed fixture that only task 7's human review gate can
  emit. The code (multi-hop substrate reuse, prompt-system role packages, context-assembly unit
  tests over a fake endpoint) can be built ahead of task 7; it just cannot pass its acceptance run
  until the verified chains exist.
- User-visible outcome: for one model and a verified chain set, a ranked comparison of
  context-management policies -- fresh retrieval per step, accumulated full history, summarized
  history, and staged role/system-prompt sequences (for example librarian -> analyst ->
  answerer built from prompt-system packages) -- with per-step and final-answer correctness,
  ending in a written recommendation on which harness and context policy improves scores and
  how to sequence system prompts for better answers.
- Scope boundary: in scope -- `src/llb/bench/chain_context.py` reusing the multi-hop
  retrieve/controller/answer substrate (`src/llb/eval/multi_hop.py`), prompt-system packages
  for role prompts (see [extended workflows](current/extended-workflows.md)), and the category
  persistence and board machinery; the policy is the row label and the model is fixed,
  mirroring the harness-comparison discipline; a recommendation block sourced from prompt
  templates naming the winning policy and its per-step evidence. Out of scope -- new agentic
  frameworks (settled in [product decisions](current/scope-boundaries.md)), cross-model
  blending in one board.
- Data and artifact paths: run bundles under `$DATA_DIR/chain-context/<timestamp>/`; board
  section rows beside the harness comparison; recommend summary gains a context-policy line
  when bundles exist.
- Execution path:
  `llb bench-chain-context --chains <chains.jsonl> --model <m> --backend <b>
  --policies fresh,history,summary,roles`; a make target with the standard MODEL/BACKEND
  variables; unit tests over a fake endpoint asserting the exact context assembled per policy
  per step.
- Acceptance gates: context-assembly unit tests pass for all four policies; a run over the
  committed chain fixture produces per-step and final scores with bootstrap CIs and ranks
  policies for the fixed model; provenance records policy, prompt-system ids, and chain set
  digest; verified-data stamping matches the category suite rules; `make ci` green.
- Documentation target: [extended workflows](current/extended-workflows.md);
  [`docs/guides/benchmarking/prompt-system-rag.md`](../guides/benchmarking/prompt-system-rag.md).

### 11. verification-gate-adjudication

- Dependencies: land after **task 3's `verify.py`/`verify_session.py` code** (both tasks extend the
  same modules; sequencing avoids a merge conflict). Task 3's human throughput evidence does not
  block this task. Otherwise independent -- all acceptance gates use synthetic reviewed fixtures.
- User-visible outcome: the human verification gate supports more than one annotator and richer
  acceptance rules: a stratified sample can be assigned to N reviewers, inter-annotator agreement
  (Cohen's/Fleiss' kappa) is reported, disagreements route to an adjudication pass, and acceptance
  arithmetic becomes configurable (per-stratum thresholds and confidence-weighted acceptance) rather
  than a single global tolerance. This is the "changes to the verification gate" item the shipped
  any-corpus autopipeline held out of scope (see [data prep](current/data-prep.md)), plus the
  multi-annotator / acceptance-arithmetic carve-outs of
  [`verify-cli-throughput`](#3-verify-cli-throughput).
- Scope boundary: in scope -- extend `src/llb/goldset/verify.py` (stratification, sampling,
  acceptance arithmetic, ledger emission) and `src/llb/goldset/verify_session.py` with a reviewer
  id on worksheet rows, an agreement report, an adjudication worksheet drawn from disagreements, and
  per-stratum / confidence-weighted acceptance thresholds; keep the accepted-ledger-through-adoption
  invariant (never hand-edit `verified`) and the CSV worksheet backward compatible (new optional
  columns only). Out of scope -- a web UI, changing the `GoldItem` schema, judge calibration.
  Reuse the existing `verify-sample`/`verify-review`/`verify-accept` command surface and the
  stratified sampler.
- Data and artifact paths: multi-reviewer worksheets and an `agreement.json` report beside the draft
  bundle's worksheets; an `adjudication.csv` worksheet; the accepted ledger under `accepted/` as
  today.
- Execution path: `make verify-sample BUNDLE=<draft> VERIFY_N=<n> VERIFY_ANNOTATORS=<k>`;
  `make verify-review VERIFY_WS=<per-reviewer-ws>`;
  `make verify-adjudicate BUNDLE=<draft>`;
  `make verify-accept BUNDLE=<draft> VERIFY_ACCEPT_POLICY=<per-stratum|weighted>`; unit tests for
  agreement math, adjudication draw, and each acceptance policy.
- Acceptance gates: `make ci` green; agreement statistics match a hand-computed fixture; adjudication
  draws exactly the disagreement rows and carries prior decisions forward; each acceptance policy is
  unit-tested against a synthetic reviewed sample; a reused id can still never certify changed
  content (adoption-through-ledger test preserved).
- Documentation target: [data prep](current/data-prep.md) verification gate;
  [`docs/guides/human-tooling/verification-tooling.md`](../guides/human-tooling/verification-tooling.md)
  and
  [`docs/guides/human-tooling/human-in-the-loop-evaluation.md`](../guides/human-tooling/human-in-the-loop-evaluation.md).

### morphology-aware-typo-guard (optional)

- Dependencies: the shipped query-side processing lane's `typos` step (see
  [RAG core](current/rag-core.md) query-side processing). Agent-buildable; the `[lex]` extra
  (pymorphy3) is already used index-side.
- Why this is forward work: the deterministic edit-distance `typos` step corrects any query token
  ABSENT from the corpus vocabulary to its nearest in-vocabulary token. On inflection-rich
  Ukrainian this also "corrects" grammatically-valid inflected query forms (e.g. `поділяють` ->
  `поділяти`, `документами` -> `документа`) that are not misspellings but simply a different case
  than the corpus surface form -- a crude inflection-match that can HURT retrieval on a
  non-saturated corpus (the committed fixture saturates at recall@5 1.000, so the A/B could not
  surface the regression there).
- User-visible outcome: a `typos` step that skips a token pymorphy3 recognizes as a valid
  Ukrainian word form (so genuine misspellings are still corrected, but valid inflections are left
  for the shipped index+query lemmatization to match), gated behind a flag so the pure
  edit-distance behavior stays the default when the `[lex]` extra is absent.
- Scope boundary: in scope -- an optional morphology check in `src/llb/rag/query_prep.py`
  `apply_typos` reusing `llb.rag.lexical.load_uk_lemmatizer` / a pymorphy3 `word_is_known` probe;
  an A/B row demonstrating the guard's effect on an inflection-rich non-saturated corpus. Out of
  scope -- a learned spell-corrector (the deterministic ceiling stands), changing the default
  edit-distance thresholds.
- Data and artifact paths: no new artifact; the guard is a `RunConfig` sub-knob of the `typos`
  step recorded in the manifest fingerprint.
- Execution path: `llb validate-retrieval --query-prep typos --query-prep-ab` on an
  inflection-rich goldset with and without the guard; unit tests: a valid inflected form is left
  untouched under the guard, a genuine misspelling is still corrected.
- Acceptance gates: `make ci` green; the guard leaves a pymorphy3-known valid form unchanged while
  still correcting an unknown misspelling (unit-tested); the A/B records the guarded-vs-unguarded
  retrieval delta on a non-saturated corpus.
- Documentation target: [RAG core](current/rag-core.md) query-side processing.

### citation-coverage-metric (optional)

- Dependencies: the shipped groundedness/citation metrics (`--cited-answers`; see
  [RAG core](current/rag-core.md) groundedness and citation metrics). Agent-buildable, deterministic.
- Why this is forward work: the shipped `citation_validity` collapses two very different failure
  modes into one low number -- a model that emits NO `[i]` citations and a model that cites the
  WRONG chunk both score 0.0. The durable llama3.2:3b run made this concrete: validity 0.000 with
  hallucination 0.000 because the small model simply ignored the citation instruction (mostly
  emitted no citations), so the metric could not separate "does not cite" from "cites wrongly".
- User-visible outcome: a `citation_coverage` signal -- the share of factual claims that carry ANY
  `[i]` citation -- reported beside `citation_validity`, so the board distinguishes an
  instruction-following gap (low coverage) from a grounding gap (high coverage, low validity).
- Scope boundary: in scope -- extend `src/llb/scoring/groundedness.py` with a per-claim coverage
  count (a claim with >= `MIN_CLAIM_TOKENS` content tokens is "covered" when it carries at least
  one citation) and add `citation_coverage` as an additive per-case field + manifest metric. Out of
  scope -- changing the validity/hallucination definitions, a learned claim-detector (the
  sentence-split heuristic is the deterministic ceiling), scoring non-cited runs.
- Data and artifact paths: additive `citation_coverage` in `scores.jsonl` and the manifest
  `metrics`; no bundle-shape change otherwise.
- Execution path: `llb run-eval --cited-answers`; unit tests -- a fully-cited answer scores 1.0
  coverage, an uncited answer 0.0, a partially-cited answer in between, all independent of validity.
- Acceptance gates: `make ci` green; coverage separates a no-citation answer from a wrong-citation
  answer on the synthetic fixture (the two now yield different coverage at equal validity); the
  manifest carries mean `citation_coverage` when cited-answers is on.
- Documentation target: [RAG core](current/rag-core.md) groundedness and citation metrics.

### external-rag-source-mapping (optional)

- Dependencies: none.
- User-visible outcome: external RAG answer-log scoring can audit retrieval evidence, not only
  answer text, by joining provider source records onto benchmark corpus spans. Operators supply a
  mapping sidecar from provider article ids or URLs to corpus `doc_id` plus optional character
  ranges, and the CSV/report gains source-hit, first-hit-rank, and missing-mapping columns.
- Scope boundary: in scope -- extend `llb score-external-rag` with `--source-map <json|jsonl|csv>`;
  support mappings keyed by `article_id`, `url`, or `article_title`; reuse
  `llb.rag.retrieval.first_hit_rank` once mapped records carry `doc_id`, `char_start`, and
  `char_end`; report unmapped returned sources separately from mapped retrieval misses. Out of
  scope -- crawling external article URLs, mutating the external system, or treating title-only
  fuzzy matches as proof.
- Data and artifact paths: source-map sidecars live beside the answer log or under
  `$DATA_DIR/external-rag/<system>/`; per-row mapping diagnostics stay in the CSV and report.
- Execution path: `llb score-external-rag --answers <answered-jsonl> --source-map <map.jsonl>`;
  unit tests cover id/url/title key precedence, missing mappings, and span-overlap hit ranks.
- Acceptance gates: `make ci` green; a fixture with mapped top sources reports recall@3 and MRR by
  the same source-span metric as local retrieval; title-only mappings are flagged as weak evidence
  unless spans are present.
- Documentation target: [RAG core](current/rag-core.md) external answer log scoring and
  [`docs/guides/data-prep/external-ai-service-artifacts.md`](../guides/data-prep/external-ai-service-artifacts.md).

### 18. local-model-self-improvement-loop

- Dependencies: soft-consumes the shipped miss analysis (`llb analyze-misses` emits
  `misses.jsonl`; see [evaluation rigor](current/rigor-board-judge.md)): when an analysis
  exists, the training-set export targets and weights the miss clusters; without one it falls
  back to the whole tuning split. Reuses
  the shipped split discipline (`src/llb/goldset/splits.py` -- calibration/tuning/final are
  disjoint by seeded assignment precisely so tuning can never leak into the final leaderboard
  number), the durable-eval-runner (per-round resume), and the board/recommend machinery. The
  heavy fine-tune + re-eval rounds execute seeded on the CUDA host with no human judgment --
  the same heavy-run discipline the shipped durable-evidence runs already follow.
- User-visible outcome: the benchmark closes its loop from measurement to improvement: one
  command turns a scored run into a measurably better *local* model. It exports a
  contamination-guarded training set from the tuning split (SFT records in the exact prompt
  shape the eval sends; optional preference pairs built from the model's own scored misses),
  LoRA/QLoRA fine-tunes the local model, re-evaluates the adapter as a new board row through
  the unchanged eval runner, and iterates rounds until the gain disappears -- ending with a
  per-round report (base vs tuned on the held-out final split, bootstrap CIs) and an explicit
  accept/reject verdict for the adapter. The miss analysis's evidence-backed "model X fails on
  cluster Y" becomes "model X + adapter-`<digest>` passes, with the round-by-round proof".
- Scope boundary: in scope -- `src/llb/finetune/dataset.py`: a deterministic export from a
  finalized run bundle plus its goldset -- SFT records (question + retrieved context ->
  reference answer, reusing the eval's own prompt templates so train and eval formats cannot
  drift) drawn ONLY from tuning-split items, optional DPO preference pairs (the model's scored
  wrong answer = rejected, the reference = chosen) from the miss analysis's `misses.jsonl`
  when present,
  and a `dataset_manifest.json` recording item ids, split provenance, and a content digest;
  `src/llb/finetune/trainer.py`: seeded LoRA/QLoRA behind an injectable trainer seam (real
  implementation via a new `[finetune]` optional extra -- peft/trl -- following the existing
  extras pattern; CI drives a fake trainer), emitting an adapter directory plus
  `adapter_manifest.json` (base model id, dataset digest, hyperparameters, seed, loss curve);
  the contamination guard as the non-negotiable invariant: `run-eval` refuses to score a
  model+adapter whose recorded dataset digest intersects calibration/final item ids --
  extending the split discipline from configs to weights -- and a tuned model is barred from
  judging its own answers, mirroring the recorded planter != judge guard
  (`src/llb/prep/frontier.py`); adapter serving through the existing backend seam (vLLM LoRA
  modules directly; a merge-to-GGUF lane for ollama/llama.cpp), with base model + adapter
  digest recorded in the run manifest; an orchestrator `llb self-improve` chaining
  run-eval -> analyze-misses -> export -> fine-tune -> re-eval per round, stopping on a
  CI-overlapping delta or the round budget, resumable mid-campaign; tuned board rows labeled
  `<model>+adapter-<digest>` beside the base row, and a self-improvement section in the
  `recommend` summary when rounds exist. Out of scope -- full-parameter training, RLHF/online
  RL, training or altering the judge or the embedder, frontier-API distillation (egress; the
  frontier lane belongs to human task 2), auto-adopting an adapter as the recommended default
  (the verdict is reported; adoption stays an operator decision), changing the `GoldItem`
  schema or the split assignment.
- Data and artifact paths:
  `$DATA_DIR/self-improve/<timestamp>/round-<i>/{dataset/,adapter/,run/,report.md}` with
  `dataset_manifest.json` and `adapter_manifest.json` as above; a synthetic scored-bundle +
  goldset fixture and a poisoned adapter-manifest fixture under `samples/` for the export and
  guard tests.
- Execution path: `llb export-finetune-set --run-dir <run> --goldset <gs> --out <dir>`;
  `llb finetune-adapter --dataset <dir> --model <m> --seed <s>` (heavy, CUDA host, outside
  quick CI); `llb self-improve --model <m> --backend <b> --rounds 2` and
  `make self-improve MODEL=<m> BACKEND=<b>`; unit tests cover split discipline of the export
  (it can never emit a calibration/final id), preference-pair construction from the synthetic
  miss fixture, the contamination guard's refusal on the poisoned manifest, fake-trainer loop
  wiring including the stop rule, and per-round resume.
- Acceptance gates: `make ci` green with the fake trainer; the split-leakage test proves zero
  calibration/final items in any export; the contamination guard blocks the poisoned-manifest
  fixture with a clear message naming the offending ids; one real seeded QLoRA round on the
  CUDA host over the committed goldset records base vs tuned final-split scores with bootstrap
  CIs in current docs -- reported honestly (gain, tie, or regression are all acceptable
  evidence; the mechanism must not require a gain to land); the tuned row shows no security-tier
  regression (`bench-security` delta recorded beside the correctness delta); provenance chains
  adapter -> dataset digest -> source run so every tuned board number traces to its exact
  training data.
- Documentation target: [extended workflows](current/extended-workflows.md) self-improvement
  section; a new operator guide
  [`docs/guides/benchmarking/self-improvement-loop.md`](../guides/benchmarking/self-improvement-loop.md).

### 19. finetune-campaign-multi-model

- Dependencies: follows task 18 (`local-model-self-improvement-loop`) -- the campaign reuses
  its dataset export, injectable trainer seam, contamination guard, and per-round report; land
  18's code first. Soft-consumes the shipped miss analysis (per-model miss-targeted exports when an analysis
  exists). Reuses the feasibility planner (`src/llb/backends/planner.py` -- can this model run
  on THIS host, and at what context), the VRAM reclaim gate (`src/llb/executor/vram.py` -- the
  sequential-execution contract between roster entries), and the durable-runner journal pattern
  for campaign resume. The heavy campaign executes seeded on the CUDA host, no human judgment.
- User-visible outcome: one command fine-tunes and re-evaluates a whole roster of local models
  over the same corpus goldset, answering a question the base-model leaderboard cannot: which
  model is the best pick for this corpus *after* adaptation. Sequential VRAM-safe scheduling,
  per-model rounds, and a campaign report ranking roster models by measured tunability --
  final-split gain with bootstrap CIs against training wall-clock and peak VRAM -- beside
  base-vs-adapted board rows, so `recommend` can name the best adapted model with evidence.
- Scope boundary: in scope -- `src/llb/finetune/campaign.py` orchestrating task 18's loop per
  roster entry from an explicit `--models` list (there is no single roster variable today; the
  campaign defines the roster shape and `make` passes it through); the model-independent SFT
  export computed once per campaign and shared across entries while preference pairs stay
  per-model (each model's own scored misses); feasibility-aware scheduling -- a roster entry
  the planner rejects, or whose trainer cannot fit beside the serving stack, is skipped with
  the recorded reason, never crashed into; VRAM reclaim enforced between entries exactly as
  between eval runs; a `campaign.progress.jsonl` journal with `--resume` that never re-trains
  a finished entry; adapted rows labeled per task 18 beside every base row; a tunability
  section in the `recommend` summary when a campaign report exists. Out of scope -- parallel
  or multi-GPU training (the sequential single-host contract stands), cross-model weight
  merging or model soups, changing the contamination guard (task 18 owns it), frontier or API
  teachers (egress).
- Data and artifact paths:
  `$DATA_DIR/finetune-campaign/<timestamp>/{campaign.progress.jsonl,report.md}` plus
  `<model>/round-<i>/` per entry reusing task 18's round layout; the shared SFT export stored
  once under the campaign root with its `dataset_manifest.json`.
- Execution path: `llb finetune-campaign --models <m1,m2,...> --backend <b> --rounds <n>` and
  `make finetune-campaign MODELS=<csv> BACKEND=<b>`; unit tests drive the fake trainer plus a
  fake planner -- scheduling order, skip-with-reason on an infeasible entry, journal resume
  mid-roster, shared-export reuse (byte-identical dataset digest across entries), and the
  ranking math in the report.
- Acceptance gates: `make ci` green with the fakes; resuming a half-finished campaign
  re-trains nothing already journaled (unit-tested); an infeasible roster entry appears in the
  report with its planner reason; every adapted row passes task 18's contamination guard; one
  real campaign over at least two quickstart-scale models on the CUDA host records the
  tunability ranking with CIs in current docs -- honestly (a roster where no model gains is
  acceptable evidence).
- Documentation target: [extended workflows](current/extended-workflows.md) self-improvement
  section; the self-improvement-loop guide gains the campaign chapter.

### 20. adapter-registry-lifecycle

- Dependencies: follows task 18 (registers the adapters its rounds emit and extends its
  manifests); task 19 soft-consumes the registry when present (campaign rounds auto-register).
  Land after 18, beside 19. All gates run on committed fixtures -- no heavy run.
- User-visible outcome: adapters become first-class, traceable artifacts instead of loose
  directories: a local registry lists every adapter with its base model, dataset digest,
  source run, and eval evidence; staleness is detected (the goldset or corpus changed since
  training, so the recorded evidence no longer describes the present benchmark) and stamped,
  never silently ignored; one command serves any registered adapter through the existing
  backends; and superseded adapters can be garbage-collected without ever deleting one a run
  bundle still cites. Board and `recommend` cite only registered adapters, so every tuned
  number stays reproducible.
- Scope boundary: in scope -- `src/llb/finetune/registry.py` over an append-only
  `$DATA_DIR/adapters/registry.jsonl` (id = adapter digest; entries record base model id,
  dataset digest, goldset/corpus digests, source run path, and an eval summary); automatic
  registration on a successful task 18/19 round; a staleness check comparing recorded digests
  against the current goldset/corpus with the verdict shown in `llb list-adapters`;
  `llb serve-adapter --adapter <id> --backend vllm|ollama|llamacpp` wiring the existing
  backend seam (vLLM LoRA modules directly; the merge-to-GGUF lane for ollama/llama.cpp, with
  the merge recorded as a registry event); `run-eval` resolving adapter ids through the
  registry so the contamination guard reads recorded digests, not operator-supplied ones; GC
  that refuses to delete an adapter referenced by any run bundle unless forced. Out of scope
  -- remote registries or hubs, uploading adapters anywhere (egress), automatic retraining on
  staleness (report only), a long-running serving daemon.
- Data and artifact paths: `$DATA_DIR/adapters/registry.jsonl` plus the adapter directories it
  indexes; a committed registry fixture with a stale entry and a poisoned-digest entry under
  `samples/` for the lifecycle tests.
- Execution path: `llb list-adapters`; `llb serve-adapter --adapter <id> --backend <b>`;
  `llb gc-adapters [--force]`; unit tests -- registry round-trip, staleness flip when the
  goldset digest changes, guard resolution through the registry (the poisoned entry is
  refused), GC refusal on a cited adapter, merge-event recording via a fake backend.
- Acceptance gates: `make ci` green; a stale adapter is always stamped before its row can
  render on the board; the contamination guard rejects the poisoned-digest fixture with a
  message naming the intersecting ids; GC never deletes a cited adapter without `--force`
  (unit-tested); serving smoke passes against the fake backend for all three backends.
- Documentation target: [extended workflows](current/extended-workflows.md); the
  self-improvement-loop guide's serving section.

### 21. finetune-hparam-search

- Dependencies: follows task 18 (searches over its injectable trainer seam); reuses the Optuna
  study conventions of `src/llb/optimize/tuner.py` (the `[track]` extra already pins optuna) --
  seeded study, pruned infeasible points, bounded budget -- and feeds task 19 (a recorded best
  config becomes that model's campaign default). The real bounded search runs on the CUDA host.
- User-visible outcome: fine-tuning stops guessing hyperparameters: per model, a budgeted
  Optuna search over the LoRA space (rank, alpha, learning rate, epochs, target modules) finds
  the best configuration -- scored on a held-out dev slice carved from the tuning split ONLY,
  so neither calibration nor final ever influences the search -- and records it beside the
  adapter artifacts for tasks 18/19 to consume as defaults.
- Scope boundary: in scope -- `src/llb/finetune/hparam_search.py` reusing the tuner's study
  conventions; a seeded dev-slice split *within* the tuning split (train sub-slice vs dev
  sub-slice, disjointness unit-tested), extending the recorded split discipline from
  configuration knobs to hyperparameters; `--max-trials` / `--max-hours` budget guards that
  abort cleanly with the study resumable; an `hparams_manifest.json` per model (best config,
  study seed, trial table, objective values) consumed by the task 18 trainer as defaults;
  fake-trainer trials in CI with a synthetic objective. Out of scope -- searching on the final
  or calibration split (forbidden by the guard), full-parameter tuning, a second tuner
  framework, per-trial human review.
- Data and artifact paths: `$DATA_DIR/finetune-hparams/<model>/<timestamp>/` with the study
  journal and `hparams_manifest.json`; the manifest path recorded in `adapter_manifest.json`
  when a searched config trained the adapter.
- Execution path: `llb finetune-hparams --model <m> --dataset <dir> --max-trials 8` and
  `make finetune-hparams MODEL=<m>`; unit tests -- dev-slice disjointness (no calibration or
  final id can enter a trial), deterministic fake-trainer study given the seed, budget abort
  plus resume, and manifest consumption by the trainer.
- Acceptance gates: `make ci` green with the fake trainer; the dev-slice test proves zero
  calibration/final leakage into any trial; an aborted study resumes without repeating
  finished trials; one real bounded search (at most 8 trials) on the CUDA host records a
  per-model best config and its dev-slice objective in current docs; a task 18 round trained
  with the manifest config reproduces the recorded configuration in its provenance.
- Documentation target: [extended workflows](current/extended-workflows.md); the
  self-improvement-loop guide's tuning appendix.

### 22. local-distillation-lane (optional)

- Dependencies: follows task 18 (trainer seam and contamination guard; registry integration
  through task 20 when present) and soft-follows task 19 (the campaign report names the natural
  teacher -- the roster's best adapted model). Local-only, so no egress question arises. The
  heavy distillation run executes on the CUDA host.
- User-visible outcome: the roster's strongest local model teaches the smaller ones: the
  teacher answers tuning-split questions with retrieved context, its answers are quality-gated
  deterministically against the reference answers (only an answer scoring at or above the gate
  becomes a training target -- teacher misses are dropped, never invented into data), and the
  student is fine-tuned on the accepted set through the task 18 trainer. The report compares
  student-distilled against student-SFT-on-references over the same items, so distillation must
  demonstrate its value, not assume it.
- Scope boundary: in scope -- `src/llb/finetune/distill.py`: teacher generation through the
  existing backend seam over tuning-split items only; the deterministic quality gate reusing
  the existing correctness scorers; identity guards -- teacher != student, and the
  calibration-gated judge is never the teacher (extending the recorded planter != judge rule
  in `src/llb/prep/frontier.py`); distilled adapters flow through the same contamination guard
  and (when task 20 exists) registry as every other adapter; the paired
  distilled-vs-reference-SFT comparison in the report. Out of scope -- frontier or API
  teachers (egress; human task 2's lane is drafting-only), logit or soft-label distillation
  across tokenizers (text-level SFT only), training or improving the teacher itself, chain or
  agentic trace distillation (task 8 owns chain evaluation).
- Data and artifact paths: `$DATA_DIR/distill/<timestamp>/` holding `teacher_outputs.jsonl`,
  the accepted `dataset/`, the student `adapter/`, and `report.md`; the dataset manifest
  records the teacher id, gate threshold, and per-item gate scores.
- Execution path: `llb distill --teacher <m1> --student <m2> --backend <b> --gate <t>` and
  `make distill TEACHER=<m1> STUDENT=<m2> BACKEND=<b>`; unit tests with a fake teacher
  endpoint and the fake trainer -- gate exclusion of below-threshold answers, identity-guard
  refusals, tuning-split discipline, and the paired-comparison report math.
- Acceptance gates: `make ci` green with the fakes; a below-gate teacher answer can never
  reach the training set (unit-tested); teacher == student and judge == teacher both refuse
  with clear messages; one real CUDA run distills the campaign's best teacher into one smaller
  student and records the distilled-vs-reference-SFT delta with CIs honestly (no-gain is
  acceptable evidence); the distilled adapter passes the contamination guard and registers
  like any other.
- Documentation target: [extended workflows](current/extended-workflows.md); the
  self-improvement-loop guide.

### embedding-bakeoff-full-corpus

- Dependencies: the shipped `llb compare-embeddings` bake-off (see
  [RAG core](current/rag-core.md) Embedder Conventions And Bake-off). Heavy store builds run on the
  CUDA host (deterministic, no human judgment).
- Why this is forward work: the committed durable evidence ranks the four local candidates on the
  tiny `samples/goldsets/ip_regulation_uk` fixture (8 items / 10 chunks), where recall@10 SATURATES
  (e5-base, e5-large, and bge-m3 all hit 1.000) so the winner is decided only by an MRR + throughput
  tie-break, and the reported `chunks/s` is cold-load-dominated rather than steady-state. The
  recommendation "e5-base for the 16 GB host" is therefore under-discriminated.
- User-visible outcome: a bake-off report over a REAL full Ukrainian corpus (e.g. the quickstart PDF
  corpus index) at a larger `k`, where recall@k actually separates the candidates and embed
  throughput reflects steady state, yielding a confidently-ranked embedder recommendation the
  operator can trust before pinning `RunConfig.embedding_model`.
- Scope boundary: in scope -- run `make compare-embeddings` on a full-corpus goldset, record the
  ranked table + winner in [RAG core](current/rag-core.md) and
  [platform matrix](current/platform-vector-matrix.md), and note the per-candidate index size / VRAM
  fit on the 16 GB host. Out of scope -- new bake-off code (the command is shipped), an API row
  unless the corpus is explicitly open, changing the pinned drafting-side E5 seams.
- Data and artifact paths: `$DATA_DIR/compare-embeddings/<timestamp>/report.md` plus the per-model
  stores; the durable table lands in current docs.
- Execution path: `make compare-embeddings GOLDSET=<full-corpus goldset> RAG_K=20` on the CUDA host
  (outside quick CI); then `make build-index EMBEDDING_MODEL=<winner>`.
- Acceptance gates: the report ranks all four local candidates with a NON-saturated recall@k spread
  on the full corpus; current docs record the discriminated winner and its index size / device fit.
- Documentation target: [RAG core](current/rag-core.md) Embedder Conventions And Bake-off;
  [platform matrix](current/platform-vector-matrix.md).

### chunking-comparison-full-corpus (optional)

- Dependencies: the shipped chunking-strategy comparison (`compare-retrieval --strategies` /
  `make compare-retrieval CHUNK_STRATEGIES=...`; see [RAG core](current/rag-core.md) chunking
  strategies). Heavy store builds run on the CUDA host (deterministic, no human judgment).
- Why this is forward work: the committed durable evidence ranks the eight strategies on the tiny
  `samples/goldsets/ip_regulation_uk` fixture, where recall@10 SATURATES at 1.000 for every
  strategy and even k=3 separates only `late` (0.875/0.750) from a seven-way 1.000/1.000 tie; the
  single-`.md` corpus also has no PDF sidecars, so the `page` strategy degenerates to `recursive`
  and its page-alignment value is never exercised on real data.
- User-visible outcome: a chunker ranking over a REAL full Ukrainian PDF corpus (e.g. the
  quickstart PDF corpus, whose `*.citations.json` sidecars make `page` meaningful) at a k where
  recall separates strategies, yielding a demonstrated per-corpus chunker recommendation before an
  operator pins `RunConfig.strategy`.
- Scope boundary: in scope -- run the shipped comparison over a full-corpus goldset, record the
  ranked table + winner in [RAG core](current/rag-core.md), and note `late`'s extra embed
  wall-clock beside its quality delta. Out of scope -- new comparison code (the command is
  shipped), new strategies, changing the source-span metric.
- Data and artifact paths: per-strategy stores under `$DATA_DIR/llb/rag/<strategy>/`; the ranked
  table lands in current docs.
- Execution path: `make compare-retrieval GOLDSET=<full-corpus goldset> RAG_K=10
  CHUNK_STRATEGIES=page,heading,late,markdown,semantic,recursive` on the CUDA host (outside
  quick CI); then `make build-index CHUNK_STRATEGY=<winner>`.
- Acceptance gates: the report shows a NON-saturated recall@k spread over a sidecar-bearing
  corpus; current docs record the discriminated winner and the measured `page`-vs-`recursive` and
  `late`-vs-`sentence` deltas.
- Documentation target: [RAG core](current/rag-core.md) chunking strategies.

### hybrid-comparison-full-corpus (optional)

- Dependencies: the shipped hybrid retrieval comparison (`compare-retrieval --hybrid` /
  `make compare-retrieval HYBRID=1`; see [RAG core](current/rag-core.md) hybrid retrieval).
  Heavy store builds run on the CUDA host (deterministic, no human judgment).
- Why this is forward work: the committed durable evidence ranks dense vs hybrid on two tiny
  single-document fixtures where three signals stay UNDER-MEASURED: recall@10 saturates at 1.000
  (only MRR separates the rows), the lemmatization on/off delta is zero because the exact-term
  queries are built around non-inflecting numbers/codes, and the `dense+oracle-doc` router
  headroom row degenerates to the dense row (a document filter is a no-op on a one-document
  corpus).
- User-visible outcome: a dense-vs-hybrid(-vs-lemmas) ranking plus a MEANINGFUL router-headroom
  number over a real multi-document Ukrainian corpus (e.g. the quickstart PDF corpus) with
  inflection-rich queries, yielding a per-corpus verdict on the fusion default
  (`fusion_weight`, lemmas on/off) an operator can trust before pinning
  `RunConfig.retrieval_mode=hybrid`.
- Scope boundary: in scope -- run the shipped comparison over a full-corpus goldset at a k where
  recall separates the rows, record the ranked table + fusion-knob verdict in
  [RAG core](current/rag-core.md), and grid `fusion_weight` in one sweep to cross-check the
  compare-retrieval verdict against end-to-end scores. Out of scope -- new comparison code (the
  command is shipped), the shipped query-side processing lane (`--query-prep`), a learned document
  router (the oracle row only measures headroom).
- Data and artifact paths: the hybrid store under `$DATA_DIR/llb/rag/hybrid/`; the ranked table
  lands in current docs.
- Execution path: `make compare-retrieval GOLDSET=<full-corpus goldset> RAG_K=10 HYBRID=1` on the
  CUDA host (outside quick CI); then `make build-index RETRIEVAL_MODE=hybrid [LEMMATIZE=1]` and
  `make sweep SWEEP_RAG_GRID="top_k=3,5;fusion_weight=0.4,0.6"`.
- Acceptance gates: the report shows a non-saturated dense-vs-hybrid spread, a non-degenerate
  oracle-doc headroom row (multi-document corpus), and a measured lemmatization delta (positive,
  zero, or negative -- reported honestly); current docs record the fusion-knob verdict.
- Documentation target: [RAG core](current/rag-core.md) hybrid retrieval.

### rerank-order-full-cohort (optional)

- Dependencies: the shipped rerank + context-order stage (`compare-retrieval --reranker`,
  `probe-context-position`; see [RAG core](current/rag-core.md) reranking and context order).
  Heavy runs execute on the CUDA host (deterministic, no human judgment).
- Why this is forward work: the committed rerank evidence lives on the two tiny fixtures where
  recall@10 saturates at 1.000 (only MRR discriminates -- the exact-term fixture shows the big
  cross-encoder win, dense MRR 0.713 -> 1.000, but recall headroom is invisible), and the
  committed position-probe run (llama3.2:3b, n=20) ends with OVERLAPPING head/tail CIs, so no
  model has a resolved ordering verdict yet.
- User-visible outcome: a rerank on/off verdict at a k where recall separates (does the
  cross-encoder recover the real-corpus recall@10=0.729 shortfall dense-only shows on the
  quickstart PDF index?) plus a resolved per-model `context_order` recommendation for each
  roster model at an n where the CIs separate -- or the honest verdict that the model is not
  position-sensitive.
- Scope boundary: in scope -- run `make compare-retrieval RERANKER=... [HYBRID=1]` over a
  full-corpus goldset, grid `rerank_candidates=0,30` in one sweep to cross-check retrieval
  uplift against end-to-end scores, and run `make probe-context-position` per roster model at
  full-split n; record verdicts in current docs. Out of scope -- new probe/rerank code (the
  commands are shipped), API rerankers (egress policy).
- Data and artifact paths: probe reports under `$DATA_DIR/context-position/<timestamp>/`; the
  ranked rerank rows land in current docs.
- Execution path: `make compare-retrieval GOLDSET=<full-corpus goldset> RAG_K=10 HYBRID=1
  RERANKER=BAAI/bge-reranker-v2-m3`; `make sweep SWEEP_RAG_GRID="rerank_candidates=0,30"`;
  `make probe-context-position MODEL=<m> BACKEND=<b> PROBE_K=5` (no LIMIT cap) -- all on the
  CUDA host, outside quick CI.
- Acceptance gates: the rerank rows report a non-saturated pre/post-rerank recall@k spread plus
  steady-state latency on the full corpus; each probed model gets either non-overlapping
  head/tail CIs or an explicit not-position-sensitive verdict; current docs record both.
- Documentation target: [RAG core](current/rag-core.md) reranking and context order;
  [evaluation rigor](current/rigor-board-judge.md) context-position probe.

### external-import-needle-parity (optional)

- Dependencies: the shipped grounded-JSONL import (see [data prep](current/data-prep.md)
  grounded-JSONL import) and the shipped `prepare-goldset-draft --retrieval-index-dir` needle-rank
  annotation. Agent-buildable with the committed `samples/external-drafts` fixture; no network.
- Why this is forward work: `import-external-draft` records each item's `question_type`/`difficulty`
  in `item_provenance.jsonl`, but -- unlike the local ontology lane -- it does NOT annotate imported
  items with `retrieval_rank` against a full-corpus index, so a reviewer verifying an externally
  imported bundle loses the confidence-ordering + retrieval-uniqueness signal local drafts carry.
- User-visible outcome: an optional `--retrieval-index-dir`/`--retrieval-k` on
  `import-external-draft` (and `--drop-nonretrievable-needles`) that annotates each imported item
  with its gold-span retrieval rank, so external and local drafts reach the verify gate with the
  same per-item signal.
- Scope boundary: in scope -- reuse the shipped needle-rank annotator
  (`src/llb/prep/ontology/needles.py`) over the imported items and record the rank in item
  provenance; a verify-worksheet column when present. Out of scope -- changing the `GoldItem`
  schema, building the index (the operator points at an existing one).
- Data and artifact paths: `retrieval_rank` added to `item_provenance.jsonl`; no bundle-shape
  change otherwise.
- Execution path: `make import-external-draft ARTIFACT= CORPUS= SIDECAR= RETRIEVAL_INDEX_DIR=`;
  unit tests over the committed fixture + a fake retriever.
- Acceptance gates: `make ci` green; imported items carry `retrieval_rank` when an index is given
  and the lane is an exact no-op when it is not; a non-retrievable item is dropped only under the
  explicit flag.
- Documentation target: [data prep](current/data-prep.md) grounded-JSONL import.

## Human-Assisted Tasks

Each task's code and unit tests are agent-buildable; the marked **human step** is what gates
completion. Tasks 3 and 7 also gate agent work: land task 3's code before agent task 11, and
finish task 7's human review before agent task 8's acceptance run.

### 1. draft-yield-quality-max -- residual empirical acceptance

- Dependencies: none (uses the shipped drafting knobs). Human step: the acceptance evidence below
  needs a local drafter model and a human reviewer and cannot run in CI; the optional multi-hop
  hardening is agent-buildable and unit-tested.
- Context: coverage-target sampling (`--coverage-target`), 2-hop graph-path multi-hop drafting
  (`--multi-hop`), pinned-E5 prior-bundle near-duplicate suppression (`--dedup-against`), and the
  closed question-type + difficulty labels are implementable and unit-covered (module map, report
  fields, and command reference in
  [robust backends and ontology drafting](current/robustness-ontology-backends.md) and
  [data prep](current/data-prep.md)). What remains is the heavy manual acceptance evidence and one
  optional quality hardening, both needing a local drafter model and human review (out of CI):
- Acceptance evidence (human to-do): on the local quickstart PDF corpus, draft once with
  `DRAFT_COVERAGE_TARGET=<n>` and once with the 180-cap default over the same corpus/model, run a
  `make verify-sample VERIFY_N=40` review of each, and record in [data prep](current/data-prep.md)
  whether the coverage-target bundle keeps more citation-valid needles at an equal-or-better accept
  rate, with the retrieval-unique needle fraction per question type. This is the "keeps more needles
  at equal-or-better accept rate" gate that cannot run in CI.
- Optional quality hardening (agent-buildable): multi-hop items ground the two hop-evidence spans
  but leave the reference answer free-text. Require the multi-hop reference answer to be (or
  contain) the verbatim bridge/end-entity span so the answer itself is span-checkable, and extend
  the multi-hop unit tests to assert it -- a free-text answer can drift from the chain even when the
  evidence spans hold.
- Documentation target: [data prep](current/data-prep.md) and
  [robust backends and ontology drafting](current/robustness-ontology-backends.md).

### 2. frontier-ua-draft-lane

- Dependencies: none (code reuses `src/llb/prep/frontier.py`). Human step: the real-frontier
  2-document probe requires **human egress consent and API spend** under the recorded egress policy;
  the code and all fake-completer tests are agent-buildable without any network call.
- User-visible outcome: for the best Ukrainian question quality and completeness, an operator
  can opt a draft run into a best-of-breed external API (litellm-routed) for extraction,
  drafting, or both, with an explicit consent gate, a hard budget cap, per-call cost telemetry
  in provenance, and a side-by-side local-vs-frontier yield and quality report over the same
  seeds. Frontier cross-check exists (`make cross-check-goldset CROSS_CHECK_MODEL=`); the draft
  lane does not.
- Scope boundary: in scope -- a frontier endpoint option for the ontology pipeline reusing the
  litellm conventions in `src/llb/prep/frontier.py` behind the same endpoint seam as Ollama and
  vLLM drafting (`src/llb/prep/ontology/endpoint.py`); `--max-usd` and `--max-calls` guards
  that abort cleanly and record the reason; an interactive egress consent prompt naming the
  corpus and destination (policy stays as recorded in
  [product decisions](current/scope-boundaries.md)); a `llb draft-compare` command that drafts
  the same bounded seed subset locally and via frontier and reports kept-yield, gate results,
  and verify-sample accept rate. Out of scope -- making frontier the default, egress for
  scoring or judging, retries of the egress policy discussion.
- Data and artifact paths: `provenance.json` gains `endpoint.cost_usd`, call counts, and
  latency; comparison reports under `$DATA_DIR/draft-compare/<timestamp>/`.
- Execution path:
  `make prepare-goldset-draft DRAFT_ENDPOINT=frontier DRAFT_FRONTIER_MODEL=<litellm-id>
  DRAFT_MAX_USD=<n>`; `llb draft-compare --corpus-root <dir> --seeds <n> --frontier-model <id>
  --local-model <model>`; unit tests use an injected fake litellm completer.
- Acceptance gates: no network call happens without the flag plus consent (unit-tested via the
  injected completer); the budget guard aborts mid-draft and the bundle remains inspectable
  with the abort recorded; a 2-document probe against a real frontier model passes bundle gates
  with parse rate at least matching the local drafter; the comparison report ranks both lanes
  on kept-yield and accept rate.
- Documentation target: [data prep](current/data-prep.md) frontier lane notes;
  [`docs/guides/data-prep/goldset-from-scratch.md`](../guides/data-prep/goldset-from-scratch.md).

### 3. verify-cli-throughput

- Dependencies: none. **Blocks agent task 11** (that task extends the same
  `verify.py`/`verify_session.py` surface -- land this task's code first). Human step: the
  "materially more items per hour" outcome needs a recorded human 40-item review pass with measured
  throughput; the CLI code and unit tests are agent-buildable.
- User-visible outcome: a human reviewer clears materially more items per hour in the terminal
  review session: a confidence-ordered queue (cross-check verdict and `retrieval_rank` decide
  order), on-card evidence with PDF page citations, accept-with-edit that re-grounds an edited
  answer span immediately, additive sample enlargement that never re-shows decided rows,
  session stats with an items-per-hour ETA, and coded rejection reasons exported for draft
  feedback.
- Scope boundary: in scope -- extend `src/llb/goldset/verify.py` and
  `src/llb/goldset/verify_session.py`; keep the worksheet CSV shape backward compatible (new
  optional columns only); a `verify-sample` merge mode that draws additional stratified rows
  while carrying prior decisions forward; a rejection-reason summary artifact the drafting
  pipeline can read to tighten prompts. Out of scope -- a web UI, multi-annotator merging,
  changes to acceptance arithmetic.
- Data and artifact paths: worksheets under the draft bundle as today; a
  `rejection_reasons.json` summary beside the accepted ledger.
- Execution path: `make verify-sample BUNDLE=<draft> VERIFY_N=<n> VERIFY_MERGE=1`;
  `make verify-review VERIFY_WS=<ws> VERIFY_ORDER=confidence`; unit tests for merge
  idempotence, ordering, and re-grounding of edited spans.
- Acceptance gates: `make ci` green including session golden-path tests; merging a larger
  sample adds only new rows and preserves every decided row byte-for-byte; an edited answer
  that no longer matches its span is blocked until re-grounded; manual evidence -- one recorded
  40-item review pass on the quickstart draft with the measured items-per-hour noted in the
  current docs.
- Documentation target: [data prep](current/data-prep.md) verification gate;
  [`docs/guides/human-tooling/human-in-the-loop-evaluation.md`](../guides/human-tooling/human-in-the-loop-evaluation.md)
  and [`docs/guides/human-tooling/verification-tooling.md`](../guides/human-tooling/verification-tooling.md).

### 5. security-corpus-probes

- Dependencies: the shipped ontology artifacts (`ontology.json`, `extraction.jsonl`) from a
  local-drafter bundle. Human step: derived security cases must clear the human
  `verify-sample`/`verify-review`/`verify-accept` gate before any headline/composite use; the
  generator and unit tests are agent-buildable.
- User-visible outcome: the security tier gains corpus-specific cases derived from the target
  corpus itself: prohibited-topic denial-guard probes built from the corpus ontology's
  sensitive topics and entities, benign near-boundary controls that catch over-refusal on
  legitimate corpus questions, and matched-pair bias probes (entity or group swapped, behavior
  fixed) scored for decision consistency -- all flowing through the existing detectors,
  cross-language grouping, and the human verification gate before any headline use.
- Scope boundary: in scope -- `llb derive-security-cases --bundle <draft-bundle>` reading
  `ontology.json` and `extraction.jsonl`, generating cases locally through the same drafter
  endpoint seam; emitted cases reuse the committed case schema (`lang`, `attrs.vector`,
  `xlang_group`) so `bench-security`, `cross_language_consistency`, and refusal-appropriateness
  work unchanged (see [category suite](current/category-benchmark-suite.md) and the Ukrainian
  security adaptation in [evaluation rigor](current/rigor-board-judge.md)); a bias-pair
  consistency metric alongside ASR reusing the matched-group machinery. Out of scope -- new
  detector kinds, safety classifiers, ranking unverified case sets.
- Data and artifact paths: derived cases under `$DATA_DIR/security-derive/<timestamp>/cases.json`
  with per-case grounding spans back to the corpus; a small derived-and-verified sample
  committed under `samples/` for regression.
- Execution path: `llb derive-security-cases --bundle <draft> --out <cases.json>`;
  `make bench-security SECURITY_CASES=<cases.json> MODEL=<m> BACKEND=<b>`; verification through
  the existing `verify-sample`/`verify-review`/`verify-accept` path.
- Acceptance gates: every generated probe cites a corpus topic or entity with an exact span
  (unit-tested); benign controls feed refusal-appropriateness only, never ASR; a full
  `bench-security` run over derived quickstart-corpus cases reports per-family ASR plus
  bias-pair consistency with bootstrap CIs; unverified derived sets are rejected from
  composite/headline paths.
- Documentation target: [category suite](current/category-benchmark-suite.md) security section;
  [`docs/guides/learning-path/learning-path-security.md`](../guides/learning-path/learning-path-security.md).

### 7. chain-goldset-generation

- Dependencies: none (reuses the shipped graph-path walker `src/llb/prep/ontology/graph_paths.py`).
  **Blocks agent task 8** (`context-policy-bench` scores this task's verified chain fixture). Human
  step: >= 10 chains must be human-reviewed and accepted into the committed fixture; the schema,
  drafting, and validation code are agent-buildable.
- User-visible outcome: the draft pipeline also emits chain-of-questions test sets: ordered 2-4
  step sequences in which each step supplies more specific context for the topic (topic
  overview -> narrowing detail -> exact fact), every step carrying its own reference answer and
  exact source spans, validated and human-verified with the same discipline as flat items.
- Scope boundary: in scope -- a `ChainItem` schema in `src/llb/goldset/chains.py` whose steps
  embed `GoldItem`-compatible question/answer/span fields plus a chain id, step order, and a
  dependency note describing what the previous step establishes; chain drafting seeded from
  knowledge-graph paths and heading hierarchies (reuses the shipped graph-path walker
  `src/llb/prep/ontology/graph_paths.py`);
  span-exact validation via an extended `validate-goldset`; verify-session support rendering a
  chain as one card with per-step checks. Out of scope -- the scoring runner
  (`context-policy-bench`), multi-annotator flows.
- Data and artifact paths: draft bundles gain `chains.jsonl`; a committed fixture
  `samples/goldsets/<name>/chains.jsonl` with verified chains for smoke tests.
- Execution path: `make prepare-goldset-draft DRAFT_CHAINS=1`;
  `llb validate-goldset --chains <chains.jsonl> --corpus <dir>`; chains flow through
  `verify-sample`/`verify-review`/`verify-accept` unchanged at the command level.
- Acceptance gates: every step of every chain passes span-exact validation; steps within a
  chain reference distinct spans and the final step's answer is not answerable from step-1
  context alone (checked by the retrieval-uniqueness filter applied per step); at least 10
  chains reviewed and accepted into the committed fixture; `make ci` green.
- Documentation target: [data prep](current/data-prep.md).

## Adding Future Tasks

Add a task only when there is concrete forward work with enough detail for an engineer or an
agent to execute without guessing. Use a stable descriptive id such as `platform-matrix-power`
or `prompt-system-tuning`; keep the id only while work remains under it. Place it under
**Agent Implementation Tasks** if it can land to `make ci` green with fixtures/fakes (heavy
deterministic runs on the CUDA host are fine), or under **Human-Assisted Tasks** if a human
review/judgment or authorization gates completion; either way give it a `Dependencies` line and
mark any cross-section block explicitly.

Each task entry must include:

- Dependencies: prerequisite tasks (by number/id), any cross-section block, and -- for
  human-assisted tasks -- the specific human step that gates completion.
- User-visible outcome: what new capability or decision the work should create.
- Scope boundary: what is in scope, what is explicitly out of scope, and which existing modules or
  commands should be reused.
- Data and artifact paths: expected corpus, gold set, config, `$DATA_DIR/<method>/<run>/` outputs,
  and any committed `samples/` outputs.
- Execution path: commands, manual run steps, required local services, and any heavy/dependent steps
  that must stay outside quick CI.
- Acceptance gates: tests, lint/type checks, retrieval thresholds, score comparison method, or manual
  evidence required before the item leaves this file.
- Documentation target: the narrow `docs/impl/current/*.md` topic and any guide that should receive
  the resulting behavior and run notes.

When a task surfaces new future work, add that as a new forward task. Put current behavior and
durable decisions in current docs, never in this plan.
