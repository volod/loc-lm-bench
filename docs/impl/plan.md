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
and the recommended build order within each section follows those lines. One dependency crosses the
section boundary and is called out because it is **blocked by human work**:

- **Agent task 8 (`context-policy-bench`) is BLOCKED BY human task 7 (`chain-goldset-generation`).**
  Task 8 scores a *verified* chain fixture, and only the human review gate in task 7 can produce
  one. Task 8's code (context-assembly + fake-endpoint tests) can be written earlier, but its
  acceptance run cannot pass until task 7's human-accepted chains exist.

For remaining tasks that depend on retrieval behavior, use the current RAG baseline documented in
[RAG core](current/rag-core.md) and the mixed-corpus ingestion baseline documented in
[data prep](current/data-prep.md).

The remaining fine-tuning cluster (23) extends the spine one step past recommendation: from
naming the best base model to naming the best *adapted* model for the operator's corpus, with the
single-model self-improvement loop, the multi-model campaign substrate, the adapter registry, and the
budgeted LoRA hyperparameter search as reusable bases (see
[extended workflows](current/extended-workflows.md)). Task 22 (local distillation) is shipped
(`llb distill`; see [extended workflows](current/extended-workflows.md)). Task 23 (optional) adds
native support for compressed QAT checkpoints whose linear layers need adapter injection beyond
ordinary PEFT LoRA defaults -- all local, no egress; it follows the baseline trainer path. The
`adapter-*` and `finetune-hparams-*` tasks above harden the shipped registry, merge lane, and search,
and are independent of the cluster.

Every task below carries an explicit `Agent status` line with one of four markers:

- **CLEAR** -- agent-buildable to `make ci` green with fixtures/fakes; no run evidence, no human
  gate.
- **RUN NEEDED** -- agent-buildable, but acceptance requires a heavy deterministic run; every dev
  box is a proper CUDA host, so the agent executes these runs itself on the current machine.
- **BLOCKED BY HUMAN** -- the acceptance gate consumes an artifact only a human step can produce.
- **HUMAN-GATED** -- the deliverable itself is human judgment or authorization; supporting code and
  unit tests are agent-buildable.

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
Recommended agent sequence (optional tasks included; human-gated work last):

1. **CLEAR (fixtures only), in order**: 11 (`verification-gate-adjudication`);
   `adapter-citation-scan-orchestrator-journals`; `finetune-hparams-stratified-dev-slice`; and the
   optional `citation-coverage-metric`, `external-rag-source-mapping`,
   `adapter-staleness-retrieval-fingerprint`, `finetune-hparams-infeasible-point-prune`,
   `verify-sample-exact-allocation`, `draft-feedback-rejection-reasons`, and
   `external-import-needle-parity` in any order.
2. **RUN NEEDED (agent-executed on the current CUDA host)**: `embedding-bakeoff-full-corpus` first
   (its winner feeds every later store build); then the optional
   `chunking-comparison-full-corpus`, `hybrid-comparison-full-corpus`, and
   `rerank-order-full-cohort` in that order; then the optional `morphology-aware-typo-guard` (its
   A/B needs the non-saturated full-corpus goldset the runs above establish),
   `adapter-merge-serving-cuda-evidence`, `finetune-hparams-effective-batch-axis`, and 23
   (`compressed-qat-adapter-support`).
3. **Human-gated tail**: task 8's code (context assembly + fake-endpoint tests) can be pre-built at
   any point, but its acceptance run stays last -- it is blocked by human task 7's verified chain
   fixture.

The durable-eval-runner (retry + `cases.progress.jsonl` journal +
`--resume` + bounded backend relaunch + `manifest.durability` counters) is now shipped; see
[RAG core](current/rag-core.md) durability section.

### 8. context-policy-bench

- Agent status: **BLOCKED BY HUMAN task 7** -- the code is agent-buildable now, but the acceptance
  run consumes the verified `chains.jsonl` fixture only task 7's human review gate can emit.
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

- Agent status: **CLEAR** -- fixtures only; no human step gates this task.
- Dependencies: none -- the `verify.py`/`verify_session.py` review-CLI surface this task extends is
  shipped (see [data prep](current/data-prep.md) reviewer throughput tooling); all acceptance gates
  use synthetic reviewed fixtures.
- User-visible outcome: the human verification gate supports more than one annotator and richer
  acceptance rules: a stratified sample can be assigned to N reviewers, inter-annotator agreement
  (Cohen's/Fleiss' kappa) is reported, disagreements route to an adjudication pass, and acceptance
  arithmetic becomes configurable (per-stratum thresholds and confidence-weighted acceptance) rather
  than a single global tolerance. This is the "changes to the verification gate" item the shipped
  any-corpus autopipeline held out of scope (see [data prep](current/data-prep.md)), plus the
  multi-annotator / acceptance-arithmetic carve-outs the shipped review CLI deliberately left out
  (see [data prep](current/data-prep.md) reviewer throughput tooling).
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

- Agent status: **RUN NEEDED** -- code and unit tests are CLEAR; the A/B acceptance row needs a run
  over an inflection-rich non-saturated goldset on the current CUDA host. No human gate.
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

- Agent status: **CLEAR** -- deterministic, fixtures only; no run evidence, no human gate.
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

- Agent status: **CLEAR** -- fixtures only; no run evidence, no human gate.
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

### adapter-merge-serving-cuda-evidence (optional)

- Agent status: **RUN NEEDED** -- the whole deliverable is a heavy deterministic merge + serve +
  `run-eval` comparison, agent-executed on the current CUDA host. No human gate.
- Dependencies: the shipped adapter registry, merge lane, and `serve-adapter` (see
  [extended workflows](current/extended-workflows.md) adapter registry and lifecycle). The heavy
  merge + serve executes on the CUDA host (deterministic, no human judgment).
- Why this is forward work: the merge-to-GGUF lane (PEFT `merge_and_unload` ->
  `convert_hf_to_gguf.py` -> `ollama create`) is only exercised against an INJECTED fake merge in
  CI, so the real path has never run: the converter's architecture coverage, the merged model's
  tokenizer round-trip, and whether a merged adapter actually answers as the ADAPTER (and not as
  the base model) are all unverified. A merge that silently produced base-model behavior would be
  invisible to every current test.
- User-visible outcome: a recorded CUDA-host run merging one registered adapter and serving it on
  BOTH `ollama` and `llamacpp`, with a `run-eval` comparison proving the merged artifact scores like
  the vLLM LoRA row (within overlapping CIs) rather than like the base model.
- Scope boundary: in scope -- run the shipped `llb serve-adapter --backend ollama|llamacpp` against
  a real adapter; record the merge wall-clock, GGUF size, and the merged-vs-LoRA-vs-base objective
  triple in current docs; note any architecture the converter rejects. Out of scope -- new merge
  code (the lane is shipped), quantized GGUF outtypes beyond the pinned `f16`, uploading merged
  artifacts anywhere (egress).
- Data and artifact paths: `$DATA_DIR/adapters/merged/<short-id>/<backend>/`; the comparison table
  lands in current docs.
- Execution path: `llb serve-adapter --adapter <id> --backend ollama --smoke`, then
  `llb run-eval --model <merged-tag> --backend ollama` on the CUDA host (outside quick CI).
- Acceptance gates: both GGUF backends serve the merged adapter and answer the probe; the merged
  row's final-split objective matches the vLLM LoRA row within overlapping CIs and differs from the
  base row -- or the divergence is recorded honestly as a merge-fidelity finding; current docs record
  the merge cost and the three-way objective comparison.
- Documentation target: [extended workflows](current/extended-workflows.md) adapter registry and
  lifecycle; the self-improvement-loop guide's serving section.

### adapter-citation-scan-orchestrator-journals

- Agent status: **CLEAR** -- fixtures only; no run evidence, no human gate.
- Dependencies: the shipped GC citation scan (see
  [extended workflows](current/extended-workflows.md) adapter registry and lifecycle).
  Agent-buildable; all gates use committed fixtures.
- Why this is forward work: `lifecycle.cited_adapters` scans ONLY published run bundles under
  `$DATA_DIR/run-eval/*/manifest.json`. Self-improvement `state.json`, campaign
  `campaign.progress.jsonl`, and both `report.md` files also cite `adapter_dir` paths, and those
  citations are invisible to GC. `llb gc-adapters` can therefore delete a superseded adapter whose
  directory a campaign report still links, leaving a dangling path in durable evidence -- exactly
  the failure the citation guard exists to prevent, one directory up.
- User-visible outcome: GC refuses to delete an adapter cited by any durable artifact, not just a
  published run bundle, and names the citing artifact in the refusal reason.
- Scope boundary: in scope -- extend `cited_adapters` to additionally scan
  `$DATA_DIR/self-improve/*/state.json` (`rounds[].adapter_dir`) and
  `$DATA_DIR/finetune-campaign/*/campaign.progress.jsonl` (`entry.adapter_dir`), resolving each path
  through the registry's `adapter_dir` index the way the `adapter_path` match already does; carry the
  citing artifact kind into `GcDecision.cited_by`. Out of scope -- rewriting orchestrator journals to
  store adapter ids instead of paths (a separate migration), scanning arbitrary operator files.
- Data and artifact paths: no new artifact; `gc_rows` gains the citing-artifact kind.
- Execution path: `llb gc-adapters --dry-run`; unit tests -- a superseded adapter cited only by a
  campaign journal is refused, and `--force` still deletes it.
- Acceptance gates: `make ci` green; a campaign-journal-only citation blocks an unforced GC
  (unit-tested against a committed journal fixture); the refusal message names the journal.
- Documentation target: [extended workflows](current/extended-workflows.md) adapter registry and
  lifecycle.

### adapter-staleness-retrieval-fingerprint (optional)

- Agent status: **CLEAR** -- deterministic, fixtures only; no run evidence, no human gate.
- Dependencies: the shipped staleness check (see
  [extended workflows](current/extended-workflows.md) adapter registry and lifecycle) and the RAG
  store meta (`store_meta.json`; see [RAG core](current/rag-core.md)). Agent-buildable,
  deterministic.
- Why this is forward work: staleness compares the goldset digest and the CORPUS fingerprint, but an
  adapter is trained on retrieved CONTEXT, which also depends on the embedder, chunk strategy, and
  retrieval mode. Re-embedding the same corpus with a different `embedding_model`, or rechunking it,
  leaves `corpus_fingerprint` unchanged, so an adapter whose training contexts no longer exist still
  reads `current`. The staleness stamp is therefore weaker than it appears.
- User-visible outcome: an adapter also goes `stale` when the RAG store that produced its training
  contexts was rebuilt with a different embedder, chunker, or retrieval mode, with the changed knob
  named in the reason.
- Scope boundary: in scope -- record the store's retrieval fingerprint (embedder, strategy, chunk
  size/overlap, retrieval mode) from `store_meta.json` in the registry entry at registration, and add
  a third comparison to `staleness()` with a per-knob reason. Out of scope -- rebuilding the store,
  changing `corpus_fingerprint`, retraining on staleness (report only).
- Data and artifact paths: an additive `retrieval_fingerprint` field on registry entries; older
  entries lacking it report `unknown` on that axis, never `current`.
- Execution path: `llb list-adapters`; unit tests -- an entry registered against one embedder flips
  to `stale` when the store meta names another, and a legacy entry without the field reports
  `unknown`.
- Acceptance gates: `make ci` green; the embedder swap flips the verdict and names the knob; a
  registry entry predating the field never reads `current` on the retrieval axis.
- Documentation target: [extended workflows](current/extended-workflows.md) adapter registry and
  lifecycle.

### finetune-hparams-stratified-dev-slice

- Agent status: **CLEAR** -- fixtures only; no run evidence, no human gate.
- Dependencies: the shipped budgeted LoRA search (see
  [extended workflows](current/extended-workflows.md) hyperparameter search). Agent-buildable; all
  gates use committed fixtures.
- Why this is forward work: `carve_dev_slice` draws the held-out sub-slice UNIFORMLY at random from
  the tuning item ids. On a corpus where the base model answers only a minority of items, a uniform
  slice can land almost entirely on items it scores 0.0 on, and the objective becomes a
  near-constant that ranks every trial the same. The first CUDA search on this repo hit exactly
  that: a 12-item dataset produced a 3-item dev slice holding ONE item the base model could answer,
  and every trial tied at 0.0000. The workaround was a bigger dataset, not a better slice.
- User-visible outcome: the dev slice is stratified so it carries a representative share of items
  the base model answers, making the trial objective discriminate at small dev sizes;
  `hparams_manifest.json` records the strata and the base-model score distribution the slice was
  drawn against.
- Scope boundary: in scope -- an optional `--stratify-by-base-score <tuning-run-dir>` that buckets
  tuning items by their base-model `objective_score` from a scored run bundle and draws the dev
  slice proportionally per bucket, keeping the train/dev disjointness and seeded determinism the
  current slice guarantees; a refusal (or loud warning) when the drawn dev slice has zero answerable
  items, because a study cannot rank trials against a constant objective. Out of scope -- changing
  the default uniform slice when no run bundle is supplied, a learned slice selector, changing the
  objective metric.
- Data and artifact paths: an additive `dev_slice.strata` block in `hparams_manifest.json`; no new
  artifact.
- Execution path: `llb finetune-hparams --stratify-by-base-score <tuning-run>`; unit tests -- a
  synthetic score distribution with 3 answerable of 12 items yields a dev slice holding at least one
  answerable item at every seed, disjointness still holds, and an all-zero slice is refused.
- Acceptance gates: `make ci` green; the stratified slice beats the uniform slice on
  answerable-item coverage over a committed score fixture across seeds; the zero-signal refusal is
  unit-tested.
- Documentation target: [extended workflows](current/extended-workflows.md) hyperparameter search.

### finetune-hparams-infeasible-point-prune (optional)

- Agent status: **CLEAR** -- deterministic, fixtures only; no run evidence, no human gate.
- Dependencies: the shipped budgeted LoRA search and the memory planner
  (`src/llb/backends/planner.py`; see
  [robust backends and ontology drafting](current/robustness-ontology-backends.md) memory planner).
  Agent-buildable, deterministic.
- Why this is forward work: `optimize/tuner.py` prunes over-context RAG points BEFORE a trial runs,
  so a doomed configuration never costs a run. The LoRA search has no analogous pre-run prune: it
  only prunes on a MEASURED OOM, after the trial has already paid for a full fine-tune plus a
  backend launch. On a constrained host a large rank crossed with the widest target-module preset
  can be known-infeasible up front.
- User-visible outcome: a trial whose adapter cannot fit the host's VRAM alongside the base model is
  pruned before training starts, with the estimated footprint in the prune reason, so a bounded
  budget spends its trials on configurations that can actually run.
- Scope boundary: in scope -- an adapter-parameter estimate (rank x target modules x layer count)
  fed through the existing planner's VRAM headroom, raising `optuna.TrialPruned` from the objective
  before `trainer_fn` is called; the estimate recorded per trial in `hparams_manifest.json`. Out of
  scope -- replacing the measured-OOM prune (both are needed), calibrating the estimate against a
  benchmark, a second planner.
- Data and artifact paths: an additive `estimated_adapter_mib` per trial record; no new artifact.
- Execution path: `llb finetune-hparams --max-trials 8` on a small-VRAM host; unit tests -- a
  rank-64 point on a fixture host with no headroom prunes before the trainer is invoked, and a
  rank-8 point still trains.
- Acceptance gates: `make ci` green; the pre-run prune fires without calling the injected trainer
  (unit-tested); a pruned trial still leaves a manifest row naming the estimated footprint.
- Documentation target: [extended workflows](current/extended-workflows.md) hyperparameter search.

### finetune-hparams-effective-batch-axis (optional)

- Agent status: **RUN NEEDED** -- code and fake-trainer CI gates are CLEAR; acceptance also needs
  one real bounded search, agent-executed on the current CUDA host. No human gate.
- Dependencies: the shipped budgeted LoRA search. Agent-buildable; the real search runs on the CUDA
  host.
- Why this is forward work: the search space covers rank, alpha, dropout, learning rate, epochs, and
  target modules, but `per_device_train_batch_size`, `gradient_accumulation_steps`, and `max_length`
  stay pinned at the trainer's conservative defaults. Effective batch size interacts strongly with
  learning rate, so a recorded best learning rate is only best AT the pinned batch size -- an
  operator who changes the batch size silently invalidates the searched config.
- User-visible outcome: the study searches effective batch size beside learning rate, so the
  recorded best config is self-consistent, and `hparams_manifest.json` records the batch geometry
  the learning rate was chosen under.
- Scope boundary: in scope -- add `per_device_train_batch_size` x `gradient_accumulation_steps`
  (sampled as an effective-batch categorical so the two are never independently meaningless) and
  `max_length` to `suggest_lora_hyperparameters`; the trainer already consumes all three keys. Out
  of scope -- a learning-rate schedule search, gradient checkpointing, a second optimizer.
- Data and artifact paths: additive keys in the sampled hyperparameters; no new artifact.
- Execution path: `llb finetune-hparams --max-trials 12`; unit tests -- the sampled effective batch
  is always the product of the two knobs, and a seeded study still reproduces its trial table.
- Acceptance gates: `make ci` green with the fake trainer; the effective-batch invariant is
  unit-tested; one real bounded search on the CUDA host records whether the widened space beats the
  pinned-batch best config (no-gain is acceptable evidence).
- Documentation target: [extended workflows](current/extended-workflows.md) hyperparameter search.

### 23. compressed-qat-adapter-support (optional)

- Agent status: **RUN NEEDED** -- code and fixture gates are CLEAR; acceptance also needs one
  CUDA-host trainability probe, agent-executed on the current machine. No human gate.
- Dependencies: follows the baseline trainer path in
  [extended workflows](current/extended-workflows.md); registered adapter provenance is available
  through the shipped adapter registry.
- User-visible outcome: compressed-tensors QAT checkpoints can participate in adapter campaigns
  instead of serving only as base models: the trainer detects compressed linear modules, chooses a
  compatible adapter-injection strategy or an explicit skip reason, and reports whether the
  checkpoint is trainable on the host without crashing mid-campaign.
- Scope boundary: in scope -- model-introspection helpers for native quantization configs,
  per-architecture target-module selection, a compatibility shim or documented fallback for PEFT
  injection into compressed linear layers, and campaign skip/report plumbing that names the exact
  trainability blocker. Out of scope -- dequantizing full checkpoints into new base weights,
  uploading converted models, or adding a second training framework before the PEFT path is
  exhausted.
- Data and artifact paths: compatibility probes live under
  `$DATA_DIR/finetune-compat/<model>/<timestamp>/`; campaign skip reasons stay in
  `campaign.progress.jsonl` and `report.md`.
- Execution path: `llb finetune-compat --model <m> --backend <b>` plus campaign auto-probing for
  native compressed checkpoints; unit tests use fake compressed linear modules and a fake trainer.
- Acceptance gates: `make ci` green; compressed native-quant fixtures either receive a working
  adapter or a deterministic skip reason before training starts; one CUDA-host probe records the
  trainable/not-trainable verdict for a compressed QAT checkpoint in current docs.
- Documentation target: [extended workflows](current/extended-workflows.md); the
  self-improvement-loop guide's compatibility notes.

### embedding-bakeoff-full-corpus

- Agent status: **RUN NEEDED** -- no new code; the deliverable is heavy deterministic full-corpus
  store builds + the bake-off run, agent-executed on the current CUDA host. No human gate.
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

- Agent status: **RUN NEEDED** -- no new code; heavy deterministic full-corpus comparison run,
  agent-executed on the current CUDA host. No human gate.
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

- Agent status: **RUN NEEDED** -- no new code; heavy deterministic full-corpus comparison + sweep,
  agent-executed on the current CUDA host. No human gate.
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

- Agent status: **RUN NEEDED** -- no new code; heavy deterministic rerank comparison + per-model
  position probes, agent-executed on the current CUDA host. No human gate.
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

### verify-sample-exact-allocation (optional)

- Agent status: **CLEAR** -- fixtures only; no run evidence, no human gate.
- Dependencies: the shipped stratified sampler (`draw_stratified_sample` in
  `src/llb/goldset/verify.py`; see [data prep](current/data-prep.md) verification gate).
- Why this is forward work: the sampler trims when proportional quotas overshoot `n` but never
  tops up when rounding undershoots, so `verify-sample VERIFY_N=40` can emit a 39-row worksheet
  (the quickstart-draft review hit exactly this; the operator had to merge-enlarge with a larger
  `n` to cross the target). Related sizing fact: at tolerance 0.05 a stratum needs >= 20 decided
  rows to absorb a single reject, so under-filled cells guarantee advisory per-stratum FAIL
  warnings on any reject.
- User-visible outcome: `verify-sample` draws exactly `min(n, population)` rows, so a requested
  40-item review is a 40-item review; the sample manifest records the final per-stratum
  allocation.
- Scope boundary: in scope -- a deterministic largest-remainder top-up in
  `draw_stratified_sample` distributing the rounding shortfall across strata while keeping the
  per-stratum floor of one and seeded reproducibility. Out of scope -- changing the floor-of-one
  rule, the acceptance arithmetic, or the merge lane.
- Data and artifact paths: no new artifact; `sample_manifest.json` already records strata sizes.
- Execution path: `make verify-sample BUNDLE=<draft> VERIFY_N=<n>`; unit tests -- a population
  whose proportional rounding undershoots today yields exactly `n` rows at every seed, and a
  seeded draw stays reproducible.
- Acceptance gates: `make ci` green; the exact-`n` draw is unit-tested against an undershooting
  fixture; existing determinism tests still pass.
- Documentation target: [data prep](current/data-prep.md) verification gate.

### draft-feedback-rejection-reasons (optional)

- Agent status: **CLEAR** -- fixtures only; no run evidence, no human gate.
- Dependencies: the shipped coded-rejection export (`rejection_reasons.json`; see
  [data prep](current/data-prep.md) reviewer throughput tooling).
- Why this is forward work: the verify gate exports WHY items were rejected, but the drafting
  pipeline never reads it -- an operator re-drafting after a failed acceptance gets the same
  prompts that produced the rejected items, so the feedback loop currently ends at a JSON file.
- User-visible outcome: `prepare-goldset-draft` accepts a rejection-feedback file and tightens the
  draft prompts per dominant reject code (e.g. a `circular`-heavy summary adds an explicit
  non-circularity instruction with a rejected example), with the applied feedback recorded in
  bundle provenance.
- Scope boundary: in scope -- a deterministic reject-code-to-prompt-hint mapper in the ontology
  draft stage reusing the closed reject-code set; provenance records the applied hints and the
  feedback file digest. Out of scope -- a learned prompt optimizer, changing the reject-code set,
  automatic re-drafting.
- Data and artifact paths: no new artifact; `provenance.json` gains an applied-feedback block.
- Execution path:
  `make prepare-goldset-draft DRAFT_REJECTION_FEEDBACK=<bundle>/accepted/rejection_reasons.json`;
  unit tests -- each reject code maps to a deterministic hint, and an empty summary is a no-op.
- Acceptance gates: `make ci` green; the hint mapping is unit-tested per code; provenance names
  the feedback source.
- Documentation target: [data prep](current/data-prep.md).

### external-import-needle-parity (optional)

- Agent status: **CLEAR** -- fixtures only (committed `samples/external-drafts` + fake retriever);
  no run evidence, no human gate.
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
completion. Task 7 also gates agent work: finish its human review before agent task 8's
acceptance run.

Recommended order for the human steps once the agent has pre-built each task's code: task 7's
chain review first (it is the only human step blocking agent work -- task 8's acceptance run);
then task 1's coverage-vs-cap review and task 5's derived-case review (both consume the same
verify-gate muscle over agent-prepared bundles); task 2's egress consent + spend decision is
independent and can happen whenever the operator is ready to authorize it.

### 1. draft-yield-quality-max -- residual empirical acceptance

- Agent status: **HUMAN-GATED** -- the coverage-vs-cap accept-rate evidence needs a human
  `verify-sample` review pass; the optional multi-hop answer hardening is CLEAR (agent-buildable
  now). The agent can run both draft passes on the current CUDA host so only the review gates.
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

- Agent status: **HUMAN-GATED** -- human egress consent + API spend authorization gate the real
  2-document frontier probe; all code and fake-completer tests are CLEAR (agent-buildable now, no
  network).
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

### 5. security-corpus-probes

- Agent status: **HUMAN-GATED** -- derived cases must clear the human verify gate before any
  headline/composite use; the generator, unit tests, and the `bench-security` run itself are CLEAR
  / agent-executable on the current CUDA host.
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

- Agent status: **HUMAN-GATED** -- >= 10 chains must be human-reviewed and accepted into the
  committed fixture; the schema, drafting, and validation code are CLEAR (agent-buildable now).
  This human step is the single upstream blocker of agent task 8's acceptance run.
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
